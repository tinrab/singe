use std::{
    collections::HashMap,
    error::Error as StdError,
    fs,
    path::{Path, PathBuf},
    process,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

use crate::{
    backend::{
        cuda::{
            CudaEvent, CudaStream, EventTiming, clear_cache,
            export::{
                BufferKind, CudaExport, CudaProgram, ProgramArtifact,
                compiled::{CompiledProgram, ProgramInstance},
            },
            tuning::{
                comparison::Comparison,
                library::LibraryRecorder,
                plan::{MeasureError, Measured, PlanRunner, nanoseconds},
                records::{Measurement, TunedChoice, TuningEnvironment, TuningRecordSet},
                report::{Attempt, CandidateFailure, FailureStage, SelfComparison, TimeSpent},
                small_graph::{Judging, SmallGraph, judged, restore_written},
                tune::GraphTuneOptions,
            },
        },
        registry::CudaGraphProvider,
    },
    device::Device,
    error::Error,
    graph::{GraphCudaOptions, GraphExecutor, GraphPrepareOptions},
    tensor::Tensor,
};

// How many built programs a runner keeps, each with a host copy of its weights: enough for the best plan and the one it is being compared with.
const BUILT_PROGRAMS: usize = 2;

// How many runs a candidate takes on its turn in a round, of which only the last is timed, so that it is not timed on what the other left in the caches.
const TURN_RUNS: usize = 2;

// How many rounds of a comparison each of `GraphTuneOptions::runs` buys, over all its pairs of instances, since a choice rests on these more than on any other timing.
const ROUNDS_PER_RUN: usize = 4;

// Whether the baseline is made first, for each fresh pair of instances a comparison runs over.
// This order has no period of two, since an allocator reusing memory in a fixed order could otherwise give each side the same place in every pair, which would then look like a difference between them.
const BASELINE_MADE_FIRST: &[bool] = &[true, false, false, true, false];

// How many pairs of blocks two candidates that do not fit together alternate over.
const BLOCK_PAIRS: usize = 6;

fn median(mut samples: Vec<Duration>) -> Option<Duration> {
    samples.sort_unstable();
    samples.get(samples.len() / 2).copied()
}

fn measured_median(samples: Vec<Duration>) -> Attempt<Duration> {
    median(samples).ok_or_else(|| CandidateFailure::at(FailureStage::Run, "no times"))
}

fn refuse(bytes: u64, refuse_over: Option<u64>) -> Attempt<(), MeasureError> {
    match refuse_over {
        Some(limit) if bytes > limit => Err(MeasureError::OverMemory(bytes)),
        _ => Ok(()),
    }
}

fn prepare_replay(
    small: &SmallGraph,
    records: &TuningRecordSet,
    device: Device,
    options: &GraphTuneOptions,
) -> Attempt<GraphExecutor, MeasureError> {
    let cuda = GraphCudaOptions::new()
        .with_lowering_policy(options.policy.clone())
        .with_tuning_records(records.clone())
        .with_inputs_in_place(
            small
                .inputs
                .iter()
                .map(|(name, tensor)| (name.clone(), tensor.clone())),
        );
    small
        .graph
        .prepare(GraphPrepareOptions::new(device).with_cuda_options(cuda))
        .map_err(|error| {
            // A library that refuses what the records chose leaves nothing timed, which is how a candidate learns that one is refused.
            if refused_by_library(&error) {
                MeasureError::Refused
            } else {
                CandidateFailure::from_error(FailureStage::Prepare, &error).into()
            }
        })
}

// Whether `error` comes of a library refusing what a candidate's records had it run, a way or a configuration.
fn refused_by_library(error: &(dyn StdError + 'static)) -> bool {
    let mut cause = Some(error);
    while let Some(error) = cause {
        if matches!(
            error.downcast_ref::<Error>(),
            Some(Error::ConfigurationRefused { .. } | Error::LibraryRefused { .. })
        ) {
            return true;
        }

        cause = error.source();
    }
    false
}

// What the libraries `records` send a small graph to could run it with instead, as compiling it lists them.
// A graph that does not compile lists nothing, and running it reports why.
pub fn listed_library_choices(
    small: &SmallGraph,
    records: &TuningRecordSet,
    device: Device,
    options: &GraphTuneOptions,
    key: &str,
    provider: CudaGraphProvider,
) -> Vec<TunedChoice> {
    let recorder = LibraryRecorder::default();
    let cuda = GraphCudaOptions::new()
        .with_lowering_policy(options.policy.clone())
        .with_tuning_records(records.clone().measured_as(Measurement::GraphReplay))
        .with_library_recorder(recorder.clone())
        .with_inputs_in_place(
            small
                .inputs
                .iter()
                .map(|(name, tensor)| (name.clone(), tensor.clone())),
        );
    if small
        .graph
        .prepare(GraphPrepareOptions::new(device).with_cuda_options(cuda))
        .is_err()
    {
        return Vec::new();
    }
    recorder.take(key, provider).unwrap_or_default()
}

// A candidate ready to time, holding what it runs so that several can take turns.
struct Prepared<'a> {
    agrees: bool,
    bytes: u64,
    runs: RunTarget<'a>,
}

enum RunTarget<'a> {
    Replay {
        executor: Box<GraphExecutor>,
        inputs: &'a HashMap<String, Tensor>,
        stream: CudaStream,
        start: CudaEvent,
        end: CudaEvent,
    },
    Program {
        instance: ProgramInstance,
        operands: Vec<Tensor>,
    },
}

impl Prepared<'_> {
    // How long each of `runs` runs took.
    fn timed(&mut self, runs: usize) -> Attempt<Vec<Duration>> {
        match &mut self.runs {
            RunTarget::Replay {
                executor,
                inputs,
                stream,
                start,
                end,
            } => {
                let failed = |error: &Error| CandidateFailure::from_error(FailureStage::Run, error);
                let mut samples = Vec::with_capacity(runs);
                for _ in 0..runs {
                    // With the host held out of what the events span, a busy host cannot hide what differs between candidates.
                    let hold = stream.hold().map_err(|error| failed(&error))?;
                    start.record(stream).map_err(|error| failed(&error))?;
                    let (_, written_back) = executor
                        .run_on_then(inputs, stream, |stream| {
                            let recorded = end.record(stream);
                            hold.release();
                            recorded
                        })
                        .map_err(|error| failed(&error))?;
                    end.synchronize().map_err(|error| failed(&error))?;
                    if hold.overran() {
                        return Err(CandidateFailure::at(
                            FailureStage::Run,
                            "replay waited for a stream held until it was enqueued",
                        ));
                    }

                    samples.push(
                        end.elapsed_since(start).map_err(|error| failed(&error))? + written_back,
                    );
                }
                Ok(samples)
            }
            RunTarget::Program { instance, operands } => instance
                .timed_runs(operands, runs)
                .map_err(|error| CandidateFailure::from_error(FailureStage::Run, &error)),
        }
    }

    // One turn in a round.
    fn turn(&mut self) -> Attempt<Duration> {
        self.timed(TURN_RUNS)?
            .pop()
            .ok_or_else(|| CandidateFailure::at(FailureStage::Run, "no times"))
    }
}

// Prepares and times candidates for one tuning step, keeping what it has built.
pub struct Measurer<'a> {
    device: Device,
    options: &'a GraphTuneOptions,
    // What it built by rendered source, most recently used last, with what stopped those that failed.
    built: Vec<(String, Attempt<CompiledProgram>)>,
    // How many builds it holds at most.
    held: usize,
    // Whether a program's forced configurations come from a table, as they do for a choice point's small graph, and whether one a library refuses is then not timed.
    library_table: bool,
    spent: TimeSpent,
}

impl<'a> Measurer<'a> {
    pub const fn new(device: Device, options: &'a GraphTuneOptions) -> Self {
        Self {
            device,
            options,
            built: Vec::new(),
            held: BUILT_PROGRAMS,
            library_table: false,
            spent: TimeSpent {
                total_ns: 0,
                export_ns: 0,
                build_ns: 0,
                create_ns: 0,
                run_ns: 0,
                check_ns: 0,
            },
        }
    }

    // A measurer for choice points, whose candidates share a small graph and differ in little.
    pub const fn for_choice_points(device: Device, options: &'a GraphTuneOptions) -> Self {
        let mut measurer = Self::new(device, options);
        measurer.held = usize::MAX;
        measurer.library_table = true;
        measurer
    }

    pub const fn spent(&self) -> &TimeSpent {
        &self.spent
    }

    const fn is_program(records: &TuningRecordSet) -> bool {
        matches!(
            records.environment(),
            Some(TuningEnvironment {
                measurement: Measurement::Program,
                ..
            })
        )
    }

    fn exported(
        &mut self,
        small: &SmallGraph,
        records: &TuningRecordSet,
    ) -> Attempt<(CudaProgram, ProgramArtifact)> {
        let started = Instant::now();
        let exported = export_program(small, records, self.options, self.library_table);
        let exported = exported.and_then(|program| {
            let files = program
                .render("model")
                .map_err(|error| CandidateFailure::from_error(FailureStage::Export, &error))?;
            Ok((program, files))
        });
        self.spent.export_ns += nanoseconds(started.elapsed());
        exported
    }

    fn remember(&mut self, source: String, built: Attempt<CompiledProgram>) {
        if self.built.len() >= self.held {
            drop(self.built.remove(0));
        }

        self.built.push((source, built));
    }

    // Builds at once what the candidates of a choice point need, so no build runs beside a timing, and forgets what the last one built.
    pub fn build_ahead(&mut self, small: &SmallGraph, candidates: &[TuningRecordSet]) {
        self.built.clear();
        if !candidates.first().is_some_and(Self::is_program) {
            return;
        }

        let mut pending = Vec::new();
        let mut sources = Vec::new();
        for records in candidates {
            // What cannot export says so when it is timed.
            let Ok((program, files)) = self.exported(small, records) else {
                continue;
            };
            let source = source_of(&files);
            if self
                .options
                .memory_limit
                .is_some_and(|limit| program_bytes(&program) > limit)
                || sources.contains(&source)
            {
                continue;
            }

            sources.push(source);
            pending.push((program, files, candidate_directory(self.options)));
        }

        let started = Instant::now();
        let built = CompiledProgram::create_many(pending, self.options.build_jobs);
        self.spent.build_ns += nanoseconds(started.elapsed());
        for (source, built) in sources.into_iter().zip(built) {
            let built =
                built.map_err(|error| CandidateFailure::from_error(FailureStage::Build, &error));
            self.built.push((source, built));
        }
    }

    fn compiled(
        &mut self,
        program: &CudaProgram,
        files: &ProgramArtifact,
    ) -> Attempt<CompiledProgram> {
        let source = source_of(files);
        if let Some(index) = self.built.iter().position(|(built, _)| *built == source) {
            let entry = self.built.remove(index);
            let compiled = entry.1.clone();
            self.built.push(entry);
            return compiled;
        }

        let started = Instant::now();
        let compiled = build_program(program, self.options);
        self.spent.build_ns += nanoseconds(started.elapsed());
        self.remember(source, compiled.clone());
        compiled
    }

    fn prepare<'s>(
        &mut self,
        small: &'s SmallGraph,
        records: &TuningRecordSet,
        refuse_over: Option<u64>,
        judging: Judging,
    ) -> Attempt<Prepared<'s>, MeasureError> {
        restore_written(small)?;
        if Self::is_program(records) {
            let (program, files) = self.exported(small, records)?;
            let bytes = program_bytes(&program);
            refuse(bytes, refuse_over)?;
            let compiled = self.compiled(&program, &files)?;
            return self.prepare_program(small, &program, &files, &compiled, bytes, judging);
        }
        self.prepare_replay(small, records, refuse_over, judging)
    }

    fn prepare_replay<'s>(
        &mut self,
        small: &'s SmallGraph,
        records: &TuningRecordSet,
        refuse_over: Option<u64>,
        judging: Judging,
    ) -> Attempt<Prepared<'s>, MeasureError> {
        let failed = |error: &Error| CandidateFailure::from_error(FailureStage::Run, error);
        let started = Instant::now();
        let mut executor = prepare_replay(small, records, self.device, self.options)?;
        let memory = executor
            .memory_stats()
            .map_err(|error| CandidateFailure::from_error(FailureStage::Prepare, &error))?;
        let bytes = u64::try_from(
            memory.active_workspace_bytes() + memory.active_replay_allocation_bytes(),
        )
        .unwrap_or(u64::MAX);
        refuse(bytes, refuse_over)?;
        let stream = CudaStream::create(self.device).map_err(|error| failed(&error))?;
        let start = CudaEvent::create(self.device, EventTiming::Recorded)
            .map_err(|error| failed(&error))?;
        let end = CudaEvent::create(self.device, EventTiming::Recorded)
            .map_err(|error| failed(&error))?;
        self.spent.create_ns += nanoseconds(started.elapsed());
        // The first run warms the kernels and caches, and what it leaves is what this candidate computes.
        let started = Instant::now();
        let outputs = executor
            .run_on(&small.inputs, &stream)
            .map_err(|error| failed(&error))?;
        stream.synchronize().map_err(|error| failed(&error))?;
        self.spent.run_ns += nanoseconds(started.elapsed());
        let started = Instant::now();
        let agrees = judged(small, judging, outputs, || {
            let outputs = executor
                .run_on(&small.inputs, &stream)
                .map_err(|error| failed(&error))?;
            stream.synchronize().map_err(|error| failed(&error))?;
            Ok(outputs)
        })?;
        self.spent.check_ns += nanoseconds(started.elapsed());
        Ok(Prepared {
            agrees,
            bytes,
            runs: RunTarget::Replay {
                executor: Box::new(executor),
                inputs: &small.inputs,
                stream,
                start,
                end,
            },
        })
    }

    fn prepare_program<'s>(
        &mut self,
        small: &'s SmallGraph,
        program: &CudaProgram,
        files: &ProgramArtifact,
        compiled: &CompiledProgram,
        bytes: u64,
        judging: Judging,
    ) -> Attempt<Prepared<'s>, MeasureError> {
        let failed = |error: &(dyn StdError + 'static)| {
            CandidateFailure::from_error(FailureStage::Run, error)
        };
        let started = Instant::now();
        let instance = compiled.instance_with_table(&files.library_table);
        self.spent.create_ns += nanoseconds(started.elapsed());
        // A program fails to start when a library refuses what its records had it run, which is how a candidate learns that one is refused.
        let instance = match instance {
            Ok(instance) => instance,
            Err(error) if error.is_library_refusal() => return Err(MeasureError::Refused),
            Err(error) => return Err(failed(&error).into()),
        };

        let operands = program
            .inputs
            .iter()
            .map(|input| {
                small.inputs.get(&input.graph_name).cloned().ok_or_else(|| {
                    CandidateFailure::at(
                        FailureStage::Run,
                        format!("no input {}", input.graph_name),
                    )
                })
            })
            .collect::<Attempt<Vec<_>, _>>()?;
        let started = Instant::now();
        let outputs = instance.run(&operands, 1).map_err(|error| failed(&error))?;
        self.spent.run_ns += nanoseconds(started.elapsed());
        let started = Instant::now();
        let agrees = judged(small, judging, outputs, || {
            instance.run(&operands, 1).map_err(|error| failed(&error))
        })?;
        self.spent.check_ns += nanoseconds(started.elapsed());
        Ok(Prepared {
            agrees,
            bytes,
            runs: RunTarget::Program { instance, operands },
        })
    }

    // Runs `small` under `records` the way they will be read, and judges what it computes.
    // A plan over `refuse_over` bytes is not timed, and one whose first time passes `abandon_above` has only that time.
    //
    // # Errors
    //
    // Returns what stopped the candidate, or that it does not fit.
    pub fn measure(
        &mut self,
        small: &SmallGraph,
        records: &TuningRecordSet,
        refuse_over: Option<u64>,
        abandon_above: Option<Duration>,
    ) -> Attempt<Measured, MeasureError> {
        let mut prepared = self.prepare(small, records, refuse_over, Judging::Compare)?;
        let started = Instant::now();
        let mut samples = prepared.timed(1)?;
        let first = measured_median(samples.clone())?;
        let abandoned = abandon_above.is_some_and(|limit| first > limit);
        if !abandoned {
            samples.extend(prepared.timed(self.options.runs.saturating_sub(1))?);
        }

        self.spent.run_ns += nanoseconds(started.elapsed());
        Ok(Measured {
            agrees: prepared.agrees,
            median: measured_median(samples)?,
            bytes: prepared.bytes,
            abandoned,
        })
    }

    // Times `candidate` against `baseline` over fresh pairs of instances, since one instance can differ from another of the same plan for as long as it lives.
    // Within a pair, rounds each take a turn, in alternating order, so what drifts moves both.
    // Two that do not fit together alternate in blocks instead, each made afresh.
    //
    // # Errors
    //
    // Returns what stopped either.
    pub fn compare(
        &mut self,
        small: &SmallGraph,
        baseline: &TuningRecordSet,
        candidate: &TuningRecordSet,
    ) -> Attempt<Comparison, MeasureError> {
        let mut pairs = Vec::with_capacity(BASELINE_MADE_FIRST.len());
        for &made_first in BASELINE_MADE_FIRST {
            let compared = if made_first {
                self.compare_pair(small, baseline, candidate)
            } else {
                self.compare_pair(small, candidate, baseline)
            };
            let Some(compared) = compared? else {
                return self.compare_in_blocks(small, baseline, candidate);
            };

            pairs.push(if made_first {
                compared
            } else {
                compared.inverted()
            });
        }
        Ok(no_rounds(Comparison::across(&pairs))?)
    }

    // One pair of fresh instances in rounds, of `second` over `first`, or `None` if they do not fit together.
    fn compare_pair(
        &mut self,
        small: &SmallGraph,
        first: &TuningRecordSet,
        second: &TuningRecordSet,
    ) -> Attempt<Option<Comparison>, MeasureError> {
        self.free_for_program(first)?;
        let mut first = self.prepare(small, first, None, Judging::Skip)?;
        let Ok(mut second) = self.prepare(small, second, None, Judging::Skip) else {
            return Ok(None);
        };

        let started = Instant::now();
        // Even, so going first falls to each as often.
        let rounds =
            (self.options.runs.max(1) * ROUNDS_PER_RUN / BASELINE_MADE_FIRST.len()).max(2) * 2;
        let mut first_times = Vec::with_capacity(rounds);
        let mut second_times = Vec::with_capacity(rounds);
        for round in 0..rounds {
            if round % 2 == 0 {
                first_times.push(first.turn()?);
                second_times.push(second.turn()?);
            } else {
                second_times.push(second.turn()?);
                first_times.push(first.turn()?);
            }
        }

        self.spent.run_ns += nanoseconds(started.elapsed());
        Ok(Some(no_rounds(Comparison::of_alternating(
            &first_times,
            &second_times,
        ))?))
    }

    pub fn compare_in_blocks(
        &mut self,
        small: &SmallGraph,
        baseline: &TuningRecordSet,
        candidate: &TuningRecordSet,
    ) -> Attempt<Comparison, MeasureError> {
        let runs = (self.options.runs / BLOCK_PAIRS).max(3);
        let mut baseline_times = Vec::with_capacity(BLOCK_PAIRS);
        let mut candidate_times = Vec::with_capacity(BLOCK_PAIRS);
        for pair in 0..BLOCK_PAIRS {
            let order = if pair % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            };
            for is_candidate in order {
                let records = if is_candidate { candidate } else { baseline };
                self.free_for_program(records)?;
                let mut prepared = self.prepare(small, records, None, Judging::Skip)?;
                let started = Instant::now();
                let time = measured_median(prepared.timed(runs)?)?;
                self.spent.run_ns += nanoseconds(started.elapsed());
                if is_candidate {
                    candidate_times.push(time);
                } else {
                    baseline_times.push(time);
                }
            }
        }

        let mut comparison = no_rounds(Comparison::of(&baseline_times, &candidate_times))?;
        comparison.together = false;
        Ok(comparison)
    }

    // A program allocates from the driver, which cannot see what the process's allocator keeps cached.
    fn free_for_program(&self, records: &TuningRecordSet) -> Attempt<()> {
        if Self::is_program(records) {
            clear_cache(self.device)
                .map_err(|error| CandidateFailure::from_error(FailureStage::Run, &error))?;
        }
        Ok(())
    }

    // How far timing alone moves a comparison, by timing one plan against itself, in rounds and in turn.
    //
    // # Errors
    //
    // Returns what stopped the plan.
    pub fn self_comparison(
        &mut self,
        key: &str,
        small: &SmallGraph,
        records: &TuningRecordSet,
    ) -> Attempt<SelfComparison, MeasureError> {
        let in_rounds = self.compare(small, records, records)?;
        let earlier = self.measure(small, records, None, None)?.median;
        let later = self.measure(small, records, None, None)?.median;
        Ok(SelfComparison {
            key: key.to_owned(),
            in_rounds,
            in_turn_ppm: no_rounds(Comparison::of(&[earlier], &[later]))?.median_ppm,
        })
    }
}

fn no_rounds(comparison: Option<Comparison>) -> Attempt<Comparison> {
    comparison.ok_or_else(|| CandidateFailure::at(FailureStage::Run, "no rounds"))
}

fn source_of(files: &ProgramArtifact) -> String {
    format!("{}{}{}", files.header, files.source, files.manifest)
}

pub fn export_program(
    small: &SmallGraph,
    records: &TuningRecordSet,
    options: &GraphTuneOptions,
    library_table: bool,
) -> Attempt<CudaProgram> {
    let environment = records.environment().ok_or_else(|| {
        CandidateFailure::at(FailureStage::Export, "records state no architecture")
    })?;
    let architecture = environment
        .architecture
        .parse()
        .map_err(|error| CandidateFailure::from_error(FailureStage::Export, &error))?;
    CudaExport::new(architecture)
        .with_lowering_policy(options.policy.clone())
        .with_tuning_records(records.clone())
        .with_library_table(library_table)
        .export(&small.graph)
        .map_err(|error| CandidateFailure::from_error(FailureStage::Export, &error))
}

// What a program allocates for itself, without its callers' buffers or its state.
fn program_bytes(program: &CudaProgram) -> u64 {
    program
        .buffers
        .iter()
        .filter(|buffer| matches!(buffer.kind, BufferKind::Internal | BufferKind::Scratch))
        .map(|buffer| buffer.bytes)
        .sum()
}

// Removes what a tuning process that no longer runs left behind, since a killed one cannot clean up after itself.
pub fn remove_abandoned_candidates(build_directory: &Path) {
    // Without `/proc`, nothing here can tell whether a process is still running.
    if !Path::new("/proc/self").exists() {
        return;
    }

    let Ok(entries) = fs::read_dir(build_directory) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let owner = name
            .to_str()
            .and_then(|name| name.strip_prefix("candidate-"))
            .and_then(|rest| rest.split('-').next())
            .filter(|owner| owner.chars().all(|digit| digit.is_ascii_digit()));
        if owner.is_some_and(|owner| !Path::new("/proc").join(owner).exists()) {
            let _ = fs::remove_dir_all(entry.path());
        }
    }
}

fn candidate_directory(options: &GraphTuneOptions) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    options.build_directory.join(format!(
        "candidate-{}-{}",
        process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}

fn build_program(program: &CudaProgram, options: &GraphTuneOptions) -> Attempt<CompiledProgram> {
    CompiledProgram::create_in(program, candidate_directory(options))
        .map_err(|error| CandidateFailure::from_error(FailureStage::Build, &error))
}

// Runs whole plans on a graph, as their records will be read.
pub struct WholeGraphRunner<'a> {
    whole: &'a SmallGraph,
    measurer: Measurer<'a>,
    options: &'a GraphTuneOptions,
    device: Device,
    // What a plan with merged projections holds in joined weights, which no measurement of its run sees.
    merged_bytes: u64,
}

impl<'a> WholeGraphRunner<'a> {
    pub const fn new(whole: &'a SmallGraph, device: Device, options: &'a GraphTuneOptions) -> Self {
        Self {
            whole,
            measurer: Measurer::new(device, options),
            options,
            device,
            merged_bytes: 0,
        }
    }

    pub const fn with_merged_bytes(mut self, merged_bytes: u64) -> Self {
        self.merged_bytes = merged_bytes;
        self
    }

    pub const fn spent(&self) -> &TimeSpent {
        self.measurer.spent()
    }

    // How far timing alone moves a comparison of this plan.
    //
    // # Errors
    //
    // Returns what stopped the plan.
    pub fn self_comparison(
        &mut self,
        key: &str,
        records: &TuningRecordSet,
    ) -> Attempt<SelfComparison> {
        self.measurer
            .self_comparison(key, self.whole, records)
            .map_err(whole_plan_failure)
    }
}

// A whole plan runs with no limit and no table, so neither of those can stop it.
fn whole_plan_failure(error: MeasureError) -> CandidateFailure {
    match error {
        MeasureError::Failed(failure) => failure,
        MeasureError::OverMemory(bytes) => CandidateFailure::at(
            FailureStage::Run,
            format!("refused {bytes} bytes with no limit"),
        ),
        MeasureError::Refused => CandidateFailure::at(FailureStage::Run, "refused with no table"),
    }
}

impl PlanRunner for WholeGraphRunner<'_> {
    fn fingerprint(&mut self, records: &TuningRecordSet) -> Attempt<String> {
        if Measurer::is_program(records) {
            let program = export_program(self.whole, records, self.options, false)?;
            let files = program
                .render("model")
                .map_err(|error| CandidateFailure::from_error(FailureStage::Export, &error))?;
            return Ok(files.header + &files.source);
        }
        prepare_replay(self.whole, records, self.device, self.options)
            .map_err(whole_plan_failure)?
            .plan_identity()
            .map_err(|error| CandidateFailure::from_error(FailureStage::Prepare, &error))
    }

    fn run(
        &mut self,
        records: &TuningRecordSet,
        abandon_above: Option<Duration>,
    ) -> Attempt<Measured> {
        self.measurer.free_for_program(records)?;
        let mut measured = self
            .measurer
            .measure(self.whole, records, None, abandon_above)
            .map_err(whole_plan_failure)?;
        if records.merges_projections() {
            measured.bytes += self.merged_bytes;
        }
        Ok(measured)
    }

    fn compare(
        &mut self,
        baseline: &TuningRecordSet,
        candidate: &TuningRecordSet,
    ) -> Attempt<Comparison> {
        self.measurer
            .compare(self.whole, baseline, candidate)
            .map_err(whole_plan_failure)
    }
}
