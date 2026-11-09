use std::{
    collections::{HashMap, HashSet, VecDeque},
    env,
    hash::BuildHasher,
    iter,
    path::PathBuf,
    thread,
    time::{Duration, Instant},
};

use crate::{
    backend::{
        cuda::{
            codegen::{KernelRecorder, PlanningContext},
            compiled::{
                build::prepare::channels_last_convolution_chains,
                plan::{
                    fusion::{FusionPlan, GroupKind, ModuleKind, plan_fusion},
                    group::module_parameters,
                    kernel::KernelFeasibility,
                    lowering::{CudaGraphProvider, LoweringPlan, LoweringPolicy},
                },
            },
            graph_binding::{InPlaceWrites, WritableInputs, graph_users},
            kernel_plan::{KernelInput, KernelParameters, KernelPlanner},
            kernel_tuning::{self, Outcome},
            runtime::runtime_for_device,
            tuning::{
                comparison::Comparison,
                keys::{
                    dependent_launch_key, fused_attention_key, group_key, library_key,
                    merged_projections_key, schedule_key, way_key,
                },
                measure::{
                    Evaluated, Measurer, WholeGraphRunner, evaluate, listed_library_choices,
                    remove_abandoned_candidates, restore_written, subgraph, whole_graph,
                },
                plan::{
                    Alternative, Candidate, ChoicePoint, MeasureError, Plan, PlanLimits,
                    PlanRunner, PlanSearchOptions, check_plan, confirm, decide, decide_in_two,
                    plan_records, search_plans,
                },
                records::{
                    Measurement, TunedChoice, TunedRecord, TunedWay, TuningEnvironment,
                    TuningRecords,
                },
                report::{
                    CandidateFailure, CandidateOutcome, CandidateReport, ChoiceReport,
                    ComparisonReport, FailureStage, SelfComparison, TuningReport,
                },
            },
        },
        registry::Operation,
    },
    device::Device,
    error::{Error, Result},
    graph::{
        Graph, GraphCompileOptions, GraphCudaOptions, GraphPrepareOptions,
        ir::{Graph as IrGraph, ValueId},
        merged_projection_bytes,
    },
    tensor::Tensor,
};

// How many choice points one move of the search of whole plans changes.
const PLAN_MUTATIONS: usize = 2;

// How the tuning step runs.
#[derive(Debug, Clone)]
pub struct GraphTuneOptions {
    // The providers and libraries candidates may use.
    pub policy: LoweringPolicy,
    // The time after which a choice point times no more candidates, which does not count building them.
    pub choice_budget: Duration,
    // How many replays each candidate is timed over.
    pub runs: usize,
    // Whether generated kernels' schedules are searched too, once the other choices are made.
    pub schedules: bool,
    // How candidates are timed, which decides what can read the records: graphs, or programs.
    pub measurement: Measurement,
    // Where `Measurement::Program` builds its candidates.
    pub build_directory: PathBuf,
    // The most a whole plan may allocate for itself, without inputs, outputs, state, driver resources, or libraries' internal memory.
    pub memory_limit: Option<u64>,
    // The time to search whole plans for, once each choice point has a winner; zero searches nothing.
    pub plan_budget: Duration,
    // What the search of whole plans draws its moves from, so one seed gives one sequence of moves.
    pub plan_seed: u64,
    // Whether a whole plan slower than the path with no records loses its records.
    pub plan_speed: bool,
    // How many programs compile at once under `Measurement::Program`, which never run beside a timing.
    pub build_jobs: usize,
    // Whether to time each choice point and whole plan against itself too, which says how far timing alone moves a comparison.
    pub self_comparisons: bool,
}

impl Default for GraphTuneOptions {
    fn default() -> Self {
        Self {
            policy: LoweringPolicy::default(),
            choice_budget: Duration::from_secs(10),
            runs: 25,
            schedules: true,
            measurement: Measurement::GraphReplay,
            build_directory: env::temp_dir().join("singe-tuning"),
            memory_limit: None,
            plan_budget: Duration::ZERO,
            plan_seed: 0,
            plan_speed: true,
            build_jobs: thread::available_parallelism()
                .map_or(1, |threads| (threads.get() / 2).max(1)),
            self_comparisons: false,
        }
    }
}

// A record of `choice` no timing has decided yet.
fn held_record(choice: &TunedChoice) -> TunedRecord {
    TunedRecord {
        choice: choice.clone(),
        default_ns: None,
        best_ns: 0,
        bytes: 0,
    }
}

// The records that force `candidate`.
pub fn forced_records(candidate: &Candidate, environment: &TuningEnvironment) -> TuningRecords {
    let mut forced = TuningRecords::new(Some(environment.clone()));
    for (key, choice) in &candidate.choices {
        forced.insert(
            key.clone(),
            TunedRecord {
                choice: choice.clone(),
                default_ns: None,
                best_ns: 0,
                bytes: 0,
            },
        );
    }
    forced
}

// Which libraries' own configurations a choice point lists, and the key their records go under.
#[derive(Default)]
struct LibraryChoices {
    providers: Vec<CudaGraphProvider>,
    key: String,
}

// What tuning has found so far.
struct Found<'a, 'o> {
    // What this search assumes, which every candidate of it runs with.
    assumed: Vec<(String, TunedChoice)>,
    measurer: &'a mut Measurer<'o>,
    ir: &'a IrGraph,
    evaluated: &'a Evaluated,
    in_place: &'a InPlaceWrites,
    environment: &'a TuningEnvironment,
    device: Device,
    options: &'a GraphTuneOptions,
    keys: HashSet<String>,
    points: Vec<ChoicePoint>,
    report: TuningReport,
    self_comparisons: Vec<SelfComparison>,
}

impl Found<'_, '_> {
    // Times one choice point's candidates on the small graph of `members`, and keeps what they came to.
    fn decide(
        &mut self,
        key: String,
        members: &[ValueId],
        outputs: &[ValueId],
        candidates: Vec<Candidate>,
        libraries: &LibraryChoices,
    ) {
        let small = subgraph(
            self.ir,
            self.evaluated,
            members,
            outputs,
            self.in_place,
            self.device,
        );
        let small = match small {
            Ok(small) => small,
            Err(error) => {
                self.report.choices.push(ChoiceReport {
                    key,
                    candidates: Vec::new(),
                    failure: Some(CandidateFailure::from_error(FailureStage::Graph, &error)),
                    comparisons: Vec::new(),
                    between_winners: Vec::new(),
                    second_reference: None,
                });
                return;
            }
        };
        // What libraries could run this with comes last, with a budget of its own, since a program's candidates take seconds to build.
        let mut candidates = candidates;
        let ways = candidates.len();
        let mut lists = Vec::with_capacity(libraries.providers.len());
        for &provider in &libraries.providers {
            let way = Candidate {
                choices: vec![(key.clone(), TunedChoice::Way(TunedWay::Library(provider)))],
            };
            let listed = listed_library_choices(
                &small,
                &{
                    let mut records = forced_records(&way, self.environment);
                    for (key, choice) in &self.assumed {
                        records.insert(key.clone(), held_record(choice));
                    }
                    records
                },
                self.device,
                self.options,
                &key,
                provider,
            );
            lists.push(
                listed
                    .into_iter()
                    .map(|choice| Candidate {
                        choices: vec![way.choices[0].clone(), (libraries.key.clone(), choice)],
                    })
                    .collect::<VecDeque<_>>(),
            );
        }

        // In turns, so one library's long list does not take the other's budget.
        while lists.iter().any(|list| !list.is_empty()) {
            candidates.extend(lists.iter_mut().filter_map(VecDeque::pop_front));
        }

        let assumed = Candidate {
            choices: self.assumed.clone(),
        };
        let environment = self.environment;
        let records_of = |candidate: &Candidate| {
            let mut records = forced_records(&assumed, environment);
            for (key, choice) in &candidate.choices {
                records.insert(key.clone(), held_record(choice));
            }
            records
        };
        let all_records = candidates.iter().map(&records_of).collect::<Vec<_>>();
        self.measurer.build_ahead(&small, &all_records);
        let measurer = &mut *self.measurer;
        let memory_limit = self.options.memory_limit;
        let (mut point, mut report) = decide_in_two(
            key.clone(),
            &candidates,
            ways,
            self.options.choice_budget,
            |candidate| measurer.measure(&small, &records_of(candidate), memory_limit, None),
        );
        confirm(&mut point, &mut report, |first, candidate| {
            measurer.compare(&small, &records_of(first), &records_of(candidate))
        });
        report.second_reference = small.second.as_ref().map(|second| second.reference);
        if self.options.self_comparisons
            && let Ok(compared) = measurer.self_comparison(&key, &small, &all_records[0])
        {
            self.self_comparisons.push(compared);
        }

        self.points.push(point);
        self.report.choices.push(report);
    }
}

// Finds and times what a graph can choose that `found` has not already decided: fusion groups first, since whether a group runs at all decides which nodes have choices of their own, and then each node with more than one way to run.
fn find_choices(
    found: &mut Found<'_, '_>,
    planner: &KernelPlanner,
    lowering: &LoweringPlan,
    order: &[ValueId],
    fusion: &FusionPlan,
) -> Result<()> {
    let ir = found.ir;
    if found
        .options
        .policy
        .provider_enabled(CudaGraphProvider::Fusion)
    {
        for group in fusion.groups() {
            let key = group_key(ir, &group.members)?;
            if !found.keys.insert(key.clone()) {
                continue;
            }

            let mut candidates = vec![Candidate {
                choices: Vec::new(),
            }];
            // The first of a module's parameters is its default, which the empty candidate runs.
            if let GroupKind::Module(kind) = &group.kind {
                let (parameter_key, parameters) = module_parameters(planner, ir, kind)?;
                candidates.extend(parameters.into_iter().skip(1).map(|parameters| Candidate {
                    choices: vec![(parameter_key.clone(), TunedChoice::Parameters(parameters))],
                }));
            }

            candidates.push(Candidate {
                choices: vec![(key.clone(), TunedChoice::Way(TunedWay::Separate))],
            });
            found.decide(
                key,
                &group.members,
                &group.outputs,
                candidates,
                &LibraryChoices::default(),
            );
        }
    }

    // Each node with more than one way to run.
    let mut seen = HashSet::new();
    for &id in order {
        let Ok(operation) = ir.required_operation_for_value(id) else {
            continue;
        };
        let Some(opcode) = operation.opcode() else {
            continue;
        };
        if !seen.insert(operation.id()) {
            continue;
        }

        let key = way_key(ir, id)?;
        if found.keys.contains(&key) {
            continue;
        }

        let libraries = lowering
            .candidates(id)
            .iter()
            .map(|candidate| candidate.provider())
            .filter(|provider| {
                !matches!(
                    provider,
                    CudaGraphProvider::Alias
                        | CudaGraphProvider::Kernel
                        | CudaGraphProvider::Fusion
                )
            })
            .collect::<Vec<_>>();
        let metadata = operation
            .inputs()
            .iter()
            .map(|&input| {
                Ok(KernelInput::new(
                    ir.required_value_data_type(input)?,
                    ir.required_value_layout(input)?.clone(),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let (parameter_key, parameters) = planner
            .parameter_candidates(opcode, operation.attributes(), &metadata)
            .unwrap_or_else(|| (String::new(), Vec::new()));
        let generated = lowering.node_supports(id, CudaGraphProvider::Kernel);
        let mut candidates = libraries
            .iter()
            .map(|&provider| Candidate {
                choices: vec![(key.clone(), TunedChoice::Way(TunedWay::Library(provider)))],
            })
            .collect::<Vec<_>>();
        if generated {
            if parameters.is_empty() {
                candidates.push(Candidate {
                    choices: vec![(key.clone(), TunedChoice::Way(TunedWay::Generated))],
                });
            }

            candidates.extend(
                parameters
                    .iter()
                    .map(|&parameters: &KernelParameters| Candidate {
                        choices: vec![
                            (key.clone(), TunedChoice::Way(TunedWay::Generated)),
                            (parameter_key.clone(), TunedChoice::Parameters(parameters)),
                        ],
                    }),
            );
        }

        if candidates.len() < 2 {
            continue;
        }

        let members = operation.result_values().collect::<Vec<_>>();
        let members = if matches!(
            opcode,
            Operation::PagedAttentionOutput
                | Operation::PagedAttentionLse
                | Operation::PagedMlaAttentionOutput
                | Operation::PagedMlaAttentionLse
        ) {
            attention_results(ir, order, id)?
        } else {
            members
        };
        // A library's own configurations are choices too, for those the policy enables.
        let tunable = LibraryChoices {
            providers: libraries
                .iter()
                .copied()
                .filter(|provider| {
                    matches!(
                        provider,
                        CudaGraphProvider::CublasLt | CudaGraphProvider::CudnnBackend
                    )
                })
                .collect(),
            key: library_key(ir, id)?,
        };
        found.keys.insert(key.clone());
        found.decide(key, &members, &members, candidates, &tunable);
    }
    Ok(())
}

// Tunes what a whole-plan choice leaves new on `ir`, under `assumed`, and returns what won.
fn tune_for_choice(
    found: Found<'_, '_>,
    planner: &KernelPlanner,
    lowering: &LoweringPlan,
    order: &[ValueId],
    fusion: &FusionPlan,
) -> Result<(Vec<(String, TunedChoice)>, TuningReport)> {
    let mut found = found;
    find_choices(&mut found, planner, lowering, order, fusion)?;
    let mut report = found.report;
    report.self_comparisons = found.self_comparisons;
    let chosen = found
        .points
        .iter()
        .filter_map(|point| point.chosen.and_then(|index| point.alternatives.get(index)))
        .flat_map(|alternative| alternative.candidate.choices.iter().cloned())
        .collect();
    Ok((chosen, report))
}

// Measures what `graph` can choose on `device`, on `inputs`, and returns records that compiling and exporting read.
// Each choice point is timed alone, as its records will be read, against the values the graph evaluates to, and identical choice points are timed once.
// The whole plan is then checked and searched, and what it returns loses whatever failed, disagreed, or did not fit, which `TuningRecords::report` explains.
//
// # Errors
//
// Returns an error if the graph cannot be analyzed or evaluated, or a candidate list cannot be made.
pub fn tune_graph<S: BuildHasher>(
    graph: &Graph,
    inputs: &HashMap<String, Tensor, S>,
    device: Device,
    options: &GraphTuneOptions,
) -> Result<TuningRecords> {
    let tuning_started = Instant::now();
    if options.measurement == Measurement::Program {
        remove_abandoned_candidates(&options.build_directory);
    }

    let environment = TuningEnvironment::current(device, options.measurement)?;
    let placed = graph.placed_on(device)?;
    let compiled = placed.compile_with_options(
        GraphCompileOptions::new().with_cuda_providers(options.policy.graph_providers()),
    )?;
    let ir = channels_last_convolution_chains(compiled.shared_ir())?;
    let order = compiled.value_order().to_vec();
    let target = runtime_for_device(device)?.architecture();
    let context = PlanningContext {
        generated_convolutions: true,
        target: Some(target),
        ..PlanningContext::default()
    };
    let planner = KernelPlanner::new(context.clone());
    let feasibility = KernelFeasibility::new(&context);
    let lowering = LoweringPlan::create(&ir, &order, &options.policy, &feasibility)?;
    let inputs = inputs
        .iter()
        .map(|(name, tensor)| Ok((name.clone(), tensor.to_device(device)?)))
        .collect::<Result<HashMap<String, Tensor>>>()?;
    let inputs = &inputs;
    let evaluated = evaluate(&ir, &order, inputs, device)?;

    let users = graph_users(&ir, &order)?;
    let outputs = ir.output_value_set();
    let in_place = InPlaceWrites::analyze(&ir, &order, &users, &outputs, WritableInputs::Every)?;
    let mut measurer = Measurer::for_choice_points(device, options);
    let mut found = Found {
        assumed: Vec::new(),
        measurer: &mut measurer,
        ir: &ir,
        evaluated: &evaluated,
        in_place: &in_place,
        environment: &environment,
        device,
        options,
        keys: HashSet::new(),
        points: Vec::new(),
        report: TuningReport::default(),
        self_comparisons: Vec::new(),
    };
    let fusion = plan_fusion(&ir, &order, &users, &outputs, &lowering, None, None)?;
    find_choices(&mut found, &planner, &lowering, &order, &fusion)?;

    let Found {
        mut points,
        mut report,
        keys,
        self_comparisons,
        ..
    } = found;
    report.self_comparisons = self_comparisons;

    // A whole-plan choice is judged tuned for itself, so what it leaves new is timed first, against values that graph evaluates to.
    let mut whole_choices = Vec::new();
    let merged_bytes = merged_projection_bytes(&placed)?;
    if merged_bytes > 0 {
        let compiled = placed.compile_with_options(
            GraphCompileOptions::new()
                .with_cuda_providers(options.policy.graph_providers())
                .with_merged_projections(true),
        )?;
        let merged_ir = channels_last_convolution_chains(compiled.shared_ir())?;
        let merged_order = compiled.value_order().to_vec();
        let merged_lowering =
            LoweringPlan::create(&merged_ir, &merged_order, &options.policy, &feasibility)?;
        let merged_evaluated = evaluate(&merged_ir, &merged_order, inputs, device)?;
        let merged_users = graph_users(&merged_ir, &merged_order)?;
        let merged_outputs = merged_ir.output_value_set();
        let merged_in_place = InPlaceWrites::analyze(
            &merged_ir,
            &merged_order,
            &merged_users,
            &merged_outputs,
            WritableInputs::Every,
        )?;
        let merged_fusion = plan_fusion(
            &merged_ir,
            &merged_order,
            &merged_users,
            &merged_outputs,
            &merged_lowering,
            None,
            None,
        )?;
        let (tuned, tuned_report) = tune_for_choice(
            Found {
                assumed: Vec::new(),
                measurer: &mut measurer,
                ir: &merged_ir,
                evaluated: &merged_evaluated,
                in_place: &merged_in_place,
                environment: &environment,
                device,
                options,
                keys: keys.clone(),
                points: Vec::new(),
                report: TuningReport::default(),
                self_comparisons: Vec::new(),
            },
            &planner,
            &merged_lowering,
            &merged_order,
            &merged_fusion,
        )?;
        report.choices.extend(tuned_report.choices);
        report
            .self_comparisons
            .extend(tuned_report.self_comparisons);
        whole_choices.push((
            merged_projections_key(),
            TunedChoice::MergedProjections,
            tuned,
        ));
    }

    // Attention fused with its query's rotation, wherever a graph has one to fuse.
    let mut fused = TuningRecords::new(None);
    fused.insert(
        fused_attention_key(),
        held_record(&TunedChoice::FusedAttention),
    );
    let fused_fusion = plan_fusion(&ir, &order, &users, &outputs, &lowering, None, Some(&fused))?;
    if fused_fusion.groups().iter().any(|group| {
        matches!(
            group.kind,
            GroupKind::Module(ModuleKind::RotaryAttention(_))
        )
    }) {
        let (tuned, tuned_report) = tune_for_choice(
            Found {
                assumed: vec![(fused_attention_key(), TunedChoice::FusedAttention)],
                measurer: &mut measurer,
                ir: &ir,
                evaluated: &evaluated,
                in_place: &in_place,
                environment: &environment,
                device,
                options,
                keys,
                points: Vec::new(),
                report: TuningReport::default(),
                self_comparisons: Vec::new(),
            },
            &planner,
            &lowering,
            &order,
            &fused_fusion,
        )?;
        report.choices.extend(tuned_report.choices);
        report
            .self_comparisons
            .extend(tuned_report.self_comparisons);
        whole_choices.push((fused_attention_key(), TunedChoice::FusedAttention, tuned));
    }

    // Each choice held alone, so now the plan they make must hold whole.
    let whole = whole_graph(&placed, &ir, &order, &evaluated, inputs)?;
    // The expected values are on the host now, so only the plan's own tensors stay on the device.
    drop(evaluated);
    let mut runner = WholeGraphRunner::new(&whole, device, options).with_merged_bytes(merged_bytes);
    let limits = PlanLimits {
        memory: options.memory_limit,
        speed: options.plan_speed,
    };
    // What only timing a whole plan can show, and how a program launches its kernels.
    if options.measurement == Measurement::Program
        && runtime_for_device(device)?
            .architecture()
            .compute_capability()
            >= (9, 0)
    {
        whole_choices.push((
            dependent_launch_key(),
            TunedChoice::DependentLaunch,
            Vec::new(),
        ));
    }

    for (key, choice, tuned) in whole_choices {
        let chosen = points.iter().map(|point| point.chosen).collect::<Plan>();
        let base = plan_records(&points, &chosen, Some(&environment));
        let candidates = [
            Candidate {
                choices: Vec::new(),
            },
            Candidate {
                choices: iter::once((key.clone(), choice)).chain(tuned).collect(),
            },
        ];
        // Each is one decision of two whole plans, and a budget meant for many small graphs would skip the second on a large model.
        let records_of = |candidate: &Candidate| {
            let mut records = base.clone();
            for (key, choice) in &candidate.choices {
                records.insert(key.clone(), held_record(choice));
            }
            records
        };
        let (mut point, mut choice) = decide(key, &candidates, Duration::MAX, |candidate| {
            runner
                .run(&records_of(candidate), None)
                .map_err(MeasureError::Failed)
        });
        confirm(&mut point, &mut choice, |first, candidate| {
            runner
                .compare(&records_of(first), &records_of(candidate))
                .map_err(MeasureError::Failed)
        });
        points.push(point);
        report.choices.push(choice);
    }

    let chosen = points.iter().map(|point| point.chosen).collect::<Plan>();
    if options.self_comparisons {
        let records = plan_records(&points, &chosen, Some(&environment));
        match runner.self_comparison("whole plan", &records) {
            Ok(compared) => report.self_comparisons.push(compared),
            Err(failure) => report.choices.push(ChoiceReport {
                key: "whole plan timed against itself".to_owned(),
                candidates: Vec::new(),
                failure: Some(failure),
                comparisons: Vec::new(),
                between_winners: Vec::new(),
                second_reference: None,
            }),
        }
    }

    let (plan, check) = check_plan(&points, chosen, Some(&environment), limits, &mut runner);
    report.checks.push(check);
    let (searched, search) = search_plans(
        &points,
        plan.clone(),
        Some(&environment),
        PlanSearchOptions {
            budget: options.plan_budget,
            seed: options.plan_seed,
            mutations: PLAN_MUTATIONS,
            memory_limit: options.memory_limit,
        },
        &mut runner,
    );
    let mut changed = searched != plan;
    let mut plan = searched;
    report.searches.push(search);

    // With every other choice made, the schedules of the generated kernels those choices leave.
    if options.schedules {
        let before_schedules = plan.clone();
        let recorder = KernelRecorder::new();
        let mut prepared = placed.prepare(
            GraphPrepareOptions::new(device).with_cuda_options(
                GraphCudaOptions::new()
                    .with_lowering_policy(options.policy.clone())
                    .with_inputs_in_place(
                        whole
                            .inputs
                            .iter()
                            .map(|(name, tensor)| (name.clone(), tensor.clone())),
                    )
                    .with_tuning_records(
                        plan_records(&points, &plan, Some(&environment))
                            .measured_as(Measurement::GraphReplay),
                    )
                    .with_kernel_recorder(recorder.clone()),
            ),
        )?;
        prepared.run(&whole.inputs)?;
        for planned in recorder.take()? {
            let key = schedule_key(&planned.kernel, &planned.bindings);
            let search = kernel_tuning::tune(
                device,
                &planned.kernel,
                planned.schedule,
                &planned.bindings,
                kernel_tuning::TuneOptions {
                    budget: options.choice_budget,
                    rounds: options.runs,
                    ..kernel_tuning::TuneOptions::default()
                },
            );
            let result = match search {
                Ok(result) => result,
                Err(error) => {
                    report.choices.push(ChoiceReport {
                        key,
                        candidates: Vec::new(),
                        failure: Some(CandidateFailure::from_error(FailureStage::Run, &error)),
                        comparisons: Vec::new(),
                        between_winners: Vec::new(),
                        second_reference: None,
                    });
                    continue;
                }
            };
            report.choices.push(ChoiceReport {
                key: key.clone(),
                candidates: result
                    .candidates
                    .iter()
                    .map(|(schedule, outcome)| CandidateReport {
                        choice: format!("{schedule:?}"),
                        outcome: schedule_outcome(outcome),
                    })
                    .collect(),
                failure: None,
                comparisons: Vec::new(),
                between_winners: Vec::new(),
                second_reference: None,
            });
            if result.best != result.default {
                let nanoseconds = |schedule| {
                    result.timing(schedule).map_or(0, |timing| {
                        u64::try_from(timing.median.as_nanos()).unwrap_or(u64::MAX)
                    })
                };
                points.push(ChoicePoint {
                    key: key.clone(),
                    default_ns: Some(nanoseconds(result.default)),
                    default_bytes: None,
                    alternatives: vec![Alternative {
                        candidate: Candidate {
                            choices: vec![(key, TunedChoice::Schedule(result.best))],
                        },
                        median_ns: nanoseconds(result.best),
                        bytes: 0,
                    }],
                    chosen: Some(0),
                });
                plan.push(Some(0));
            }
        }

        // Each schedule won alone, on its own kernel, so what they do together is still to be shown.
        if plan.len() > before_schedules.len() {
            let mut without = before_schedules;
            without.resize(plan.len(), None);
            let compared = runner.compare(
                &plan_records(&points, &without, Some(&environment)),
                &plan_records(&points, &plan, Some(&environment)),
            );
            let wins = compared.as_ref().is_ok_and(Comparison::wins);
            let (comparison, failure) = match compared {
                Ok(comparison) => (Some(comparison), None),
                Err(failure) => (None, Some(failure)),
            };
            report.choices.push(ChoiceReport {
                key: "schedules".to_owned(),
                candidates: Vec::new(),
                failure: None,
                comparisons: vec![ComparisonReport {
                    choice: "every schedule that won alone".to_owned(),
                    comparison,
                    failure,
                }],
                between_winners: Vec::new(),
                second_reference: None,
            });
            if wins {
                changed = true;
            } else {
                plan = without;
            }
        }
    }

    // The check runs last, so what the search and the schedules changed answers to it too.
    if changed {
        let (checked, check) = check_plan(&points, plan, Some(&environment), limits, &mut runner);
        report.checks.push(check);
        plan = checked;
    }

    // The runs wrote the caller's tensors, which now get back what they held.
    restore_written(&whole).map_err(|failure| Error::invalid_argument(failure.message))?;
    let mut records = plan_records(&points, &plan, Some(&environment));
    report.spent.add(measurer.spent());
    report.spent.add(runner.spent());
    report.spent.total_ns = u64::try_from(tuning_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    records.set_report(report);
    Ok(records)
}

fn schedule_outcome(outcome: &Outcome) -> CandidateOutcome {
    match outcome {
        Outcome::Timed(timing) => CandidateOutcome::Timed {
            median_ns: u64::try_from(timing.median.as_nanos()).unwrap_or(u64::MAX),
            bytes: 0,
        },
        Outcome::Failed(failure) => CandidateOutcome::Failed(failure.clone()),
        Outcome::Mismatched => CandidateOutcome::Disagreed,
        Outcome::Skipped => CandidateOutcome::Skipped,
    }
}

// The values of both results of the attention node producing `id`, in order, so a small graph computes them together.
fn attention_results(ir: &IrGraph, order: &[ValueId], id: ValueId) -> Result<Vec<ValueId>> {
    let operation = ir.required_operation_for_value(id)?;
    let inputs = operation.inputs();
    let mut results = Vec::new();
    for &other in order {
        let candidate = ir.required_operation_for_value(other)?;
        if candidate.inputs() == inputs
            && candidate.attributes() == operation.attributes()
            && matches!(
                candidate.opcode(),
                Some(
                    Operation::PagedAttentionOutput
                        | Operation::PagedAttentionLse
                        | Operation::PagedMlaAttentionOutput
                        | Operation::PagedMlaAttentionLse
                )
            )
        {
            results.push(other);
        }
    }
    Ok(results)
}
