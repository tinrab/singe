use std::{
    collections::{HashMap, HashSet},
    error::Error as StdError,
    fs,
    path::{Path, PathBuf},
    process,
    result::Result as StdResult,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

use crate::{
    backend::{
        cuda::{
            CudaEvent, CudaStream, EventTiming, clear_cache,
            export::{
                BufferKind, CudaExport, CudaProgram, ProgramFiles,
                compiled::{CompiledProgram, ProgramInstance},
            },
            graph_binding::{InPlaceWrites, ScatterTarget},
            tuning::{
                comparison::Comparison,
                library::LibraryRecorder,
                plan::{MeasureError, Measured, PlanRunner},
                records::{Measurement, TunedChoice, TuningEnvironment, TuningRecords},
                report::{
                    CandidateFailure, FailureStage, SecondReference, SelfComparison, TimeSpent,
                },
                tune::GraphTuneOptions,
            },
        },
        registry::{CudaGraphProvider, Operation, OperationAttributes},
    },
    data_type::DataType,
    device::Device,
    dispatch::call,
    error::{Error, Result},
    graph::{
        Graph, GraphCompileOptions, GraphCudaOptions, GraphExecutor, GraphPrepareOptions,
        ir::{Graph as IrGraph, ValueId},
    },
    tensor::Tensor,
};

// What can stop a candidate, as opposed to what stops the tuning step.
type Attempt<T, E = CandidateFailure> = StdResult<T, E>;

// Rounding builds up through a whole graph, so what a whole plan computes may sit this much further from the evaluated graph than one choice point's result.
const WHOLE_GRAPH_TOLERANCE: f64 = 4.0;

// How many elements of a tensor one comparison holds as `f64` at a time, so comparing a model's caches costs the host little.
const COMPARED_ELEMENTS: usize = 1 << 18;

// An evaluated result of at least this many bytes waits on the host, as a model's written caches would otherwise fill the device.
const EVALUATED_ON_HOST_BYTES: usize = 1 << 24;

// A small graph whose inputs hold at most this many bytes has its second inputs' values evaluated on the host, which shares no library with a candidate.
const HOST_REFERENCE_BYTES: usize = 1 << 26;

// How many built programs a runner keeps, each with a host copy of its weights: enough for the best plan and the one it is being compared with.
const BUILT_PROGRAMS: usize = 2;

// How many runs a candidate takes on its turn in a round, of which only the last is timed, so that it is not timed on what the other left in the caches.
const TURN_RUNS: usize = 2;

// How many rounds of a comparison each of `GraphTuneOptions::runs` buys, over all its pairs of instances, since a choice rests on these more than on any other timing.
const ROUNDS_PER_RUN: usize = 4;

// Whether the baseline is made first, for each fresh pair of instances a comparison runs over.
// This order has no period of two, since an allocator reusing memory in a fixed order could otherwise give each side the same place in every pair, which would then look like a difference between them.
const BASELINE_MADE_FIRST: [bool; 5] = [true, false, false, true, false];

// How many pairs of blocks two candidates that do not fit together alternate over.
const BLOCK_PAIRS: usize = 6;

// The values of a graph, computed once, which every candidate must match.
pub struct Evaluated {
    values: HashMap<ValueId, Tensor>,
}

// The tensor a graph holds for `id` itself, a constant or bound state such as a model's weights.
fn held_tensor(ir: &IrGraph, id: ValueId) -> Option<&Tensor> {
    ir.constant_tensor(id).or_else(|| ir.initializer_tensor(id))
}

pub fn evaluate(
    ir: &IrGraph,
    order: &[ValueId],
    inputs: &HashMap<String, Tensor>,
    device: Device,
) -> Result<Evaluated> {
    let mut values = HashMap::new();
    for (id, name) in ir.input_values() {
        if let Some(tensor) = inputs.get(name) {
            values.insert(id, tensor.to_device(device)?);
        }
    }

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

        let operands = operation
            .inputs()
            .iter()
            .map(|input| {
                values
                    .get(input)
                    .or_else(|| held_tensor(ir, *input))
                    .ok_or_else(|| Error::missing_input("tuning evaluation"))?
                    .to_device(device)
            })
            .collect::<Result<Vec<_>>>()?;
        let operand_references = operands.iter().collect::<Vec<_>>();
        let outputs = call(opcode, operation.attributes().clone(), &operand_references)?;
        for (result, output) in operation.result_values().zip(outputs) {
            let bytes = output.element_count() * output.data_type().size_bytes();
            let output = if bytes >= EVALUATED_ON_HOST_BYTES {
                output.to_device(Device::Cpu)?
            } else {
                output
            };
            values.insert(result, output);
        }
    }
    Ok(Evaluated { values })
}

impl Evaluated {
    // What `id` evaluated to, as a candidate's result is compared.
    fn expected(&self, ir: &IrGraph, id: ValueId) -> Result<Expected> {
        Expected::create(
            self.values
                .get(&id)
                .or_else(|| held_tensor(ir, id))
                .ok_or_else(|| Error::missing_output("tuning evaluation"))?,
        )
    }
}

// A tensor on the host, in its own type, as one row.
fn host_row(tensor: &Tensor) -> Result<Tensor> {
    let count = i64::try_from(tensor.element_count())
        .map_err(|_| Error::out_of_range("tuning comparison elements"))?;
    tensor
        .to_device(Device::Cpu)?
        .contiguous()?
        .reshape([count])
}

// Calls `visit` with successive chunks of a row of `tensor` and of `other`, as `f64`, until it returns false.
fn all_chunks(
    tensor: &Tensor,
    other: Option<&Tensor>,
    mut visit: impl FnMut(&[f64], &[f64]) -> bool,
) -> Result<bool> {
    let count = tensor.element_count();
    let numbers = |tensor: &Tensor, start: usize, length: usize| -> Result<Vec<f64>> {
        tensor
            .narrow(
                0,
                i64::try_from(start).map_err(|_| Error::out_of_range("tuning comparison start"))?,
                i64::try_from(length)
                    .map_err(|_| Error::out_of_range("tuning comparison length"))?,
            )?
            .cast(DataType::F64)?
            .to_vec::<f64>()
    };
    let mut start = 0;
    while start < count {
        let length = COMPARED_ELEMENTS.min(count - start);
        let chunk = numbers(tensor, start, length)?;
        let other_chunk = other
            .map(|other| numbers(other, start, length))
            .transpose()?
            .unwrap_or_default();
        if !visit(&chunk, &other_chunk) {
            return Ok(false);
        }

        start += length;
    }
    Ok(true)
}

// What a candidate's result should hold, on the host, in its own type, so a model's caches cost no device memory and no `f64`s while they wait.
#[derive(Debug, Clone)]
pub struct Expected {
    values: Tensor,
    // The largest finite magnitude it holds, at least one, which its tolerance is relative to.
    magnitude: f64,
}

impl Expected {
    pub fn create(tensor: &Tensor) -> Result<Self> {
        let values = host_row(tensor)?;
        let mut magnitude = 1.0_f64;
        all_chunks(&values, None, |chunk, _| {
            for entry in chunk.iter().filter(|entry| entry.is_finite()) {
                magnitude = magnitude.max(entry.abs());
            }
            true
        })?;
        Ok(Self { values, magnitude })
    }

    // These values with `amount` added, for a test to make a candidate wrong.
    #[cfg(test)]
    pub fn shifted(&self, amount: f64) -> Result<Self> {
        Self::create(&self.values.add_scalar(amount)?)
    }
}

// A graph candidates run, what it reads, and what it should compute.
pub struct SmallGraph {
    graph: Graph,
    pub inputs: HashMap<String, Tensor>,
    // The inputs it writes where they are, as the whole graph does, each with what it holds before a run.
    pub written: Vec<(String, Tensor)>,
    // What its outputs and then its written inputs should hold, from the evaluated graph.
    pub expected: Vec<Expected>,
    // The type whose rounding decides how close its results must be.
    pub rounding: DataType,
    pub tolerance_scale: f64,
    // Other values for its floating inputs, which every candidate must also agree on, since one set of inputs can hide a kernel's error.
    pub second: Option<SecondInputs>,
}

// A second set of a small graph's inputs, of the same shapes, and what its outputs and written inputs should then hold.
pub struct SecondInputs {
    // The inputs that differ from the first set, with their second values, on the device.
    pub values: Vec<(String, Tensor)>,
    pub expected: Vec<Expected>,
    pub reference: SecondReference,
}

// The coarsest floating point type among `values`, which rounds the most, or an integer type if they have no other.
fn coarsest(ir: &IrGraph, values: impl IntoIterator<Item = ValueId>) -> Result<DataType> {
    let rank = |data_type| match data_type {
        DataType::Bf16 => 3,
        DataType::F16 => 2,
        DataType::F32 => 1,
        _ => 0,
    };
    let mut coarsest = DataType::F64;
    for value in values {
        let data_type = ir.required_value_data_type(value)?;
        if rank(data_type) > rank(coarsest) {
            coarsest = data_type;
        }
    }
    Ok(coarsest)
}

// A copy of `tensor` with storage of its own.
fn copied(tensor: &Tensor) -> Result<Tensor> {
    let copy = Tensor::zeroed_on(tensor.device(), tensor.data_type(), tensor.shape().clone())?;
    copy.assign(tensor)?;
    Ok(copy)
}

// A small graph of `members`, returning `outputs`, reading what they read from outside as inputs, and the values of those inputs.
// A plain cache write the whole graph makes in place stays in place here on a copy of its cache, so what a candidate is timed on holds no copy of that cache, and what it wrote is compared by reading the copy.
pub fn subgraph(
    ir: &IrGraph,
    evaluated: &Evaluated,
    members: &[ValueId],
    outputs: &[ValueId],
    in_place: &InPlaceWrites,
    device: Device,
) -> Result<SmallGraph> {
    // Views stay in the small graph, so its nodes read the layouts the whole graph's do.
    let mut nodes = Vec::new();
    let mut pending = members.to_vec();
    let mut included = HashSet::new();
    while let Some(id) = pending.pop() {
        if !included.insert(id) {
            continue;
        }

        nodes.push(id);
        let operation = ir.required_operation_for_value(id)?;
        for &input in operation.inputs() {
            if ir.required_value_alias(input).ok().flatten().is_some() {
                pending.push(input);
            }
        }
    }

    nodes.sort_unstable_by_key(|id| id.index());
    let mut writes = HashMap::new();
    for &id in outputs {
        let operation = ir.required_operation_for_value(id)?;
        let plain = operation.opcode() == Some(Operation::IndexPut)
            && matches!(operation.attributes(), OperationAttributes::IndexPut(attributes) if !attributes.accumulate);
        if plain && let Some(ScatterTarget::Input(base)) = in_place.target(id) {
            writes.insert(id, base);
        }
    }

    // A graph returns something, so a choice point of nothing but writes keeps them as results.
    if outputs.iter().all(|output| writes.contains_key(output)) {
        writes.clear();
    }

    let mut operands_by_name = Vec::new();
    let mut names = HashMap::new();
    let mut written = Vec::new();
    let mut written_values = Vec::new();
    for &id in &nodes {
        for &input in ir.required_operation_for_value(id)?.inputs() {
            if !included.contains(&input) && !names.contains_key(&input) {
                let tensor = evaluated
                    .values
                    .get(&input)
                    .cloned()
                    .or_else(|| held_tensor(ir, input).cloned())
                    .ok_or_else(|| Error::missing_input("tuning subgraph"))?;
                let name = format!("input{}", input.index());
                let tensor = tensor.to_device(device)?;
                let write = writes
                    .iter()
                    .find_map(|(&write, &base)| (base == input).then_some(write));
                let tensor = if let Some(write) = write {
                    // Nothing writes the evaluated tensor, so it holds what a run starts from.
                    written.push((name.clone(), tensor.clone()));
                    written_values.push(write);
                    copied(&tensor)?
                } else {
                    tensor
                };
                names.insert(input, operands_by_name.len());
                operands_by_name.push((name, tensor));
            }
        }
    }

    let kept = outputs
        .iter()
        .copied()
        .filter(|output| !writes.contains_key(output))
        .collect::<Vec<_>>();
    let (graph, _) = Graph::capture(operands_by_name.clone(), |tensors| {
        let mut produced: HashMap<ValueId, Tensor> = HashMap::new();
        let mut seen = HashSet::new();
        for &id in &nodes {
            let operation = ir.required_operation_for_value(id)?;
            if !seen.insert(operation.id()) {
                continue;
            }

            let opcode = operation
                .opcode()
                .ok_or_else(|| Error::missing_node("tuning subgraph"))?;
            let operands = operation
                .inputs()
                .iter()
                .map(|input| {
                    produced
                        .get(input)
                        .cloned()
                        .or_else(|| names.get(input).map(|&index| tensors[index].clone()))
                        .ok_or_else(|| Error::missing_input("tuning subgraph"))
                })
                .collect::<Result<Vec<_>>>()?;
            let references = operands.iter().collect::<Vec<_>>();
            let results = call(opcode, operation.attributes().clone(), &references)?;
            if let Some(base) = writes.get(&id) {
                tensors[names[base]].assign(&results[0])?;
            }
            for (result, tensor) in operation.result_values().zip(results) {
                produced.insert(result, tensor);
            }
        }
        kept.iter()
            .map(|id| {
                produced
                    .get(id)
                    .cloned()
                    .ok_or_else(|| Error::missing_output("tuning subgraph"))
            })
            .collect()
    })?;
    let expected = kept
        .iter()
        .chain(&written_values)
        .map(|&id| evaluated.expected(ir, id))
        .collect::<Result<Vec<_>>>()?;
    let mut touched = nodes.clone();
    for &id in &nodes {
        touched.extend_from_slice(ir.required_operation_for_value(id)?.inputs());
    }

    let mut small = SmallGraph {
        graph,
        inputs: operands_by_name.into_iter().collect(),
        written,
        expected,
        rounding: coarsest(ir, touched)?,
        tolerance_scale: 1.0,
        second: None,
    };
    small.second = second_inputs(&small, device)?;
    Ok(small)
}

// Other values for each floating input `small` does not write, of other signs and magnitudes, with zeros moved off zero, and what the graph computes from them, evaluated on the host when its inputs fit.
fn second_inputs(small: &SmallGraph, device: Device) -> Result<Option<SecondInputs>> {
    let written = small
        .written
        .iter()
        .map(|(name, _)| name.as_str())
        .collect::<HashSet<_>>();
    let mut names = small
        .inputs
        .iter()
        .filter(|(name, tensor)| {
            !written.contains(name.as_str())
                && tensor.data_type().is_float()
                && tensor.element_count() > 0
        })
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    if names.is_empty() {
        return Ok(None);
    }

    names.sort_unstable();
    let mut values = Vec::with_capacity(names.len());
    for name in names {
        let first = &small.inputs[&name];
        let second = first
            .cast(DataType::F32)?
            .mul_scalar(-1.5)?
            .add_scalar(0.125)?
            .cast(first.data_type())?;
        values.push((name, second));
    }

    // What a run starts from: the second values, and what each written input holds before a run.
    let mut inputs = small.inputs.clone();
    for (name, second) in &values {
        inputs.insert(name.clone(), second.clone());
    }

    for (name, before) in &small.written {
        inputs.insert(name.clone(), before.clone());
    }

    let compiled = small
        .graph
        .compile_with_options(GraphCompileOptions::new())?;
    let ir = compiled.shared_ir();
    let order = compiled.value_order().to_vec();
    let bytes = inputs
        .values()
        .map(|tensor| tensor.element_count() * tensor.data_type().size_bytes())
        .sum::<usize>();
    let host = (bytes <= HOST_REFERENCE_BYTES)
        .then(|| evaluate(&ir, &order, &inputs, Device::Cpu).ok())
        .flatten();
    let (evaluated, reference) = match host {
        Some(evaluated) => (evaluated, SecondReference::Host),
        None => (
            evaluate(&ir, &order, &inputs, device)?,
            SecondReference::Device,
        ),
    };
    let expected = ir
        .output_values_vec()
        .iter()
        .map(|&id| evaluated.expected(&ir, id))
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(SecondInputs {
        values,
        expected,
        reference,
    }))
}

// The whole graph as candidates run, writing the caller's tensors, whose starting values wait on the host, since a model's caches may not fit on the device twice.
// `restore_written` puts them back, which the caller of this must do last.
pub fn whole_graph(
    graph: &Graph,
    ir: &IrGraph,
    order: &[ValueId],
    evaluated: &Evaluated,
    inputs: &HashMap<String, Tensor>,
) -> Result<SmallGraph> {
    let outputs = ir.output_values_vec();
    let writes = ir.input_writes();
    let mut written = Vec::with_capacity(writes.len());
    for name in writes {
        let tensor = inputs
            .get(name)
            .ok_or_else(|| Error::missing_input("tuning whole graph"))?;
        written.push((name.clone(), tensor.to_device(Device::Cpu)?.copy()?));
    }

    let expected = outputs
        .iter()
        .map(|&id| evaluated.expected(ir, id))
        .collect::<Result<Vec<_>>>()?;
    Ok(SmallGraph {
        graph: graph.clone(),
        inputs: inputs.clone(),
        written,
        expected,
        rounding: coarsest(ir, order.iter().copied())?,
        tolerance_scale: WHOLE_GRAPH_TOLERANCE,
        // A model's caches do not fit twice, and each choice point's candidates were judged on second inputs already.
        second: None,
    })
}

// Puts back what the small graph's written inputs held, so what a candidate leaves there is its own.
pub fn restore_written(small: &SmallGraph) -> Attempt<()> {
    let failed = |error: &Error| CandidateFailure::from_error(FailureStage::Run, error);
    for (name, before) in &small.written {
        let target = small.inputs.get(name).ok_or_else(|| {
            CandidateFailure::at(FailureStage::Run, format!("no written input {name}"))
        })?;
        let before = before
            .to_device(target.device())
            .map_err(|error| failed(&error))?;
        target.assign(&before).map_err(|error| failed(&error))?;
    }
    Ok(())
}

// Whether a candidate also computes what it should from the second inputs, which `run` runs it on with them in place; the first inputs are back in place afterwards.
fn second_agrees(small: &SmallGraph, run: impl FnOnce() -> Attempt<Vec<Tensor>>) -> Attempt<bool> {
    let Some(second) = &small.second else {
        return Ok(true);
    };
    let failed = |error: &Error| CandidateFailure::from_error(FailureStage::Run, error);
    let mut firsts = Vec::with_capacity(second.values.len());
    for (name, values) in &second.values {
        let target = small.inputs.get(name).ok_or_else(|| {
            CandidateFailure::at(FailureStage::Run, format!("no second input {name}"))
        })?;
        firsts.push((target, copied(target).map_err(|error| failed(&error))?));
        target.assign(values).map_err(|error| failed(&error))?;
    }

    restore_written(small)?;
    let outcome = run().and_then(|outputs| computed_agrees_with(small, &second.expected, &outputs));
    for (target, first) in &firsts {
        target.assign(first).map_err(|error| failed(&error))?;
    }

    restore_written(small)?;
    outcome
}

// Whether what a candidate computed agrees with what it should: its outputs, then what it left in the written inputs.
fn computed_agrees(small: &SmallGraph, outputs: &[Tensor]) -> Attempt<bool> {
    computed_agrees_with(small, &small.expected, outputs)
}

// Whether `outputs` and what is left in the written inputs agree with `expected`.
fn computed_agrees_with(
    small: &SmallGraph,
    expected: &[Expected],
    outputs: &[Tensor],
) -> Attempt<bool> {
    let read_back = |error: &Error| CandidateFailure::from_error(FailureStage::ReadBack, error);
    let written = small
        .written
        .iter()
        .map(|(name, _)| {
            small.inputs.get(name).ok_or_else(|| {
                CandidateFailure::at(FailureStage::ReadBack, format!("no written input {name}"))
            })
        })
        .collect::<Attempt<Vec<_>, _>>()?;
    if outputs.len() + written.len() != expected.len() {
        return Ok(false);
    }

    for (tensor, expected) in outputs.iter().chain(written).zip(expected) {
        if !agrees(small.rounding, small.tolerance_scale, tensor, expected)
            .map_err(|error| read_back(&error))?
        {
            return Ok(false);
        }
    }
    Ok(true)
}

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
    records: &TuningRecords,
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
    records: &TuningRecords,
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

fn nanoseconds(time: Duration) -> u64 {
    u64::try_from(time.as_nanos()).unwrap_or(u64::MAX)
}

// A candidate ready to time, holding what it runs so that several can take turns.
struct Prepared<'a> {
    agrees: bool,
    bytes: u64,
    runs: Runs<'a>,
}

enum Runs<'a> {
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
            Runs::Replay {
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
            Runs::Program { instance, operands } => instance
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

    const fn is_program(records: &TuningRecords) -> bool {
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
        records: &TuningRecords,
    ) -> Attempt<(CudaProgram, ProgramFiles)> {
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
    pub fn build_ahead(&mut self, small: &SmallGraph, candidates: &[TuningRecords]) {
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
        files: &ProgramFiles,
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

    // Readies `small` under `records` the way they will be read, having run it once, and judges what that computed if `judge` asks.
    // A plan over `refuse_over` bytes goes no further.
    fn prepare<'s>(
        &mut self,
        small: &'s SmallGraph,
        records: &TuningRecords,
        refuse_over: Option<u64>,
        judge: bool,
    ) -> Attempt<Prepared<'s>, MeasureError> {
        restore_written(small)?;
        if Self::is_program(records) {
            let (program, files) = self.exported(small, records)?;
            let bytes = program_bytes(&program);
            refuse(bytes, refuse_over)?;
            let compiled = self.compiled(&program, &files)?;
            return self.prepare_program(small, &program, &files, &compiled, bytes, judge);
        }
        self.prepare_replay(small, records, refuse_over, judge)
    }

    fn prepare_replay<'s>(
        &mut self,
        small: &'s SmallGraph,
        records: &TuningRecords,
        refuse_over: Option<u64>,
        judge: bool,
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
        let mut agrees = !judge || computed_agrees(small, &outputs)?;
        drop(outputs);
        if judge && agrees {
            agrees = second_agrees(small, || {
                let outputs = executor
                    .run_on(&small.inputs, &stream)
                    .map_err(|error| failed(&error))?;
                stream.synchronize().map_err(|error| failed(&error))?;
                Ok(outputs)
            })?;
        }

        self.spent.check_ns += nanoseconds(started.elapsed());
        Ok(Prepared {
            agrees,
            bytes,
            runs: Runs::Replay {
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
        files: &ProgramFiles,
        compiled: &CompiledProgram,
        bytes: u64,
        judge: bool,
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
        let mut agrees = !judge || computed_agrees(small, &outputs)?;
        drop(outputs);
        if judge && agrees {
            agrees = second_agrees(small, || {
                instance.run(&operands, 1).map_err(|error| failed(&error))
            })?;
        }

        self.spent.check_ns += nanoseconds(started.elapsed());
        Ok(Prepared {
            agrees,
            bytes,
            runs: Runs::Program { instance, operands },
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
        records: &TuningRecords,
        refuse_over: Option<u64>,
        abandon_above: Option<Duration>,
    ) -> Attempt<Measured, MeasureError> {
        let mut prepared = self.prepare(small, records, refuse_over, true)?;
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
        baseline: &TuningRecords,
        candidate: &TuningRecords,
    ) -> Attempt<Comparison, MeasureError> {
        let mut pairs = Vec::with_capacity(BASELINE_MADE_FIRST.len());
        for made_first in BASELINE_MADE_FIRST {
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
        first: &TuningRecords,
        second: &TuningRecords,
    ) -> Attempt<Option<Comparison>, MeasureError> {
        self.free_for_program(first)?;
        let mut first = self.prepare(small, first, None, false)?;
        let Ok(mut second) = self.prepare(small, second, None, false) else {
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
        baseline: &TuningRecords,
        candidate: &TuningRecords,
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
                let mut prepared = self.prepare(small, records, None, false)?;
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
    fn free_for_program(&self, records: &TuningRecords) -> Attempt<()> {
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
        records: &TuningRecords,
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

fn source_of(files: &ProgramFiles) -> String {
    format!("{}{}{}", files.header, files.source, files.manifest)
}

pub fn export_program(
    small: &SmallGraph,
    records: &TuningRecords,
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

// Whether `tensor` agrees with `expected` within `scale` times what `data_type`'s rounding allows, relative to `expected`'s largest magnitude.
//
// # Errors
//
// Returns an error if a tensor cannot be read.
pub fn agrees(
    data_type: DataType,
    scale: f64,
    tensor: &Tensor,
    expected: &Expected,
) -> Result<bool> {
    let tolerance = scale
        * expected.magnitude
        * match data_type {
            DataType::F16 => 1e-2,
            DataType::Bf16 => 4e-2,
            DataType::F32 => 1e-4,
            _ => 1e-9,
        };
    if tensor.element_count() != expected.values.element_count() {
        return Ok(false);
    }

    all_chunks(
        &host_row(tensor)?,
        Some(&expected.values),
        |value, reference| {
            value.iter().zip(reference).all(|(value, reference)| {
                value == reference
                    || (value.is_nan() && reference.is_nan())
                    || (value - reference).abs() <= tolerance
            })
        },
    )
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
        records: &TuningRecords,
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
    fn fingerprint(&mut self, records: &TuningRecords) -> Attempt<String> {
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
        records: &TuningRecords,
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
        baseline: &TuningRecords,
        candidate: &TuningRecords,
    ) -> Attempt<Comparison> {
        self.measurer
            .compare(self.whole, baseline, candidate)
            .map_err(whole_plan_failure)
    }
}
