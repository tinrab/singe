use std::collections::{HashMap, HashSet};

use crate::{
    backend::{
        cuda::{
            graph_binding::{InPlaceWrites, ScatterTarget},
            tuning::{
                reference::{Evaluated, Expected, agrees, evaluate, held_tensor},
                report::{Attempt, CandidateFailure, FailureStage, SecondReference},
            },
        },
        registry::{Operation, OperationAttributes},
    },
    data_type::DataType,
    device::Device,
    dispatch::call,
    error::{Error, Result},
    graph::{
        Graph, GraphCompileOptions,
        ir::{Graph as IrGraph, ValueId},
    },
    tensor::Tensor,
};

// Rounding builds up through a whole graph, so what a whole plan computes may sit this much further from the evaluated graph than one choice point's result.
const WHOLE_GRAPH_TOLERANCE: f64 = 4.0;

// A small graph whose inputs hold at most this many bytes has its second inputs' values evaluated on the host, which shares no library with a candidate.
const HOST_REFERENCE_BYTES: usize = 1 << 26;

// A graph candidates run, what it reads, and what it should compute.
pub struct SmallGraph {
    pub graph: Graph,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Judging {
    Compare,
    Skip,
}

pub fn judged(
    small: &SmallGraph,
    judging: Judging,
    outputs: Vec<Tensor>,
    rerun: impl FnOnce() -> Attempt<Vec<Tensor>>,
) -> Attempt<bool> {
    if judging == Judging::Skip {
        return Ok(true);
    }

    let agrees = computed_agrees(small, &outputs)?;
    drop(outputs);
    if !agrees {
        return Ok(false);
    }

    second_agrees(small, rerun)
}
