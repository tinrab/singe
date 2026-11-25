// The first value where an export disagrees with the program it was written from, as PyTorch's `_VerificationInterpreter` finds one: the program runs natively keeping every value, and its export runs through Singe's ONNX importer with every value it names as an output, both on the same inputs.

use std::{
    collections::{HashMap, HashSet},
    hash::BuildHasher,
};

use crate::{
    data_type::DataType,
    device::Device,
    graph::{GraphProgram, ProgramInputBinding, StateSnapshot},
    onnx::{
        error::Result,
        exporter::{
            Exported, exported,
            external::encoded,
            proto::{unsupported, value_info},
        },
        model::OnnxModel,
        options::OnnxOptions,
    },
    tensor::Tensor,
};

#[derive(Debug, Clone, PartialEq)]
pub struct Divergence {
    pub name: String,
    pub operation: String,
    pub difference: Option<f64>,
    pub allowed: f64,
}

pub fn first_divergence<S: BuildHasher>(
    program: &GraphProgram,
    state: &StateSnapshot,
    inputs: &HashMap<String, Tensor, S>,
    tolerance: f64,
) -> Result<Option<Divergence>> {
    let Exported {
        mut model,
        graph: source,
        ir,
        names,
    } = exported(program, state)?;
    let graph = model
        .graph
        .as_mut()
        .ok_or_else(|| unsupported("an export without a graph"))?;
    let produced = graph
        .node
        .iter()
        .flat_map(|node| node.output.iter().cloned())
        .collect::<HashSet<_>>();
    let mut listed = graph
        .output
        .iter()
        .map(|output| output.name.clone())
        .collect::<HashSet<_>>();
    // Every value an operation computes and a node of the export writes becomes an output of the export.
    let mut compared = HashSet::new();
    for (&value, name) in &names {
        if !produced.contains(name) || ir.required_operation_for_value(value)?.opcode().is_none() {
            continue;
        }

        if listed.insert(name.clone()) {
            let rank = ir.required_value_shape(value)?.dimensions().len();
            let data_type = ir.required_value_data_type(value)?;
            graph
                .output
                .push(value_info(name.clone(), data_type, vec![None; rank])?);
        }

        compared.insert(value);
    }

    let imported = OnnxModel::from_reader(encoded(&model)?.as_slice(), OnnxOptions::new())?;
    let inputs = inputs
        .iter()
        .map(|(name, tensor)| (name.clone(), tensor.clone()))
        .collect::<HashMap<_, _>>();
    let exported = imported.run(&inputs)?;
    let mut native_inputs = inputs;
    for input in program.signature().inputs() {
        if let ProgramInputBinding::State(_) = input.binding() {
            let tensor = state
                .tensor(input.name())
                .ok_or_else(|| unsupported(format!("missing state `{}`", input.name())))?;
            native_inputs.insert(input.name().to_owned(), tensor.to_device(Device::Cpu)?);
        }
    }

    // The program may have been built on metadata, which runs no kernel.
    let (ir, order, native) = source
        .placed_on(Device::Cpu)?
        .run_keeping_values(&native_inputs)?;
    for id in order {
        let operation = ir.required_operation(id)?;
        for value in operation.result_values() {
            let (Some(name), Some(native)) = (names.get(&value), native.get(&value)) else {
                continue;
            };
            let Some(exported) = exported.get(name).filter(|_| compared.contains(&value)) else {
                continue;
            };
            if let Some((difference, allowed)) = differ(native, exported, tolerance)? {
                let opcode = operation
                    .opcode()
                    .map_or_else(|| "a constant".to_owned(), |opcode| format!("{opcode:?}"));
                return Ok(Some(Divergence {
                    name: name.clone(),
                    operation: operation.source_location().map_or_else(
                        || opcode.clone(),
                        |location| format!("{opcode} at {location}"),
                    ),
                    difference,
                    allowed,
                }));
            }
        }
    }
    Ok(None)
}

fn differ(
    native: &Tensor,
    exported: &Tensor,
    tolerance: f64,
) -> Result<Option<(Option<f64>, f64)>> {
    if native.shape() != exported.shape() {
        return Ok(Some((None, 0.0)));
    }

    let values = |tensor: &Tensor| -> Result<Vec<f64>> {
        Ok(tensor
            .to_device(Device::Cpu)?
            .cast(DataType::F64)?
            .contiguous()?
            .to_vec::<f64>()?)
    };
    let (native, exported) = (values(native)?, values(exported)?);
    let largest = native
        .iter()
        .filter(|value| value.is_finite())
        .fold(1.0_f64, |largest, value| largest.max(value.abs()));
    let allowed = tolerance * largest;
    let difference = native
        .iter()
        .zip(&exported)
        .map(
            |(native, exported)| match (native.is_nan(), exported.is_nan()) {
                (true, true) => 0.0,
                (false, false) if native == exported => 0.0,
                (false, false) => (native - exported).abs(),
                _ => f64::INFINITY,
            },
        )
        .fold(0.0_f64, f64::max);
    Ok((difference > allowed).then_some((Some(difference), allowed)))
}
