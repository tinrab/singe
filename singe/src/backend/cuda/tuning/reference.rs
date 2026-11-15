use std::collections::{HashMap, HashSet};

use half::{bf16, f16};

use crate::{
    data_type::DataType,
    device::Device,
    dispatch::call,
    error::{Error, Result},
    graph::ir::{Graph as IrGraph, ValueId},
    shape::Shape,
    tensor::Tensor,
};

// How many elements of a tensor one comparison holds as `f64` at a time, so comparing a model's caches costs the host little.
const COMPARED_ELEMENTS: usize = 1 << 18;

// An evaluated result of at least this many bytes waits on the host, as a model's written caches would otherwise fill the device.
const EVALUATED_ON_HOST_BYTES: usize = 1 << 24;

// The values of a graph, computed once, which every candidate must match.
pub struct Evaluated {
    pub values: HashMap<ValueId, Tensor>,
}

// The tensor a graph holds for `id` itself, a constant or bound state such as a model's weights.
pub fn held_tensor(ir: &IrGraph, id: ValueId) -> Option<&Tensor> {
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
    pub fn expected(&self, ir: &IrGraph, id: ValueId) -> Result<Expected> {
        Expected::create(
            self.values
                .get(&id)
                .or_else(|| held_tensor(ir, id))
                .ok_or_else(|| Error::missing_output("tuning evaluation"))?,
        )
    }
}

// Decodes `bytes`, native-endian values of `data_type`, into `out` as `f64`.
fn decode(data_type: DataType, bytes: &[u8], out: &mut Vec<f64>) -> Result<()> {
    out.clear();
    match data_type {
        DataType::F64 => out.extend(
            bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|value| f64::from_ne_bytes(*value)),
        ),
        DataType::F32 => out.extend(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|value| f64::from(f32::from_ne_bytes(*value))),
        ),
        DataType::Bf16 => out.extend(
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|value| f64::from(bf16::from_ne_bytes(*value))),
        ),
        DataType::F16 => out.extend(
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|value| f64::from(f16::from_ne_bytes(*value))),
        ),
        DataType::I64 => out.extend(
            bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|value| i64::from_ne_bytes(*value) as f64),
        ),
        DataType::I32 => out.extend(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|value| f64::from(i32::from_ne_bytes(*value))),
        ),
        DataType::Bool | DataType::U8 => out.extend(bytes.iter().map(|&value| f64::from(value))),
        // What no comparison has needed yet goes through a host cast.
        _ => {
            let count = i64::try_from(bytes.len() / data_type.size_bytes())
                .map_err(|_| Error::out_of_range("tuning comparison elements"))?;
            out.extend(
                Tensor::from_bytes(Device::Cpu, data_type, Shape::create([count])?, bytes)?
                    .cast(DataType::F64)?
                    .to_vec::<f64>()?,
            );
        }
    }
    Ok(())
}

// Calls `visit` with successive chunks of `values`, and of `other` when given, decoded to `f64`, until it returns false.
fn all_chunks(
    values: (DataType, &[u8]),
    other: Option<(DataType, &[u8])>,
    mut visit: impl FnMut(&[f64], &[f64]) -> bool,
) -> Result<bool> {
    let (mut chunk, mut other_chunk) = (Vec::new(), Vec::new());
    let count = values.1.len() / values.0.size_bytes().max(1);
    let mut start = 0;
    while start < count {
        let length = COMPARED_ELEMENTS.min(count - start);
        let size = values.0.size_bytes();
        decode(
            values.0,
            &values.1[start * size..(start + length) * size],
            &mut chunk,
        )?;
        other_chunk.clear();
        if let Some((data_type, bytes)) = other {
            let size = data_type.size_bytes();
            decode(
                data_type,
                &bytes[start * size..(start + length) * size],
                &mut other_chunk,
            )?;
        }

        if !visit(&chunk, &other_chunk) {
            return Ok(false);
        }

        start += length;
    }
    Ok(true)
}

// What a candidate's result should hold, on the host, as its bytes in its own type, so a model's caches cost no device memory and no `f64`s while they wait.
#[derive(Debug, Clone)]
pub struct Expected {
    data_type: DataType,
    bytes: Vec<u8>,
    // The largest finite magnitude it holds, at least one, which its tolerance is relative to.
    magnitude: f64,
}

impl Expected {
    pub fn create(tensor: &Tensor) -> Result<Self> {
        let bytes = tensor.to_bytes()?;
        let mut magnitude = 1.0_f64;
        all_chunks((tensor.data_type(), &bytes), None, |chunk, _| {
            for entry in chunk.iter().filter(|entry| entry.is_finite()) {
                magnitude = magnitude.max(entry.abs());
            }
            true
        })?;
        Ok(Self {
            data_type: tensor.data_type(),
            bytes,
            magnitude,
        })
    }

    // How many values it holds.
    const fn element_count(&self) -> usize {
        self.bytes.len() / self.data_type.size_bytes()
    }

    // These values with `amount` added, for a test to make a candidate wrong.
    #[cfg(test)]
    pub fn shifted(&self, amount: f64) -> Result<Self> {
        let count = i64::try_from(self.element_count())
            .map_err(|_| Error::out_of_range("tuning comparison elements"))?;
        let values = Tensor::from_bytes(
            Device::Cpu,
            self.data_type,
            Shape::create([count])?,
            &self.bytes,
        )?;
        Self::create(&values.add_scalar(amount)?)
    }
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
    if tensor.element_count() != expected.element_count() {
        return Ok(false);
    }

    let actual = tensor.to_bytes()?;
    all_chunks(
        (tensor.data_type(), &actual),
        Some((expected.data_type, &expected.bytes)),
        |value, reference| {
            value.iter().zip(reference).all(|(value, reference)| {
                value == reference
                    || (value.is_nan() && reference.is_nan())
                    || (value - reference).abs() <= tolerance
            })
        },
    )
}
