use std::iter::once;

use singe_codegen::interpret::{E4M3, E5M2};

use crate::{
    backend::{
        cpu::storage::assign,
        spec::{MultiIndexProblem, multi_index_problem},
    },
    data_type::{DataType, bf16, f16},
    device::Device,
    error::{Error, Result},
    tensor::Tensor,
    utility::{checked_element_count, to_usize},
};

/// Selects elements of the consecutive dimensions starting at `dim` with broadcast `I64` index tensors.
pub fn index(input: &Tensor, indices: &[Tensor], dim: i64) -> Result<Tensor> {
    let (problem, positions) = plan(input, indices, dim, "cpu index operand")?;
    let element = input.data_type().size_bytes();
    let block = problem.inner_size * element;
    let indexed = indexed_count(&problem)?;
    let bytes = input.to_bytes()?;
    let mut output = Vec::with_capacity(problem.output_shape.element_count() * element);
    for outer in 0..problem.outer_size {
        for &position in &positions {
            let start = (outer * indexed + position) * block;
            output.extend_from_slice(&bytes[start..start + block]);
        }
    }
    Tensor::from_bytes(
        Device::Cpu,
        input.data_type(),
        problem.output_shape,
        &output,
    )
}

/// Returns a copy of `input` with `values` written into, or added to, the elements `index` selects.
///
/// When positions repeat without `accumulate`, the last value in index order wins.
pub fn index_put(
    input: &Tensor,
    values: &Tensor,
    indices: &[Tensor],
    dim: i64,
    accumulate: bool,
) -> Result<Tensor> {
    let (problem, positions) = plan(input, indices, dim, "cpu index put operand")?;
    if values.device() != Device::Cpu {
        return Err(Error::device_mismatch(
            "cpu index put values",
            Device::Cpu,
            values.device(),
        ));
    }

    if accumulate {
        return accumulate_into(input, values, &problem, &positions);
    }

    let element = input.data_type().size_bytes();
    let block = problem.inner_size * element;
    let indexed = indexed_count(&problem)?;
    let mut output = input.to_bytes()?;
    let source = values.to_bytes()?;
    for outer in 0..problem.outer_size {
        for (selected, &position) in positions.iter().enumerate() {
            let to = (outer * indexed + position) * block;
            let from = (outer * positions.len() + selected) * block;
            output[to..to + block].copy_from_slice(&source[from..from + block]);
        }
    }
    Tensor::from_bytes(
        Device::Cpu,
        input.data_type(),
        input.shape().clone(),
        &output,
    )
}

// Writes `values` into the elements `indices` selects of `destination`, where they are when it is contiguous, else through a copy it is then assigned.
pub fn index_write(
    destination: &Tensor,
    values: &Tensor,
    indices: &[Tensor],
    dim: i64,
    accumulate: bool,
) -> Result<()> {
    if accumulate || !destination.is_contiguous() {
        let updated = index_put(destination, values, indices, dim, accumulate)?;
        return assign(destination, &updated);
    }

    let (problem, positions) = plan(destination, indices, dim, "cpu index write operand")?;
    if values.device() != Device::Cpu {
        return Err(Error::device_mismatch(
            "cpu index write values",
            Device::Cpu,
            values.device(),
        ));
    }

    let element = destination.data_type().size_bytes();
    let block = problem.inner_size * element;
    let indexed = indexed_count(&problem)?;
    let source = values.to_bytes()?;
    let storage = destination
        .storage()
        .as_cpu()
        .ok_or_else(|| Error::unsupported_operation("cpu index write storage"))?;
    let mut bytes = storage.write_bytes()?;
    let start = destination.layout().storage_offset() * element;
    for outer in 0..problem.outer_size {
        for (selected, &position) in positions.iter().enumerate() {
            let to = start + (outer * indexed + position) * block;
            let from = (outer * positions.len() + selected) * block;
            bytes[to..to + block].copy_from_slice(&source[from..from + block]);
        }
    }

    drop(bytes);
    Ok(())
}

fn accumulate_into(
    input: &Tensor,
    values: &Tensor,
    problem: &MultiIndexProblem,
    positions: &[usize],
) -> Result<Tensor> {
    // 8-bit floats add in a wider type and round each sum, as PyTorch adds them.
    let format = match input.data_type() {
        DataType::F8E4M3 => Some(E4M3),
        DataType::F8E5M2 => Some(E5M2),
        _ => None,
    };
    if let Some(format) = format {
        let mut output = input.to_bytes()?;
        let source = values.to_bytes()?;
        let inner = problem.inner_size;
        let indexed = indexed_count(problem)?;
        for outer in 0..problem.outer_size {
            for (selected, &position) in positions.iter().enumerate() {
                let to = (outer * indexed + position) * inner;
                let from = (outer * positions.len() + selected) * inner;
                for offset in 0..inner {
                    let sum =
                        format.decode(output[to + offset]) + format.decode(source[from + offset]);
                    output[to + offset] = format.encode(sum);
                }
            }
        }
        return Tensor::from_bytes(
            Device::Cpu,
            input.data_type(),
            input.shape().clone(),
            &output,
        );
    }

    macro_rules! accumulate {
        ($($ty:ty => $variant:ident, $combine:expr),+ $(,)?) => {
            match input.data_type() {
                $(DataType::$variant => {
                    let mut output = input.to_vec::<$ty>()?;
                    let source = values.to_vec::<$ty>()?;
                    let inner = problem.inner_size;
                    let indexed = indexed_count(problem)?;
                    for outer in 0..problem.outer_size {
                        for (selected, &position) in positions.iter().enumerate() {
                            let to = (outer * indexed + position) * inner;
                            let from = (outer * positions.len() + selected) * inner;
                            for offset in 0..inner {
                                output[to + offset] = $combine(output[to + offset], source[from + offset]);
                            }
                        }
                    }
                    Tensor::from_vec(Device::Cpu, input.shape().clone(), output)
                },)+
                _ => Err(Error::unsupported_data_type("cpu index put accumulation")),
            }
        };
    }
    // Integer sums wrap and Boolean sums are disjunctions, as PyTorch's accumulations are.
    accumulate!(
        bool => Bool, |a: bool, b: bool| a || b,
        u8 => U8, u8::wrapping_add,
        i8 => I8, i8::wrapping_add,
        u16 => U16, u16::wrapping_add,
        i16 => I16, i16::wrapping_add,
        u32 => U32, u32::wrapping_add,
        i32 => I32, i32::wrapping_add,
        u64 => U64, u64::wrapping_add,
        i64 => I64, i64::wrapping_add,
        f16 => F16, |a: f16, b: f16| a + b,
        bf16 => Bf16, |a: bf16, b: bf16| a + b,
        f32 => F32, |a: f32, b: f32| a + b,
        f64 => F64, |a: f64, b: f64| a + b,
    )
}

/// Checks the operands and returns each broadcast index position's linear offset within the indexed dimensions.
fn plan(
    input: &Tensor,
    indices: &[Tensor],
    dim: i64,
    context: &'static str,
) -> Result<(MultiIndexProblem, Vec<usize>)> {
    for tensor in once(input).chain(indices) {
        if tensor.device() != Device::Cpu {
            return Err(Error::device_mismatch(
                context,
                Device::Cpu,
                tensor.device(),
            ));
        }
    }

    let shapes = indices.iter().map(Tensor::shape).collect::<Vec<_>>();
    let problem = multi_index_problem(input.shape(), &shapes, dim)?;
    let mut positions = vec![0_usize; problem.index_shape.element_count()];
    let mut scale = 1_usize;
    for (index, &size) in indices.iter().zip(&problem.indexed_dimensions).rev() {
        let values = index.expand(&problem.index_shape)?.to_vec::<i64>()?;
        for (position, value) in positions.iter_mut().zip(values) {
            let value = if value < 0 { value + size } else { value };
            if !(0..size).contains(&value) {
                return Err(Error::IndexOutOfBounds);
            }

            *position += usize::try_from(value).map_err(|_| Error::IndexOutOfBounds)? * scale;
        }

        scale *= to_usize(size, "index dimension")?;
    }
    Ok((problem, positions))
}

fn indexed_count(problem: &MultiIndexProblem) -> Result<usize> {
    checked_element_count(&problem.indexed_dimensions, "cpu indexed elements")
}
