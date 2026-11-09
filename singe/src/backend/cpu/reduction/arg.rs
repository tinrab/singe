use std::cmp::Ordering;

use crate::{
    backend::cpu::{
        reduction::{
            enumerate_indices, flatten_indices,
            order::{Key, keys},
        },
        utility::cpu_device_mismatch,
    },
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    shape::{Shape, normalize_dim_usize},
    storage::cpu::CpuStorage,
    tensor::Tensor,
    utility::{to_i64, to_usize},
};

// NaN wins both reductions, as in PyTorch, and of equal values the first does.
pub fn argmax(a: &Tensor, dim: i64) -> Result<Tensor> {
    arg_reduce(a, dim, |lhs, rhs| {
        (lhs.is_nan() && !rhs.is_nan()) || (!rhs.is_nan() && lhs.compare(rhs) == Ordering::Greater)
    })
}

pub fn argmin(a: &Tensor, dim: i64) -> Result<Tensor> {
    arg_reduce(a, dim, |lhs, rhs| {
        (lhs.is_nan() && !rhs.is_nan())
            || (!lhs.is_nan() && !rhs.is_nan() && lhs.compare(rhs) == Ordering::Less)
    })
}

fn arg_reduce<F>(a: &Tensor, dim: i64, better: F) -> Result<Tensor>
where
    F: Fn(Key, Key) -> bool + Copy,
{
    if a.device() != Device::Cpu {
        return Err(cpu_device_mismatch("cpu arg reduction input", a.device()));
    }

    let input = a.contiguous()?;
    let shape = input.shape();
    let dim = normalize_dim_usize(dim, shape.rank()).map_err(|_| Error::DimensionOutOfBounds)?;

    let mut output_dims = shape.dimensions().to_vec();
    output_dims.remove(dim);
    let output_shape = Shape::create(output_dims)?;

    let input_data = keys(&input)?;
    let input_strides = input.layout().strides();
    let output_indices = enumerate_indices(output_shape.dimensions())?;
    let reduce_dim = to_usize(shape.dimensions()[dim], "cpu arg reduction dimension")?;
    if reduce_dim == 0 {
        return Err(Error::invalid_argument("cpu arg reduction dimension"));
    }

    let mut output = Vec::with_capacity(output_indices.len());
    for out_index in output_indices {
        let rank = to_usize(shape.rank(), "cpu arg reduction rank")?;
        let mut full_index = vec![0i64; rank];
        let mut out_cursor = 0usize;
        for (axis, slot) in full_index.iter_mut().enumerate() {
            if axis == dim {
                continue;
            }

            *slot = out_index[out_cursor];
            out_cursor += 1;
        }

        let mut best_index = 0usize;
        full_index[dim] = 0;
        let mut best_value = input_data[flatten_indices(&full_index, input_strides)?];
        for candidate in 1..reduce_dim {
            full_index[dim] = to_i64(candidate, "cpu arg reduction candidate index")?;
            let value = input_data[flatten_indices(&full_index, input_strides)?];
            if better(value, best_value) {
                best_value = value;
                best_index = candidate;
            }
        }

        output.push(to_i64(best_index, "cpu arg reduction output index")?);
    }

    drop(input_data);

    let storage = CpuStorage::from_bytes(
        DataType::I64,
        output
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;
    // SAFETY: the storage was built from values computed for exactly this shape and data type.
    Ok(unsafe { Tensor::from_parts_unchecked(storage, DataType::I64, output_shape) })
}
