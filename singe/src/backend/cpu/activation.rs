use std::{f32::consts::SQRT_2, f64::consts::SQRT_2 as SQRT_2_F64};

use libm::{erf, erff};

use crate::{
    backend::cpu::utility::{cpu_device_mismatch, floating_unary, get_f32_data},
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    parallel,
    shape::normalize_dim_usize,
    storage::cpu::CpuStorage,
    tensor::Tensor,
    utility::{checked_add, checked_mul, to_usize},
};

/// Defines activations whose expression is the same for `f32` and `f64`.
macro_rules! activations {
    ($($name:ident($value:ident) => $expression:expr),+ $(,)?) => {
        $(pub fn $name(input: &Tensor) -> Result<Tensor> {
            floating_unary(input, |$value: f32| $expression, |$value: f64| $expression)
        })+
    };
}

activations!(
    relu(x) => if x < 0.0 { 0.0 } else { x },
    silu(x) => x / (1.0 + (-x).exp()),
    sigmoid(x) => 1.0 / (1.0 + (-x).exp()),
);

pub fn gelu_exact(input: &Tensor) -> Result<Tensor> {
    floating_unary(
        input,
        |x| 0.5 * x * (1.0 + erff(x / SQRT_2)),
        |x| 0.5 * x * (1.0 + erf(x / SQRT_2_F64)),
    )
}

pub fn gelu(input: &Tensor) -> Result<Tensor> {
    const SQRT_2_OVER_PI_F32: f32 = 0.797_884_6;
    const SQRT_2_OVER_PI_F64: f64 = 0.797_884_560_802_865_4;
    floating_unary(
        input,
        |x| 0.5 * x * (1.0 + (SQRT_2_OVER_PI_F32 * (0.044_715 * x * x).mul_add(x, x)).tanh()),
        |x| 0.5 * x * (1.0 + (SQRT_2_OVER_PI_F64 * (0.044_715 * x * x).mul_add(x, x)).tanh()),
    )
}

pub fn softmax(a: &Tensor, dim: i64) -> Result<Tensor> {
    softmax_impl(a, dim, false)
}

pub fn log_softmax(a: &Tensor, dim: i64) -> Result<Tensor> {
    softmax_impl(a, dim, true)
}

fn softmax_impl(a: &Tensor, dim: i64, log_output: bool) -> Result<Tensor> {
    if a.device() != Device::Cpu {
        return Err(cpu_device_mismatch("cpu softmax input", a.device()));
    }

    let input = a.contiguous()?;
    let dims = input.shape().dimensions();
    let rank = input.shape().rank();
    let dim = normalize_dim_usize(dim, rank).map_err(|_| Error::DimensionOutOfBounds)?;

    let input_data = get_f32_data(&input)?;
    if input_data.is_empty() {
        let storage = CpuStorage::from_bytes(DataType::F32, Vec::new())?;
        // SAFETY: the storage was built from values computed for exactly this shape and data type.
        return Ok(unsafe {
            Tensor::from_parts_unchecked(storage, DataType::F32, input.shape().clone())
        });
    }

    let dim_size = to_usize(dims[dim], "cpu softmax dimension")?;
    if dim_size == 0 {
        let storage = CpuStorage::from_bytes(DataType::F32, Vec::new())?;
        // SAFETY: the storage was built from values computed for exactly this shape and data type.
        return Ok(unsafe {
            Tensor::from_parts_unchecked(storage, DataType::F32, input.shape().clone())
        });
    }

    let input_strides = input.layout().strides();
    let total_iters = dims
        .iter()
        .enumerate()
        .filter(|(axis, _)| *axis != dim)
        .try_fold(1usize, |acc, (_, &dimension)| {
            let dimension = to_usize(dimension, "cpu softmax iteration dimension")?;
            checked_mul(acc, dimension, "cpu softmax iteration count")
        })?;

    let mut output_data = vec![0.0_f32; input_data.len()];

    execute_softmax(
        input_data.as_ref(),
        input_strides,
        dims,
        dim,
        dim_size,
        total_iters,
        log_output,
        &mut output_data,
    )?;
    drop(input_data);

    Tensor::from_vec(Device::Cpu, input.shape().clone(), output_data)
}

pub fn execute_softmax(
    input_data: &[f32],
    input_strides: &[i64],
    dimensions: &[i64],
    dim: usize,
    dim_size: usize,
    total_iters: usize,
    log_output: bool,
    output_data: &mut [f32],
) -> Result<()> {
    let rank = dimensions.len();
    let dim_stride = usize::try_from(input_strides[dim]).map_err(|_| Error::InvalidStride)?;
    let mut indices = vec![0usize; rank];
    let mut bases = Vec::with_capacity(total_iters);
    for _ in 0..total_iters {
        bases.push(flat_offset(&indices, input_strides)?);
        increment_softmax_indices(&mut indices, dimensions, dim)?;
    }

    let rows =
        parallel::collect_with_grain(bases.len(), parallel::GRAIN_SIZE / dim_size.max(1), |row| {
            softmax_row(input_data, bases[row], dim_size, dim_stride, log_output)
        });
    for (&base, row) in bases.iter().zip(rows) {
        for (i, value) in row?.into_iter().enumerate() {
            output_data[softmax_axis_offset(base, i, dim_stride)?] = value;
        }
    }
    Ok(())
}

/// Returns the softmax or log-softmax of the row at `base_offset`, in axis order.
fn softmax_row(
    input_data: &[f32],
    base_offset: usize,
    dim_size: usize,
    dim_stride: usize,
    log_output: bool,
) -> Result<Vec<f32>> {
    let mut max_val = f32::NEG_INFINITY;
    for i in 0..dim_size {
        let idx = softmax_axis_offset(base_offset, i, dim_stride)?;
        max_val = max_val.max(input_data[idx]);
    }

    let mut sum_exp = 0.0_f32;
    for i in 0..dim_size {
        let idx = softmax_axis_offset(base_offset, i, dim_stride)?;
        sum_exp += (input_data[idx] - max_val).exp();
    }

    let log_sum_exp = sum_exp.ln();

    (0..dim_size)
        .map(|i| {
            let shifted = input_data[softmax_axis_offset(base_offset, i, dim_stride)?] - max_val;
            Ok(if log_output {
                shifted - log_sum_exp
            } else {
                shifted.exp() / sum_exp
            })
        })
        .collect()
}

fn increment_softmax_indices(indices: &mut [usize], dimensions: &[i64], dim: usize) -> Result<()> {
    let mut carry = true;
    for d in (0..indices.len()).rev() {
        if d == dim {
            continue;
        }

        if !carry {
            break;
        }

        indices[d] = checked_add(indices[d], 1, "cpu softmax index")?;
        let dimension = to_usize(dimensions[d], "cpu softmax iteration dimension")?;
        if indices[d] < dimension {
            carry = false;
        } else {
            indices[d] = 0;
        }
    }
    Ok(())
}

fn softmax_axis_offset(base_offset: usize, index: usize, dim_stride: usize) -> Result<usize> {
    let offset = checked_mul(index, dim_stride, "cpu softmax axis offset")?;
    checked_add(base_offset, offset, "cpu softmax axis offset")
}

fn flat_offset(indices: &[usize], strides: &[i64]) -> Result<usize> {
    indices
        .iter()
        .zip(strides)
        .try_fold(0usize, |acc, (&index, &stride)| {
            let stride = usize::try_from(stride).map_err(|_| Error::InvalidStride)?;
            let offset = checked_mul(index, stride, "cpu softmax flat offset")?;
            checked_add(acc, offset, "cpu softmax flat offset")
        })
}
