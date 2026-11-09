use std::ops::Deref;

use num_traits::Float;

use crate::{
    backend::{
        cpu::iteration::{ElementwisePlan, typed_values},
        spec::BinaryBroadcastProblem,
    },
    data_type::{DataType, bf16, f16},
    device::Device,
    error::{Error, Result},
    layout::Layout,
    storage::cpu::CpuSlice,
    tensor::{Tensor, TensorDataType},
    utility::{checked_add, checked_mul, to_i64, to_usize},
};

#[derive(Debug)]
pub struct BroadcastAccess {
    dimensions: Vec<i64>,
    strides: Vec<i64>,
}

pub fn broadcast_access(layout: &Layout, spec: &BinaryBroadcastProblem) -> BroadcastAccess {
    let rank = spec.output_shape.dimensions().len();
    BroadcastAccess {
        dimensions: spec.lhs_padded_dimensions.clone(),
        strides: padded_strides(layout.strides(), rank),
    }
}

pub fn broadcast_access_swapped(layout: &Layout, spec: &BinaryBroadcastProblem) -> BroadcastAccess {
    let rank = spec.output_shape.dimensions().len();
    BroadcastAccess {
        dimensions: spec.rhs_padded_dimensions.clone(),
        strides: padded_strides(layout.strides(), rank),
    }
}

fn padded_strides(strides: &[i64], len: usize) -> Vec<i64> {
    let pad = len - strides.len();
    let mut out = vec![0; pad];
    out.extend_from_slice(strides);
    out
}

pub fn linear_to_indices(mut linear: usize, dimensions: &[i64]) -> Result<Vec<i64>> {
    let mut indices = vec![0; dimensions.len()];
    for axis in (0..dimensions.len()).rev() {
        let dim = to_usize(dimensions[axis], &format!("cpu linear index axis {axis}"))?;
        if dim == 0 {
            return Err(Error::invalid_argument(format!(
                "cpu linear index axis {axis} dimension"
            )));
        }

        indices[axis] = to_i64(linear % dim, &format!("cpu linear index axis {axis}"))?;
        linear /= dim;
    }
    Ok(indices)
}

pub fn broadcast_offset_for_spec(indices: &[i64], access: &BroadcastAccess) -> Result<usize> {
    broadcast_offset(indices, &access.dimensions, &access.strides)
}

pub fn broadcast_offset(indices: &[i64], dimensions: &[i64], strides: &[i64]) -> Result<usize> {
    indices
        .iter()
        .zip(dimensions.iter().zip(strides.iter()))
        .enumerate()
        .try_fold(0usize, |acc, (axis, (&index, (&dim, &stride)))| {
            let index = if dim == 1 { 0 } else { index };
            let index = to_usize(index, &format!("cpu broadcast index axis {axis}"))?;
            let stride = usize::try_from(stride).map_err(|_| Error::InvalidStride)?;
            let offset = checked_mul(index, stride, "cpu broadcast offset")?;
            checked_add(acc, offset, "cpu broadcast offset")
        })
}

/// Logical-order values of a tensor: a locked view of contiguous storage, or a gathered copy.
#[derive(Debug)]
pub enum LogicalValues<'a, T> {
    Contiguous {
        values: CpuSlice<'a, T>,
        start: usize,
        length: usize,
    },
    Gathered(Vec<T>),
}

impl<T> Deref for LogicalValues<'_, T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        match self {
            Self::Contiguous {
                values,
                start,
                length,
            } => &values[*start..*start + *length],
            Self::Gathered(values) => values,
        }
    }
}

pub fn get_f32_data(tensor: &Tensor) -> Result<LogicalValues<'_, f32>> {
    logical_values(tensor)
}

// The floating types the native CPU kernels of convolution, pooling, and attention compute in.
pub trait NativeFloat: Float + TensorDataType + Send + Sync {}

impl<T: Float + TensorDataType + Send + Sync> NativeFloat for T {}

pub fn logical_values<T: TensorDataType>(tensor: &Tensor) -> Result<LogicalValues<'_, T>> {
    let values = typed_values::<T>(tensor)?;
    if tensor.is_contiguous() {
        return Ok(LogicalValues::Contiguous {
            values,
            start: tensor.layout().storage_offset(),
            length: tensor.shape().element_count(),
        });
    }
    Ok(LogicalValues::Gathered(
        (0..tensor.shape().element_count())
            .map(|linear| values[tensor.layout().linear_offset(linear)])
            .collect(),
    ))
}

/// Applies `single` to `F16`, `BF16`, and `F32` elements and `double` to `F64` elements; the half types compute in `f32`.
pub fn floating_unary(
    input: &Tensor,
    single: impl Fn(f32) -> f32 + Sync,
    double: impl Fn(f64) -> f64 + Sync,
) -> Result<Tensor> {
    let plan = ElementwisePlan::create(&[input])?;
    match input.data_type() {
        DataType::F16 => map_unary(input, &plan, |x: f16| f16::from_f32(single(x.to_f32()))),
        DataType::Bf16 => map_unary(input, &plan, |x: bf16| bf16::from_f32(single(x.to_f32()))),
        DataType::F32 => map_unary(input, &plan, single),
        DataType::F64 => map_unary(input, &plan, double),
        _ => Err(Error::unsupported_data_type(
            "cpu floating-point unary operation",
        )),
    }
}

/// Applies `single` or `double` to broadcast pairs of one floating data type, as [`floating_unary`] does.
pub fn floating_binary(
    lhs: &Tensor,
    rhs: &Tensor,
    single: impl Fn(f32, f32) -> f32 + Sync,
    double: impl Fn(f64, f64) -> f64 + Sync,
) -> Result<Tensor> {
    if lhs.data_type() != rhs.data_type() {
        return Err(Error::data_type_mismatch(
            "cpu floating-point binary rhs",
            lhs.data_type(),
            rhs.data_type(),
        ));
    }

    let plan = ElementwisePlan::create(&[lhs, rhs])?;
    match lhs.data_type() {
        DataType::F16 => map_binary(lhs, rhs, &plan, |x: f16, y: f16| {
            f16::from_f32(single(x.to_f32(), y.to_f32()))
        }),
        DataType::Bf16 => map_binary(lhs, rhs, &plan, |x: bf16, y: bf16| {
            bf16::from_f32(single(x.to_f32(), y.to_f32()))
        }),
        DataType::F32 => map_binary(lhs, rhs, &plan, single),
        DataType::F64 => map_binary(lhs, rhs, &plan, double),
        _ => Err(Error::unsupported_data_type(
            "cpu floating-point binary operation",
        )),
    }
}

fn map_unary<T: TensorDataType + Copy + Send + Sync + 'static>(
    input: &Tensor,
    plan: &ElementwisePlan,
    op: impl Fn(T) -> T + Sync,
) -> Result<Tensor> {
    let values = typed_values::<T>(input)?;
    let output = plan.collect(|linear| op(values[plan.offset(0, linear)]));
    Tensor::from_vec(Device::Cpu, plan.shape(), output)
}

fn map_binary<T: TensorDataType + Copy + Send + Sync + 'static>(
    lhs: &Tensor,
    rhs: &Tensor,
    plan: &ElementwisePlan,
    op: impl Fn(T, T) -> T + Sync,
) -> Result<Tensor> {
    let lhs = typed_values::<T>(lhs)?;
    let rhs = typed_values::<T>(rhs)?;
    let output =
        plan.collect(|linear| op(lhs[plan.offset(0, linear)], rhs[plan.offset(1, linear)]));
    Tensor::from_vec(Device::Cpu, plan.shape(), output)
}

pub fn cpu_device_mismatch(name: impl Into<String>, actual: Device) -> Error {
    Error::device_mismatch(name.into(), Device::Cpu, actual)
}
