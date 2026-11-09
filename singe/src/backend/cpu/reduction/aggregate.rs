use num_traits::Float;

use crate::{
    backend::{
        cpu::{
            iteration::typed_values,
            reduction::{enumerate_indices, flatten_indices},
        },
        spec::reduction_problem,
    },
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    parallel,
    tensor::{Tensor, TensorDataType},
};

#[derive(Clone, Copy)]
pub enum ReductionKind {
    Sum,
    Mean,
    Max,
    Min,
    Product,
    LogSumExp,
}

fn aggregate(
    input: &Tensor,
    dimensions: &[i64],
    keepdim: bool,
    kind: ReductionKind,
) -> Result<Tensor> {
    match input.data_type() {
        DataType::F32 => reduce::<f32>(input, dimensions, keepdim, kind),
        DataType::F64 => reduce::<f64>(input, dimensions, keepdim, kind),
        _ => Err(Error::unsupported_data_type("cpu reduction")),
    }
}

macro_rules! reductions {
    ($($method:ident => $kind:ident),+) => {$(
        pub fn $method(input: &Tensor, dimensions: &[i64], keepdim: bool) -> Result<Tensor> {
            aggregate(input, dimensions, keepdim, ReductionKind::$kind)
        }
    )+};
}

reductions!(sum => Sum, mean => Mean, max => Max, min => Min, product => Product, logsumexp => LogSumExp);

pub fn reduce<T: TensorDataType + Float>(
    input: &Tensor,
    dimensions: &[i64],
    keepdim: bool,
    kind: ReductionKind,
) -> Result<Tensor> {
    if input.device() != Device::Cpu {
        return Err(Error::unsupported_operation("cpu reduction device"));
    }

    let spec = reduction_problem(input.shape(), dimensions, keepdim)?;
    let values = typed_values::<T>(input)?;
    if input.shape().rank() == 0 {
        return Ok(input.clone());
    }

    let input_shape = input.shape().dimensions();
    let reduction_shape = spec
        .dimensions
        .iter()
        .map(|&axis| input_shape[axis])
        .collect::<Vec<_>>();
    let reduction_indices = enumerate_indices(&reduction_shape)?;
    if reduction_indices.is_empty() && matches!(kind, ReductionKind::Max | ReductionKind::Min) {
        return Err(Error::invalid_argument(
            "min/max reduction over an empty axis",
        ));
    }

    let init = match kind {
        ReductionKind::Sum | ReductionKind::Mean => T::zero(),
        ReductionKind::Product => T::one(),
        ReductionKind::Max | ReductionKind::LogSumExp => T::neg_infinity(),
        ReductionKind::Min => T::infinity(),
    };
    let strides = input.layout().strides();
    let start = input.layout().storage_offset();
    let reduction_offsets = reduction_indices
        .iter()
        .map(|index| {
            let mut full = vec![0_i64; input_shape.len()];
            for (&axis, &coordinate) in spec.dimensions.iter().zip(index) {
                full[axis] = coordinate;
            }
            flatten_indices(&full, strides)
        })
        .collect::<Result<Vec<_>>>()?;
    let output_offsets = enumerate_indices(spec.output_shape.dimensions())?
        .iter()
        .map(|output_index| {
            let base = expand_output_index(input_shape, &spec.dimensions, keepdim, output_index);
            flatten_indices(&base, strides).map(|offset| start + offset)
        })
        .collect::<Result<Vec<_>>>()?;
    let count = T::from(reduction_offsets.len())
        .ok_or_else(|| Error::out_of_range("mean reduction count"))?;
    let output = parallel::collect(output_offsets.len(), |output| {
        let base = output_offsets[output];
        let mut accumulator = init;
        for &offset in &reduction_offsets {
            let value = values[base + offset];
            accumulator = match kind {
                ReductionKind::Sum | ReductionKind::Mean => accumulator + value,
                ReductionKind::Product => accumulator * value,
                ReductionKind::Max | ReductionKind::Min | ReductionKind::LogSumExp
                    if accumulator.is_nan() || value.is_nan() =>
                {
                    T::nan()
                }
                ReductionKind::LogSumExp => {
                    let (largest, smallest) = if value > accumulator {
                        (value, accumulator)
                    } else {
                        (accumulator, value)
                    };
                    if largest.is_infinite() {
                        largest
                    } else {
                        largest + (smallest - largest).exp().ln_1p()
                    }
                }
                ReductionKind::Max => {
                    if value > accumulator {
                        value
                    } else {
                        accumulator
                    }
                }
                ReductionKind::Min => {
                    if value < accumulator {
                        value
                    } else {
                        accumulator
                    }
                }
            };
        }

        if matches!(kind, ReductionKind::Mean) {
            accumulator = accumulator / count;
        }
        accumulator
    });
    Tensor::from_vec(Device::Cpu, spec.output_shape, output)
}

fn expand_output_index(
    input_dimensions: &[i64],
    reduced_dimensions: &[usize],
    keepdim: bool,
    output_index: &[i64],
) -> Vec<i64> {
    let mut full = vec![0i64; input_dimensions.len()];
    let mut out_cursor = 0usize;
    for (axis, slot) in full.iter_mut().enumerate() {
        if reduced_dimensions.contains(&axis) {
            if keepdim {
                out_cursor += 1;
            }

            continue;
        }

        *slot = output_index[out_cursor];
        out_cursor += 1;
    }
    full
}
