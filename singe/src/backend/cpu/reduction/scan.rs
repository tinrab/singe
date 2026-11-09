use crate::{
    backend::{
        cpu::{reduction::order::keys, utility::cpu_device_mismatch},
        registry::Operation,
    },
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    shape::normalize_dim_usize,
    tensor::Tensor,
    utility::to_i64,
};

/// Slices of a tensor along one dimension, addressed in its row-major element order.
struct Slices {
    outer: usize,
    length: usize,
    inner: usize,
}

impl Slices {
    fn create(input: &Tensor, dim: i64, context: &'static str) -> Result<Self> {
        if input.device() != Device::Cpu {
            return Err(cpu_device_mismatch(context, input.device()));
        }

        let dimensions = input.shape().dimensions();
        if dimensions.is_empty() {
            return if matches!(dim, 0 | -1) {
                Ok(Self {
                    outer: 1,
                    length: 1,
                    inner: 1,
                })
            } else {
                Err(Error::DimensionOutOfBounds)
            };
        }

        let axis = normalize_dim_usize(dim, input.shape().rank())?;
        let size = |dimensions: &[i64]| {
            dimensions.iter().try_fold(1_usize, |product, &dimension| {
                usize::try_from(dimension)
                    .ok()
                    .and_then(|dimension| product.checked_mul(dimension))
                    .ok_or_else(|| Error::out_of_range("cpu scan extent"))
            })
        };
        Ok(Self {
            outer: size(&dimensions[..axis])?,
            length: size(&dimensions[axis..=axis])?,
            inner: size(&dimensions[axis + 1..])?,
        })
    }

    /// Returns the row-major offset of position `index` of the slice that starts at row-major offset `base`.
    const fn offset(&self, base: usize, index: usize) -> usize {
        base + index * self.inner
    }

    /// Returns the row-major offset of the first element of every slice.
    fn bases(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.outer).flat_map(move |outer| {
            (0..self.inner).map(move |inner| outer * self.length * self.inner + inner)
        })
    }
}

/// Orders values ascending with NaN largest, so descending order puts NaN first.
pub fn argsort(input: &Tensor, dim: i64, descending: bool) -> Result<Tensor> {
    let slices = Slices::create(input, dim, "cpu argsort input")?;
    let values = keys(input)?;
    let mut output = vec![0_i64; values.len()];
    let mut order = (0..slices.length).collect::<Vec<_>>();
    for base in slices.bases() {
        for (position, entry) in order.iter_mut().enumerate() {
            *entry = position;
        }

        // The sort is stable, so equal values keep their order either way.
        order.sort_by(|&lhs, &rhs| {
            let ordering =
                values[slices.offset(base, lhs)].compare(values[slices.offset(base, rhs)]);
            if descending {
                ordering.reverse()
            } else {
                ordering
            }
        });
        for (position, &index) in order.iter().enumerate() {
            output[slices.offset(base, position)] = to_i64(index, "cpu argsort index")?;
        }
    }
    Tensor::from_vec(Device::Cpu, input.shape().clone(), output)
}

pub fn cumsum(input: &Tensor, dim: i64) -> Result<Tensor> {
    cumulative(input, dim, Operation::Cumsum)
}

pub fn cumprod(input: &Tensor, dim: i64) -> Result<Tensor> {
    cumulative(input, dim, Operation::Cumprod)
}

// Running sums or products along `dim`, as `operation` names.
fn cumulative(input: &Tensor, dim: i64, operation: Operation) -> Result<Tensor> {
    fn accumulated<T: Copy>(
        mut values: Vec<T>,
        slices: &Slices,
        identity: T,
        combine: impl Fn(T, T) -> T,
    ) -> Vec<T> {
        for base in slices.bases() {
            let mut running = identity;
            for index in 0..slices.length {
                let offset = slices.offset(base, index);
                running = combine(running, values[offset]);
                values[offset] = running;
            }
        }
        values
    }

    let slices = Slices::create(input, dim, "cpu cumulative input")?;
    let shape = input.shape().clone();
    let product = operation == Operation::Cumprod;
    match input.data_type() {
        DataType::F32 => {
            let (identity, combine): (f32, fn(f32, f32) -> f32) = if product {
                (1.0, |a, b| a * b)
            } else {
                (0.0, |a, b| a + b)
            };
            Tensor::from_vec(
                Device::Cpu,
                shape,
                accumulated(input.to_vec::<f32>()?, &slices, identity, combine),
            )
        }
        DataType::F64 => {
            let (identity, combine): (f64, fn(f64, f64) -> f64) = if product {
                (1.0, |a, b| a * b)
            } else {
                (0.0, |a, b| a + b)
            };
            Tensor::from_vec(
                Device::Cpu,
                shape,
                accumulated(input.to_vec::<f64>()?, &slices, identity, combine),
            )
        }
        // Inference makes integer and Boolean scans `I64`, which wrap as PyTorch's do.
        DataType::I64 => {
            let (identity, combine): (i64, fn(i64, i64) -> i64) = if product {
                (1, i64::wrapping_mul)
            } else {
                (0, i64::wrapping_add)
            };
            Tensor::from_vec(
                Device::Cpu,
                shape,
                accumulated(input.to_vec::<i64>()?, &slices, identity, combine),
            )
        }
        // Half-precision scans accumulate in `F32` and round each running value, as PyTorch's do.
        data_type if data_type.is_float() => {
            cumulative(&input.cast(DataType::F32)?, dim, operation)?.cast(data_type)
        }
        _ => Err(Error::unsupported_data_type("cpu cumulative input")),
    }
}
