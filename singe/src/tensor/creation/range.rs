use num_traits::ToPrimitive;

use crate::{
    device::Device,
    error::{Error, Result},
    shape::Shape,
    tensor::{Tensor, TensorDataType},
    utility::{to_i64, to_usize},
};

/// Element types supported by [`Tensor::arange`]: FP32, FP64, and I64.
pub trait TensorRangeElement: TensorDataType + private::Range {}

/// Element types supported by [`Tensor::linspace`]: FP32 and FP64.
pub trait TensorLinspaceElement: TensorDataType + private::Linspace {}

impl Tensor {
    /// Creates a typed vector in `[start, end)` with the given nonzero step.
    /// Bounds must agree with the step direction.
    /// Equal bounds produce an empty vector.
    /// I64 ranges preserve integer precision; floating ranges use FP64 index arithmetic.
    ///
    /// # Errors
    /// Returns an error for invalid bounds, a zero/nonfinite step, size overflow, or allocation/device failure.
    pub fn arange<T: TensorRangeElement>(
        device: Device,
        start: T,
        end: T,
        step: T,
    ) -> Result<Self> {
        let count = T::length(start, end, step)?;
        let (shape, mut data) = allocate_values::<T>(count)?;
        for index in 0..count {
            data.push(T::range_value(start, step, index)?);
        }
        Self::from_vec(device, shape, data)
    }

    /// Creates a typed vector of `steps` evenly spaced values including both endpoints.
    /// Zero steps produce an empty vector; one step returns `start`.
    /// Floating bounds must be finite.
    /// Endpoints retain their exact source bits.
    ///
    /// # Errors
    /// Returns an error for nonfinite bounds, size overflow, or allocation/device failure.
    pub fn linspace<T: TensorLinspaceElement>(
        device: Device,
        start: T,
        end: T,
        steps: usize,
    ) -> Result<Self> {
        T::validate(start, end)?;
        let (shape, mut data) = allocate_values::<T>(steps)?;
        for index in 0..steps {
            data.push(if index == 0 {
                start
            } else if index + 1 == steps {
                end
            } else {
                T::interpolate(start, end, index, steps - 1)
            });
        }
        Self::from_vec(device, shape, data)
    }
}

fn allocate_values<T: TensorDataType>(count: usize) -> Result<(Shape, Vec<T>)> {
    let shape = Shape::create([to_i64(count, "tensor range length")?])?;
    shape.byte_size(T::data_type().size_bytes())?;
    let mut data = Vec::new();
    data.try_reserve_exact(count)?;
    Ok((shape, data))
}

fn floating_length(start: f64, end: f64, step: f64) -> Result<usize> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() || step == 0.0 {
        return Err(Error::invalid_argument("arange start/end/step"));
    }

    if (step > 0.0 && end < start) || (step < 0.0 && end > start) {
        return Err(Error::invalid_argument("arange step direction"));
    }

    let span = end - start;
    let count = if span.is_finite() {
        span / step
    } else {
        end / step - start / step
    }
    .ceil();
    count
        .to_usize()
        .ok_or_else(|| Error::out_of_range("arange length"))
}

#[expect(
    clippy::cast_precision_loss,
    reason = "interpolation weights intentionally round sample indices to floating point"
)]
fn interpolate(start: f64, end: f64, index: usize, denominator: usize) -> f64 {
    let weight = index as f64 / denominator as f64;
    if start.is_sign_negative() == end.is_sign_negative() {
        (end - start).mul_add(weight, start)
    } else {
        (1.0 - weight).mul_add(start, weight * end)
    }
}

mod private {
    use crate::error::Result;

    pub trait Range: Sized + Copy {
        fn length(start: Self, end: Self, step: Self) -> Result<usize>;

        fn range_value(start: Self, step: Self, index: usize) -> Result<Self>;
    }

    pub trait Linspace: Sized + Copy {
        fn validate(start: Self, end: Self) -> Result<()>;

        fn interpolate(start: Self, end: Self, index: usize, denominator: usize) -> Self;
    }
}

macro_rules! floating_element {
    ($ty:ty) => {
        impl TensorRangeElement for $ty {}

        impl private::Range for $ty {
            fn length(start: Self, end: Self, step: Self) -> Result<usize> {
                floating_length(f64::from(start), f64::from(end), f64::from(step))
            }

            #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, reason = "floating ranges evaluate indices in f64 and round to the requested f32 or f64 dtype")]
            fn range_value(start: Self, step: Self, index: usize) -> Result<Self> {
                Ok(if index == 0 {
                    start
                } else {
                    (index as f64).mul_add(f64::from(step), f64::from(start)) as Self
                })
            }
        }

        impl TensorLinspaceElement for $ty {}

        impl private::Linspace for $ty {
            fn validate(start: Self, end: Self) -> Result<()> {
                if start.is_finite() && end.is_finite() {
                    Ok(())
                } else {
                    Err(Error::invalid_argument("linspace start/end"))
                }
            }

            #[allow(clippy::cast_possible_truncation, reason = "linspace interpolation rounds to the requested f32 or f64 dtype")]
            fn interpolate(start: Self, end: Self, index: usize, denominator: usize) -> Self {
                interpolate(f64::from(start), f64::from(end), index, denominator) as Self
            }
        }
    };
}

floating_element!(f32);

floating_element!(f64);

impl TensorRangeElement for i64 {}

impl private::Range for i64 {
    fn length(start: Self, end: Self, step: Self) -> Result<usize> {
        if step == 0 || (step > 0 && end < start) || (step < 0 && end > start) {
            return Err(Error::invalid_argument("arange integer bounds/step"));
        }

        let distance = (i128::from(end) - i128::from(start)).abs();
        let step = i128::from(step).abs();
        let count = distance / step + i128::from(distance % step != 0);
        to_usize(count, "arange length")
    }

    fn range_value(start: Self, step: Self, index: usize) -> Result<Self> {
        let value = i128::from(start) + index as i128 * i128::from(step);
        Self::try_from(value).map_err(|_| Error::out_of_range("arange integer value"))
    }
}
