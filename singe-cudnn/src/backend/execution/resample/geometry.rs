use num_rational::BigRational;

use num_traits::{Signed, ToPrimitive};

use crate::{
    error::{Error, Result},
    math::Fraction,
    utility::check_range,
};

/// Validated spatial fractions shared by frontend inference and native resampling.
/// All arrays have the same nonzero length.
/// Padding may be negative, while windows and strides must be positive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResampleGeometry {
    pre_paddings: Vec<Fraction>,
    post_paddings: Vec<Fraction>,
    strides: Vec<Fraction>,
    window_dimensions: Vec<Fraction>,
}

impl ResampleGeometry {
    /// Creates spatial geometry without changing the supplied fractions or their units.
    ///
    /// # Errors
    /// Returns an error for empty geometry, mismatched ranks, nonpositive denominators,
    /// or nonpositive window and stride values.
    pub fn create(
        pre_paddings: impl Into<Vec<Fraction>>,
        post_paddings: impl Into<Vec<Fraction>>,
        strides: impl Into<Vec<Fraction>>,
        window_dimensions: impl Into<Vec<Fraction>>,
    ) -> Result<Self> {
        let pre_paddings = pre_paddings.into();
        let post_paddings = post_paddings.into();
        let strides = strides.into();
        let window_dimensions = window_dimensions.into();
        let spatial_dims = pre_paddings.len();
        if spatial_dims == 0 {
            return Err(Error::InvalidDataShape);
        }
        if post_paddings.len() != spatial_dims {
            return Err(Error::LengthMismatch {
                name: "post_paddings".into(),
                expected: spatial_dims,
                actual: post_paddings.len(),
            });
        }
        if strides.len() != spatial_dims {
            return Err(Error::LengthMismatch {
                name: "strides".into(),
                expected: spatial_dims,
                actual: strides.len(),
            });
        }
        if window_dimensions.len() != spatial_dims {
            return Err(Error::LengthMismatch {
                name: "window_dims".into(),
                expected: spatial_dims,
                actual: window_dimensions.len(),
            });
        }
        for value in &pre_paddings {
            value.validate("resample pre paddings")?;
        }
        for value in &post_paddings {
            value.validate("resample post paddings")?;
        }
        for value in &strides {
            value.validate("resample strides")?;
            check_range!("resample strides", value.numerator() > 0)?;
        }
        for value in &window_dimensions {
            value.validate("resample window dims")?;
            check_range!("resample window dims", value.numerator() > 0)?;
        }

        Ok(Self {
            pre_paddings,
            post_paddings,
            strides,
            window_dimensions,
        })
    }

    /// Returns the nonzero number of spatial axes.
    #[must_use]
    pub fn spatial_dims(&self) -> usize {
        self.window_dimensions.len()
    }

    /// Borrows the exact fractional padding before each spatial axis.
    #[must_use]
    pub fn pre_paddings(&self) -> &[Fraction] {
        &self.pre_paddings
    }

    /// Borrows the exact fractional padding after each spatial axis.
    #[must_use]
    pub fn post_paddings(&self) -> &[Fraction] {
        &self.post_paddings
    }

    /// Borrows the positive fractional stride of each spatial axis.
    #[must_use]
    pub fn strides(&self) -> &[Fraction] {
        &self.strides
    }

    /// Borrows the positive fractional window extent of each spatial axis.
    #[must_use]
    pub fn window_dims(&self) -> &[Fraction] {
        &self.window_dimensions
    }

    /// Infers `[N, C, spatial...]` response dimensions without constructing tensor layouts.
    /// Evaluates padding, windows and strides as exact fractions, including unreduced values.
    /// Batch and channel dimensions are preserved.
    /// Native support selection still determines executable mode and geometry combinations.
    ///
    /// # Errors
    /// Returns an error for mismatched ranks, nonpositive input dimensions,
    /// a negative padded window span, or response dimensions exceeding INT64.
    pub fn infer_forward_dimensions(&self, input: &[i64]) -> Result<Vec<i64>> {
        if input.len() < 3 {
            return Err(Error::LengthMismatch {
                name: "resample rank".into(),
                expected: 3,
                actual: input.len(),
            });
        }
        let spatial_dims = input.len() - 2;
        if self.spatial_dims() != spatial_dims {
            return Err(Error::LengthMismatch {
                name: "resample spatial dims".into(),
                expected: spatial_dims,
                actual: self.spatial_dims(),
            });
        }
        check_range!(
            "resample shape",
            input.iter().all(|&dimension| dimension > 0)
        )?;
        let rational = |value: Fraction| {
            BigRational::new(value.numerator().into(), value.denominator().into())
        };
        let mut output = Vec::with_capacity(input.len());
        output.extend_from_slice(&input[..2]);
        for index in 0..spatial_dims {
            // Valid i64 fractions can need intermediates wider than i128 even for small outputs.
            let span = BigRational::from_integer(input[index + 2].into())
                + rational(self.pre_paddings()[index])
                + rational(self.post_paddings()[index])
                - rational(self.window_dims()[index]);
            check_range!("resample shape", !span.is_negative())?;
            let steps = (span / rational(self.strides()[index])).to_integer();
            let dimension = steps
                .to_i64()
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| Error::out_of_range("resample shape"))?;
            output.push(dimension);
        }
        Ok(output)
    }
}
