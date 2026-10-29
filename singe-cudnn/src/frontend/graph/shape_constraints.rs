use serde::{Deserialize, Serialize};

use crate::{
    backend::tensor::{Shape, TensorSpec},
    error::{Error, Result},
    frontend::error::Error as FrontendError,
};

/// Inclusive runtime shape bounds accepted for one dynamic-shape tensor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeShapeConstraints {
    /// Inclusive lower extent for each tensor dimension, in descriptor element units.
    pub min_dimensions: Vec<i64>,
    /// Inclusive upper extent for each tensor dimension, with the same rank as the lower bounds.
    pub max_dimensions: Vec<i64>,
}

impl RuntimeShapeConstraints {
    /// Creates inclusive dimension bounds without attaching them to a tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if bound ranks differ, either dimension list cannot form a checked contiguous shape, or a lower bound exceeds its corresponding upper bound.
    pub fn create(
        min_dimensions: impl Into<Vec<i64>>,
        max_dimensions: impl Into<Vec<i64>>,
    ) -> Result<Self> {
        let min_dimensions = min_dimensions.into();
        let max_dimensions = max_dimensions.into();
        validate_runtime_shape_constraint_dimensions(&min_dimensions, &max_dimensions)?;
        Ok(Self {
            min_dimensions,
            max_dimensions,
        })
    }

    /// Returns whether the dimensions have the bounds' rank and every extent lies within its inclusive interval.
    #[must_use]
    pub fn contains(&self, dimensions: &[i64]) -> bool {
        dimensions.len() == self.min_dimensions.len()
            && dimensions.len() == self.max_dimensions.len()
            && self
                .min_dimensions
                .iter()
                .zip(&self.max_dimensions)
                .zip(dimensions)
                .all(|((min_dimension, max_dimension), dimension)| {
                    *min_dimension <= *dimension && *dimension <= *max_dimension
                })
    }

    /// Combines constraints on two identities that will share one runtime layout.
    pub(crate) fn intersection(&self, other: &Self) -> Result<Self> {
        validate_runtime_shape_constraint_dimensions(&self.min_dimensions, &self.max_dimensions)?;
        validate_runtime_shape_constraint_dimensions(&other.min_dimensions, &other.max_dimensions)?;
        if self.min_dimensions.len() != other.min_dimensions.len() {
            return Err(Error::LengthMismatch {
                name: "dynamic shape constraint intersection rank".into(),
                expected: self.min_dimensions.len(),
                actual: other.min_dimensions.len(),
            });
        }
        Self::create(
            self.min_dimensions
                .iter()
                .zip(&other.min_dimensions)
                .map(|(&left, &right)| left.max(right))
                .collect::<Vec<_>>(),
            self.max_dimensions
                .iter()
                .zip(&other.max_dimensions)
                .map(|(&left, &right)| left.min(right))
                .collect::<Vec<_>>(),
        )
    }

    pub(crate) fn validate_for(&self, tensor: &TensorSpec) -> Result<()> {
        validate_runtime_shape_constraint_dimensions(&self.min_dimensions, &self.max_dimensions)?;
        if self.min_dimensions.len() != tensor.shape.rank() {
            return Err(Error::LengthMismatch {
                name: "runtime_shape_constraints_rank".into(),
                expected: tensor.shape.rank(),
                actual: self.min_dimensions.len(),
            });
        }
        if tensor.is_by_value() {
            tensor.validate_scalar()?;
            if self
                .min_dimensions
                .iter()
                .chain(&self.max_dimensions)
                .any(|&dimension| dimension != 1)
            {
                return Err(Error::DescriptorMismatch {
                    name: "scalar dynamic shape bounds".into(),
                });
            }
        }
        Ok(())
    }
}

pub fn validate_runtime_shape_constraint_dimensions(
    min_dimensions: &[i64],
    max_dimensions: &[i64],
) -> Result<()> {
    if min_dimensions.len() != max_dimensions.len() {
        return Err(Error::LengthMismatch {
            name: "runtime_shape_constraints_max_rank".into(),
            expected: min_dimensions.len(),
            actual: max_dimensions.len(),
        });
    }
    Shape::contiguous(min_dimensions)?;
    Shape::contiguous(max_dimensions)?;
    if min_dimensions
        .iter()
        .zip(max_dimensions)
        .any(|(min_dimension, max_dimension)| min_dimension > max_dimension)
    {
        return Err(FrontendError::RuntimeShapeBoundsInvalid {
            min_dimensions: min_dimensions.to_vec(),
            max_dimensions: max_dimensions.to_vec(),
        }
        .into());
    }
    Ok(())
}
