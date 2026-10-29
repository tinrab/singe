use serde::{Deserialize, Serialize};

use crate::{
    backend::{
        attribute::BackendAttributeName,
        descriptor::{BackendDescriptor, BackendDescriptorType},
        execution::tensor_ops::TensorTransformMetadata,
        tensor::{Shape, Tensor},
    },
    error::{Error, Result},
    utility::{check_range, to_usize},
};

/// Axis permutation and diagnostic scope shared by frontend and native transpose.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransposeConfig {
    permutation: Vec<i64>,
    #[serde(default)]
    name: Option<String>,
}

impl TransposeConfig {
    /// Records an axis permutation without validating it or selecting a diagnostic scope.
    pub fn new(permutation: impl Into<Vec<i64>>) -> Self {
        Self {
            permutation: permutation.into(),
            name: None,
        }
    }

    /// Replaces the requested zero-based axis permutation while retaining the name scope.
    #[must_use]
    pub fn with_permutation(mut self, permutation: impl Into<Vec<i64>>) -> Self {
        self.permutation = permutation.into();
        self
    }

    /// Names inferred results `name::Y` while preserving explicit output names.
    /// An empty scope disables inferred naming.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Borrows the requested zero-based source-axis permutation.
    #[must_use]
    pub fn permutation(&self) -> &[i64] {
        &self.permutation
    }

    /// Returns the nonempty output-name scope, or `None` when unnamed.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref().filter(|name| !name.is_empty())
    }

    /// Infers the C++ transpose layout by permuting dimensions and strides together.
    /// Uses descriptor element units and creates no native descriptors.
    /// Explicit output layouts remain independent of this inferred layout.
    ///
    /// # Errors
    /// Returns an error for an empty, repeated, out-of-range or rank-mismatched permutation.
    ///
    /// # Panics
    ///
    /// Panics if an internal permutation or checked-shape invariant is violated during stride lookup. Malformed permutations return errors before this lookup.
    pub fn infer_output_shape(&self, input: &Shape) -> Result<Shape> {
        let dimensions = self.infer_output_dimensions(input)?;
        let strides = self
            .permutation()
            .iter()
            .map(|&axis| {
                input.strides()[usize::try_from(axis)
                    .expect("transpose axes are validated against the input rank")]
            })
            .collect::<Vec<_>>();
        Shape::with_strides(dimensions, strides)
    }

    /// Infers output dimensions without calculating or constraining destination strides.
    /// Use this for independently declared output storage.
    ///
    /// # Errors
    /// Returns an error for an empty, repeated, out-of-range or rank-mismatched permutation.
    pub fn infer_output_dimensions(&self, input: &Shape) -> Result<Vec<i64>> {
        let input = input.dimensions();
        if input.len() != self.permutation.len() {
            return Err(Error::LengthMismatch {
                name: "transpose permutation rank".into(),
                expected: input.len(),
                actual: self.permutation.len(),
            });
        }
        if self.permutation.is_empty() {
            return Err(Error::EmptyList {
                name: "transpose permutation".into(),
            });
        }

        let mut seen = vec![false; input.len()];
        self.permutation
            .iter()
            .map(|&source| {
                let source = to_usize(source, "transpose permutation")?;
                check_range!(
                    "transpose permutation",
                    source < input.len() && !seen[source]
                )?;
                seen[source] = true;
                Ok(input[source])
            })
            .collect()
    }
}

/// Native transpose. Output strides may describe independent storage.
#[derive(Debug)]
pub struct TransposeOperation {
    descriptor: BackendDescriptor,
    metadata: TensorTransformMetadata<TransposeConfig>,
}

impl TransposeOperation {
    /// Creates a native axis permutation with independently declared output strides.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty or invalid permutation, mismatched ranks, repeated or out-of-range axes, incompatible output dimensions or encoding, or native descriptor creation, attribute assignment, or finalization failure.
    pub fn create(x: &Tensor, y: &Tensor, config: TransposeConfig) -> Result<Self> {
        let output_dimensions = config.infer_output_dimensions(x.shape())?;
        if y.dimensions() != output_dimensions.as_slice() {
            return Err(Error::DescriptorMismatch {
                name: "transpose output dimensions".into(),
            });
        }
        if x.data_type() != y.data_type() {
            return Err(Error::DescriptorMismatch {
                name: "transpose data type".into(),
            });
        }
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::OperationTranspose)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationTransposeXDesc,
            x.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationTransposeYDesc,
            y.descriptor(),
        )?;
        descriptor.set_attribute_i64_slice(
            BackendAttributeName::OperationTransposePermutation,
            config.permutation(),
        )?;
        descriptor.finalize()?;
        Ok(Self {
            descriptor,
            metadata: TensorTransformMetadata {
                input: x.id(),
                output: y.id(),
                config,
            },
        })
    }

    /// Borrows the finalized native operation descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }

    /// Borrows the retained axis permutation and diagnostic scope.
    #[must_use]
    pub fn config(&self) -> &TransposeConfig {
        &self.metadata.config
    }

    /// Native tensor identities with the retained permutation and diagnostic scope.
    #[must_use]
    pub fn metadata(&self) -> &TensorTransformMetadata<TransposeConfig> {
        &self.metadata
    }
}
