use std::iter;

use serde::{Deserialize, Serialize};

use crate::{
    backend::{
        attribute::BackendAttributeName,
        descriptor::{BackendDescriptor, BackendDescriptorType},
        tensor::{BackendTensorUid, Shape, Tensor},
    },
    error::{Error, Result},
    utility::to_usize,
};

/// Whether a concatenation input already occupies its final output region.
///
/// `Input(index)` requires that input's address and strides to describe its
/// final region inside the output allocation, including the offset of preceding
/// inputs. This does not relocate data or change bindings automatically.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ConcatInPlaceMode {
    /// Declares no in-place input.
    #[default]
    None,
    /// Declares the zero-based input index already occupying its final region inside the output allocation.
    Input(i64),
}

impl ConcatInPlaceMode {
    /// Returns the declared in-place input index, or `None` when no input is declared in place.
    #[must_use]
    pub fn index(self) -> Option<i64> {
        match self {
            Self::None => None,
            Self::Input(index) => Some(index),
        }
    }
}

/// Concatenation geometry, optional input/output aliasing and diagnostic scope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConcatConfig {
    #[serde(default)]
    name: Option<String>,
    /// Concatenation axis in descriptor dimensions.
    pub axis: i64,
    /// Optional input/output aliasing declaration; does not relocate data or change bindings.
    pub in_place: ConcatInPlaceMode,
}

impl ConcatConfig {
    /// Records the concatenation axis with no in-place input or diagnostic scope.
    #[must_use]
    pub fn new(axis: i64) -> Self {
        Self {
            name: None,
            axis,
            in_place: ConcatInPlaceMode::None,
        }
    }

    /// Names inferred results without replacing explicit output tensor names.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Returns the nonempty output-name scope, or `None` when unnamed.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref().filter(|name| !name.is_empty())
    }

    /// Infers packed output strides while preserving the first input's stride order, matching C++.
    /// Uses descriptor element units and creates no graph, native descriptors or device allocations.
    /// Vectorization and in-place binding relationships remain separate tensor metadata.
    ///
    /// # Errors
    /// Returns an error for empty inputs, invalid axes, incompatible dimensions or overflowing dimensions/strides.
    pub fn infer_output_shape<'a>(
        &self,
        inputs: impl IntoIterator<Item = &'a Shape>,
    ) -> Result<Shape> {
        let mut inputs = inputs.into_iter();
        let first = inputs.next().ok_or(Error::EmptyList {
            name: "concat inputs".into(),
        })?;
        let dimensions = self.infer_output_dimensions(iter::once(first).chain(inputs))?;
        Shape::packed(dimensions, first.stride_order())
    }

    /// Infers output dimensions without calculating or constraining destination strides.
    /// Accepts checked input shapes and sums the concatenation axis in checked arithmetic.
    ///
    /// # Errors
    /// Returns an error for empty inputs, an invalid axis, incompatible dimensions or an overflowing axis sum.
    pub fn infer_output_dimensions<'a>(
        &self,
        inputs: impl IntoIterator<Item = &'a Shape>,
    ) -> Result<Vec<i64>> {
        let mut inputs = inputs.into_iter();
        let first = inputs.next().ok_or(Error::EmptyList {
            name: "concat inputs".into(),
        })?;
        let axis = to_usize(self.axis, "axis")?;
        if axis >= first.rank() {
            return Err(Error::out_of_range("axis"));
        }
        let mut dimensions = first.dimensions().to_vec();
        let mut axis_sum = dimensions[axis];
        for input in inputs {
            if input.rank() != first.rank() {
                return Err(Error::LengthMismatch {
                    name: "concat ranks".into(),
                    expected: first.rank(),
                    actual: input.rank(),
                });
            }
            for (index, (&expected, &actual)) in first
                .dimensions()
                .iter()
                .zip(input.dimensions())
                .enumerate()
            {
                if index != axis && expected != actual {
                    return Err(Error::ShapeMismatch {
                        name: "concat shape".into(),
                        expected: first.dimensions().to_vec(),
                        actual: input.dimensions().to_vec(),
                    });
                }
            }
            axis_sum = axis_sum
                .checked_add(input.dimensions()[axis])
                .ok_or_else(|| Error::out_of_range("concat axis dimension"))?;
        }
        dimensions[axis] = axis_sum;
        Ok(dimensions)
    }

    pub(crate) fn validate_in_place_index(&self, input_count: usize) -> Result<()> {
        if let Some(index) = self.in_place.index()
            && to_usize(index, "inplace_index")? >= input_count
        {
            return Err(Error::out_of_range("inplace_index"));
        }
        Ok(())
    }

    /// Replaces the in-place input declaration while retaining axis and name scope.
    #[must_use]
    pub fn with_in_place(mut self, in_place: ConcatInPlaceMode) -> Self {
        self.in_place = in_place;
        self
    }
}

/// Ordered native input identities, output identity and accepted concatenation configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConcatOperationMetadata {
    /// Ordered native identities of concatenation inputs.
    pub inputs: Vec<BackendTensorUid>,
    /// Native identity of the concatenation output.
    pub output: BackendTensorUid,
    /// Retained concatenation axis, in-place declaration, and diagnostic scope.
    pub config: ConcatConfig,
}

/// Finalized native concatenation operation with retained operand identities and aliasing declaration.
#[derive(Debug)]
pub struct ConcatOperation {
    descriptor: BackendDescriptor,
    metadata: ConcatOperationMetadata,
}

impl ConcatOperation {
    /// Creates concatenation from ordered input descriptors and an independently declared output.
    ///
    /// # Errors
    ///
    /// Returns an error for empty inputs, invalid axes or in-place indices, incompatible ranks or dimensions, an overflowing concatenation extent, a mismatched output shape, or native descriptor creation, attribute assignment, or finalization failure.
    pub fn create(inputs: &[&Tensor], output: &Tensor, config: ConcatConfig) -> Result<Self> {
        let dimensions =
            config.infer_output_dimensions(inputs.iter().map(|input| input.shape()))?;
        if output.dimensions() != dimensions.as_slice() {
            return Err(Error::DescriptorMismatch {
                name: "concat output shape".into(),
            });
        }
        config.validate_in_place_index(inputs.len())?;

        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::OperationConcat)?;
        let input_descriptors = inputs
            .iter()
            .map(|tensor| tensor.descriptor())
            .collect::<Vec<_>>();
        descriptor.set_attribute_i64(BackendAttributeName::OperationConcatAxis, config.axis)?;
        descriptor.set_attribute_descriptor_slice(
            BackendAttributeName::OperationConcatInputDescs,
            &input_descriptors,
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationConcatOutputDesc,
            output.descriptor(),
        )?;
        if let Some(inplace_index) = config.in_place.index() {
            descriptor.set_attribute_i64(
                BackendAttributeName::OperationConcatInplaceIndex,
                inplace_index,
            )?;
        }
        descriptor.finalize()?;

        Ok(Self {
            descriptor,
            metadata: ConcatOperationMetadata {
                inputs: inputs.iter().map(|input| input.id()).collect(),
                output: output.id(),
                config,
            },
        })
    }

    /// Borrows the finalized native operation descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }

    /// Borrows the retained concatenation axis, in-place declaration, and diagnostic scope.
    #[must_use]
    pub fn config(&self) -> &ConcatConfig {
        &self.metadata.config
    }

    /// Ordered native operands, output and complete configuration without native queries or allocation.
    #[must_use]
    pub fn metadata(&self) -> &ConcatOperationMetadata {
        &self.metadata
    }
}
