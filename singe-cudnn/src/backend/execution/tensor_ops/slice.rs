use std::ops::Range;

use serde::{Deserialize, Serialize};

use crate::{
    backend::{
        attribute::BackendAttributeName,
        descriptor::{BackendDescriptor, BackendDescriptorType},
        execution::tensor_ops::TensorTransformMetadata,
        tensor::{Shape, Tensor},
    },
    error::{Error, Result},
    utility::check_range,
};

/// Half-open slice ranges, optional per-axis steps and diagnostic scope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SliceConfig {
    slices: Vec<Range<i64>>,
    strides: Vec<i64>,
    #[serde(default)]
    name: Option<String>,
}

impl SliceConfig {
    /// Records half-open ranges with unit default steps and no diagnostic scope.
    pub fn new(slices: impl Into<Vec<Range<i64>>>) -> Self {
        Self {
            slices: slices.into(),
            strides: Vec::new(),
            name: None,
        }
    }

    /// Replaces the half-open slice ranges while retaining steps and the name scope.
    #[must_use]
    pub fn with_slices(mut self, slices: impl Into<Vec<Range<i64>>>) -> Self {
        self.slices = slices.into();
        self
    }

    /// Sets per-axis steps in descriptor dimensions.
    /// Omitted entries use one; extra entries must be positive and are otherwise ignored.
    /// An empty array restores unit steps on every axis.
    #[must_use]
    pub fn with_strides(mut self, strides: impl Into<Vec<i64>>) -> Self {
        self.strides = strides.into();
        self
    }

    /// Names inferred results `name::Y` while preserving explicit output names.
    /// An empty scope disables inferred naming.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Borrows requested half-open ranges in descriptor-dimension units.
    #[must_use]
    pub fn slices(&self) -> &[Range<i64>] {
        &self.slices
    }

    /// Returns the supplied steps before filling omitted axes or ignoring extra entries.
    #[must_use]
    pub fn strides(&self) -> &[i64] {
        &self.strides
    }

    /// Returns the nonempty output-name scope, or `None` when unnamed.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref().filter(|name| !name.is_empty())
    }

    fn effective_strides(&self) -> impl Iterator<Item = i64> + '_ {
        (0..self.slices.len()).map(|axis| self.strides.get(axis).copied().unwrap_or(1))
    }

    /// Infers the C++ slice layout in descriptor element units, without native descriptors.
    /// Multiplies source strides by effective steps and retains a source stride for an otherwise overflowing singleton axis.
    /// Explicit output layouts remain independent of this inferred layout.
    ///
    /// # Errors
    /// Returns an error for invalid ranges or steps, rank mismatches, or overflowing non-singleton strides.
    pub fn infer_output_shape(&self, input: &Shape) -> Result<Shape> {
        let dimensions = self.infer_output_dimensions(input)?;
        let strides = input
            .strides()
            .iter()
            .zip(self.effective_strides())
            .zip(&dimensions)
            .map(|((&stride, slice_stride), &dimension)| {
                match stride.checked_mul(slice_stride) {
                    Some(stride) => Ok(stride),
                    // A singleton axis never advances, so its source stride still describes the result.
                    None if dimension == 1 => Ok(stride),
                    None => Err(Error::out_of_range("slice output strides")),
                }
            })
            .collect::<Result<Vec<_>>>()?;

        Shape::with_strides(dimensions, strides)
    }

    /// Infers output dimensions without calculating or constraining destination strides.
    /// Use this for independently declared output storage.
    ///
    /// # Errors
    /// Returns an error for rank mismatches, invalid ranges or nonpositive steps.
    pub fn infer_output_dimensions(&self, input: &Shape) -> Result<Vec<i64>> {
        let input = input.dimensions();
        if input.len() != self.slices.len() {
            return Err(Error::LengthMismatch {
                name: "slice rank".into(),
                expected: input.len(),
                actual: self.slices.len(),
            });
        }
        check_range!(
            "slice stride",
            self.strides.iter().all(|&stride| stride > 0)
        )?;

        self.slices
            .iter()
            .zip(input)
            .zip(self.effective_strides())
            .map(|((range, &input_dimension), stride)| {
                check_range!(
                    "slice range",
                    range.start >= 0 && range.end > range.start && range.end <= input_dimension
                )?;
                Ok(1 + (range.end - range.start - 1) / stride)
            })
            .collect()
    }
}

/// Native slice with independent input and output tensor descriptors.
#[derive(Debug)]
pub struct SliceOperation {
    descriptor: BackendDescriptor,
    metadata: TensorTransformMetadata<SliceConfig>,
}

impl SliceOperation {
    /// Creates native slicing with half-open ranges and effective positive per-axis steps.
    ///
    /// # Errors
    ///
    /// Returns an error for rank mismatches, invalid ranges or nonpositive supplied steps, incompatible output dimensions or encoding, or native descriptor creation, attribute assignment, or finalization failure.
    pub fn create(x: &Tensor, y: &Tensor, config: SliceConfig) -> Result<Self> {
        let output_dimensions = config.infer_output_dimensions(x.shape())?;
        if y.dimensions() != output_dimensions.as_slice() {
            return Err(Error::DescriptorMismatch {
                name: "slice output dimensions".into(),
            });
        }
        if x.data_type() != y.data_type() {
            return Err(Error::DescriptorMismatch {
                name: "slice data type".into(),
            });
        }
        let starts = config
            .slices()
            .iter()
            .map(|range| range.start)
            .collect::<Vec<_>>();
        let limits = config
            .slices()
            .iter()
            .map(|range| range.end)
            .collect::<Vec<_>>();
        let strides = config.effective_strides().collect::<Vec<_>>();
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::OperationSlice)?;
        descriptor
            .set_attribute_descriptor(BackendAttributeName::OperationSliceXDesc, x.descriptor())?;
        descriptor
            .set_attribute_descriptor(BackendAttributeName::OperationSliceYDesc, y.descriptor())?;
        descriptor
            .set_attribute_i64_slice(BackendAttributeName::OperationSliceStartIndices, &starts)?;
        descriptor
            .set_attribute_i64_slice(BackendAttributeName::OperationSliceLimitIndices, &limits)?;
        descriptor
            .set_attribute_i64_slice(BackendAttributeName::OperationSliceStrides, &strides)?;
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

    /// Borrows retained slice ranges, supplied steps, and diagnostic scope.
    #[must_use]
    pub fn config(&self) -> &SliceConfig {
        &self.metadata.config
    }

    /// Native tensor identities and complete requested slice configuration.
    #[must_use]
    pub fn metadata(&self) -> &TensorTransformMetadata<SliceConfig> {
        &self.metadata
    }
}
