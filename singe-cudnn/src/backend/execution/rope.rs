use std::result::Result as StdResult;

use serde::{Deserialize, Serialize};

use crate::{
    backend::{
        attribute::BackendAttributeName,
        descriptor::{BackendDescriptor, BackendDescriptorType},
        tensor::{BackendTensorUid, Tensor},
    },
    error::{Error, Result},
};

/// Activation and raw-angle frequency operands for a forward rotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RopeInputs<T> {
    /// Activation tensor to rotate.
    pub input: T,
    /// FP32 raw-angle frequency tensor; values are not precomputed sines or cosines.
    pub frequencies: T,
}

impl<T> RopeInputs<T> {
    /// Records the activation and raw-angle frequency operands.
    pub fn new(input: T, frequencies: T) -> Self {
        Self { input, frequencies }
    }

    /// Borrows activation and frequency owners for native construction.
    pub fn as_ref(&self) -> RopeInputs<&T> {
        RopeInputs {
            input: &self.input,
            frequencies: &self.frequencies,
        }
    }

    /// Maps both tensor operands while preserving activation/gradient and frequency roles.
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> RopeInputs<U> {
        RopeInputs {
            input: map(self.input),
            frequencies: map(self.frequencies),
        }
    }

    /// Maps both operands while preserving their roles.
    ///
    /// # Errors
    ///
    /// Returns the first tensor-mapping error, processing the activation or gradient before frequencies.
    pub fn try_map<U, E>(
        self,
        mut map: impl FnMut(T) -> StdResult<U, E>,
    ) -> StdResult<RopeInputs<U>, E> {
        Ok(RopeInputs {
            input: map(self.input)?,
            frequencies: map(self.frequencies)?,
        })
    }
}

/// Output-gradient and raw-angle frequency operands for the transpose rotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RopeBackwardInputs<T> {
    /// Gradient with respect to the forward rotated output.
    pub output_gradient: T,
    /// FP32 raw-angle frequency tensor used by the corresponding forward rotation.
    pub frequencies: T,
}

impl<T> RopeBackwardInputs<T> {
    /// Records the output gradient and raw-angle frequency operands.
    pub fn new(output_gradient: T, frequencies: T) -> Self {
        Self {
            output_gradient,
            frequencies,
        }
    }

    /// Borrows gradient and frequency owners for native construction.
    pub fn as_ref(&self) -> RopeBackwardInputs<&T> {
        RopeBackwardInputs {
            output_gradient: &self.output_gradient,
            frequencies: &self.frequencies,
        }
    }

    /// Maps both tensor operands while preserving activation/gradient and frequency roles.
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> RopeBackwardInputs<U> {
        RopeBackwardInputs {
            output_gradient: map(self.output_gradient),
            frequencies: map(self.frequencies),
        }
    }

    /// Maps both operands while preserving their roles.
    ///
    /// # Errors
    ///
    /// Returns the first tensor-mapping error, processing the activation or gradient before frequencies.
    pub fn try_map<U, E>(
        self,
        mut map: impl FnMut(T) -> StdResult<U, E>,
    ) -> StdResult<RopeBackwardInputs<U>, E> {
        Ok(RopeBackwardInputs {
            output_gradient: map(self.output_gradient)?,
            frequencies: map(self.frequencies)?,
        })
    }
}

/// Native rotary position embedding parameters shared by forward and backward.
///
/// The trailing rotated segment uses non-interleaved halves. Frequencies are raw FP32
/// angles with dimensions `[sequence, 1, 1, effective_rotation_dimension]`, where zero
/// in the configuration selects the entire head dimension.
/// Native engine support determines the accepted activation encodings and layouts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RopeConfig {
    /// Optional operation scope used to name inferred frontend outputs.
    #[serde(default)]
    pub name: Option<String>,
    /// Size of the trailing rotated segment. Zero rotates the entire head dimension.
    /// The native operation requires an even value no larger than the head dimension.
    pub rotation_dimension: i64,
    /// Scales every output component, including the leading unrotated segment.
    /// Use the same value for the corresponding forward and backward operations.
    #[serde(with = "crate::scalar::float::f32_bits")]
    pub output_scale: f32,
}

impl Default for RopeConfig {
    fn default() -> Self {
        Self {
            name: None,
            rotation_dimension: 0,
            output_scale: 1.0,
        }
    }
}

impl RopeConfig {
    /// Names inferred results without changing explicit output tensor names.
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

    pub(crate) fn validate(&self) -> Result<()> {
        if self.rotation_dimension < 0 {
            return Err(Error::out_of_range("rope rotation dimension"));
        }
        Ok(())
    }
}

/// Requested rotation settings and actual native tensor identities for a forward operation.
/// A zero rotation dimension retains the native full-head convention.
#[derive(Debug, Clone, PartialEq)]
pub struct RopeForwardOperationMetadata {
    /// Retained native activation and frequency identities.
    pub inputs: RopeInputs<BackendTensorUid>,
    /// Native identity of the rotated output.
    pub output: BackendTensorUid,
    /// Retained rotation dimension, output scale, and optional diagnostic name.
    pub config: RopeConfig,
}

/// Finalized native rotary position embedding with retained activation and frequency identities.
#[derive(Debug)]
pub struct RopeForwardOperation {
    descriptor: BackendDescriptor,
    metadata: RopeForwardOperationMetadata,
}

impl RopeForwardOperation {
    /// Creates a native forward rotation using raw-angle frequencies.
    ///
    /// # Errors
    ///
    /// Returns an error for a negative rotation dimension or if native descriptor creation, attribute assignment, or finalization fails. cuDNN validates activation layouts and encodings, frequency geometry, and the effective rotation dimension.
    pub fn create(
        inputs: RopeInputs<&Tensor>,
        output: &Tensor,
        config: &RopeConfig,
    ) -> Result<Self> {
        config.validate()?;
        let mut descriptor =
            BackendDescriptor::create(BackendDescriptorType::OperationRopeForward)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationRopeForwardInputDesc,
            inputs.input.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationRopeForwardFrequenciesDesc,
            inputs.frequencies.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationRopeForwardOutputDesc,
            output.descriptor(),
        )?;
        descriptor.set_attribute_f32(
            BackendAttributeName::OperationRopeForwardOutputScale,
            config.output_scale,
        )?;
        descriptor.set_attribute_i64(
            BackendAttributeName::OperationRopeForwardRotationDimension,
            config.rotation_dimension,
        )?;
        descriptor.finalize()?;
        Ok(Self {
            descriptor,
            metadata: RopeForwardOperationMetadata {
                inputs: inputs.map(Tensor::id),
                output: output.id(),
                config: config.clone(),
            },
        })
    }

    /// Retained native operands, diagnostic scope, rotation dimension and output scale.
    /// Does not query cuDNN, allocate or borrow the original tensor and config wrappers.
    #[must_use]
    pub fn metadata(&self) -> &RopeForwardOperationMetadata {
        &self.metadata
    }

    /// Borrows the finalized native rotary position embedding descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

/// Requested rotation settings and native gradient/frequency identities for a transpose rotation.
#[derive(Debug, Clone, PartialEq)]
pub struct RopeBackwardOperationMetadata {
    /// Retained native output-gradient and frequency identities.
    pub inputs: RopeBackwardInputs<BackendTensorUid>,
    /// Native identity of the input-gradient output.
    pub input_gradient: BackendTensorUid,
    /// Retained rotation dimension, output scale, and optional diagnostic name.
    pub config: RopeConfig,
}

/// Finalized native transpose rotation with retained gradient and frequency identities.
#[derive(Debug)]
pub struct RopeBackwardOperation {
    descriptor: BackendDescriptor,
    metadata: RopeBackwardOperationMetadata,
}

impl RopeBackwardOperation {
    /// Creates a native transpose rotation for input gradients using raw-angle frequencies.
    ///
    /// # Errors
    ///
    /// Returns an error for a negative rotation dimension or if native descriptor creation, attribute assignment, or finalization fails. cuDNN validates activation layouts and encodings, frequency geometry, and the effective rotation dimension.
    pub fn create(
        inputs: RopeBackwardInputs<&Tensor>,
        input_gradient: &Tensor,
        config: &RopeConfig,
    ) -> Result<Self> {
        config.validate()?;
        let mut descriptor =
            BackendDescriptor::create(BackendDescriptorType::OperationRopeBackward)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationRopeBackwardInputDesc,
            inputs.output_gradient.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationRopeBackwardFrequenciesDesc,
            inputs.frequencies.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationRopeBackwardOutputDesc,
            input_gradient.descriptor(),
        )?;
        descriptor.set_attribute_f32(
            BackendAttributeName::OperationRopeBackwardOutputScale,
            config.output_scale,
        )?;
        descriptor.set_attribute_i64(
            BackendAttributeName::OperationRopeBackwardRotationDimension,
            config.rotation_dimension,
        )?;
        descriptor.finalize()?;
        Ok(Self {
            descriptor,
            metadata: RopeBackwardOperationMetadata {
                inputs: inputs.map(Tensor::id),
                input_gradient: input_gradient.id(),
                config: config.clone(),
            },
        })
    }

    /// Retained native gradient operands and the requested rotation settings without native queries or allocation.
    #[must_use]
    pub fn metadata(&self) -> &RopeBackwardOperationMetadata {
        &self.metadata
    }

    /// Borrows the finalized native rotary position embedding descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}
