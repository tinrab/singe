use std::result::Result as StdResult;

use serde::{Deserialize, Serialize};

use crate::{
    backend::{
        attribute::BackendAttributeName,
        descriptor::{BackendDescriptor, BackendDescriptorType},
        execution::normalization::NormalizationSavedStatistics,
        tensor::{BackendTensorUid, Tensor, TensorId},
    },
    data_type::DataType,
    error::{Error, Result},
};

/// Saved forward values and output gradient for the `BatchNorm` weight-gradient helper.
/// Carries frontend identities, native tensor references or retained native UIDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchNormalizationBackwardWeightsInputs<T = TensorId> {
    /// Input activation tensor.
    pub input: T,
    /// Saved mean and inverse-standard-deviation tensors.
    pub statistics: NormalizationSavedStatistics<T>,
    /// Gradient tensor for the forward output.
    pub output_gradient: T,
    /// Affine scale tensor.
    pub scale: T,
}

impl<T> BatchNormalizationBackwardWeightsInputs<T> {
    /// Borrows the input, saved statistics, output gradient and scale.
    pub fn as_ref(&self) -> BatchNormalizationBackwardWeightsInputs<&T> {
        BatchNormalizationBackwardWeightsInputs {
            input: &self.input,
            statistics: self.statistics.as_ref(),
            output_gradient: &self.output_gradient,
            scale: &self.scale,
        }
    }

    /// Maps every operand while preserving saved-statistic and gradient roles.
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> BatchNormalizationBackwardWeightsInputs<U> {
        BatchNormalizationBackwardWeightsInputs {
            input: map(self.input),
            statistics: self.statistics.map(&mut map),
            output_gradient: map(self.output_gradient),
            scale: map(self.scale),
        }
    }

    /// Resolves every operand through a fallible mapping.
    ///
    /// # Errors
    ///
    /// Returns the first error from the supplied tensor mapping; later operands are not mapped.
    pub fn try_map<U, E>(
        self,
        mut map: impl FnMut(T) -> StdResult<U, E>,
    ) -> StdResult<BatchNormalizationBackwardWeightsInputs<U>, E> {
        Ok(BatchNormalizationBackwardWeightsInputs {
            input: map(self.input)?,
            statistics: self.statistics.try_map(&mut map)?,
            output_gradient: map(self.output_gradient)?,
            scale: map(self.scale)?,
        })
    }
}

/// Weight gradients and equivalent coefficients produced by the `BatchNorm` helper.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchNormalizationBackwardWeightsOutputs<T = TensorId> {
    /// Affine scale-gradient output tensor.
    pub scale_gradient: T,
    /// Affine bias-gradient output tensor.
    pub bias_gradient: T,
    /// Equivalent bias tensor for the subsequent normalization step.
    pub equivalent_bias: T,
    /// Equivalent coefficient multiplying the output gradient in the subsequent backward step.
    pub equivalent_output_gradient_scale: T,
    /// Equivalent coefficient multiplying the input activation in the subsequent backward step.
    pub equivalent_input_scale: T,
}

impl<T> BatchNormalizationBackwardWeightsOutputs<T> {
    /// Borrows affine gradients and equivalent coefficients without copying tensor owners.
    pub fn as_ref(&self) -> BatchNormalizationBackwardWeightsOutputs<&T> {
        BatchNormalizationBackwardWeightsOutputs {
            scale_gradient: &self.scale_gradient,
            bias_gradient: &self.bias_gradient,
            equivalent_bias: &self.equivalent_bias,
            equivalent_output_gradient_scale: &self.equivalent_output_gradient_scale,
            equivalent_input_scale: &self.equivalent_input_scale,
        }
    }

    /// Maps gradients and equivalent coefficients without changing their roles.
    pub fn map<U>(
        self,
        mut map: impl FnMut(T) -> U,
    ) -> BatchNormalizationBackwardWeightsOutputs<U> {
        BatchNormalizationBackwardWeightsOutputs {
            scale_gradient: map(self.scale_gradient),
            bias_gradient: map(self.bias_gradient),
            equivalent_bias: map(self.equivalent_bias),
            equivalent_output_gradient_scale: map(self.equivalent_output_gradient_scale),
            equivalent_input_scale: map(self.equivalent_input_scale),
        }
    }

    /// Resolves all results through a fallible mapping.
    ///
    /// # Errors
    ///
    /// Returns the first error from the supplied tensor mapping; later operands are not mapped.
    pub fn try_map<U, E>(
        self,
        mut map: impl FnMut(T) -> StdResult<U, E>,
    ) -> StdResult<BatchNormalizationBackwardWeightsOutputs<U>, E> {
        Ok(BatchNormalizationBackwardWeightsOutputs {
            scale_gradient: map(self.scale_gradient)?,
            bias_gradient: map(self.bias_gradient)?,
            equivalent_bias: map(self.equivalent_bias)?,
            equivalent_output_gradient_scale: map(self.equivalent_output_gradient_scale)?,
            equivalent_input_scale: map(self.equivalent_input_scale)?,
        })
    }
}

/// Retained math precision and native identities, independent of original tensor wrappers.
/// Optional scale and output attributes preserve their absence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchNormalizationBackwardWeightsOperationMetadata {
    /// Math precision selected independently of tensor storage encodings.
    pub compute_type: DataType,
    /// Retained native input identities, preserving independently omitted attributes.
    pub inputs: BatchNormalizationBackwardWeightsInputs<Option<BackendTensorUid>>,
    /// Retained native output identities, preserving independently omitted attributes.
    pub outputs: BatchNormalizationBackwardWeightsOutputs<Option<BackendTensorUid>>,
}

/// Finalized native batch-normalization weight-gradient operation with retained operand and coefficient identities.
#[derive(Debug)]
pub struct BatchNormalizationBackwardWeightsOperation {
    descriptor: BackendDescriptor,
    metadata: BatchNormalizationBackwardWeightsOperationMetadata,
}

impl BatchNormalizationBackwardWeightsOperation {
    /// Creates weight gradients and equivalent coefficients using C++'s native operand rules.
    /// Each group accepts complete or optional tensor references.
    /// Input, output gradient, mean and inverse standard deviation are required.
    /// Scale and each result may be omitted independently; cuDNN determines native support.
    ///
    /// # Errors
    /// Returns an error for a missing required operand or native descriptor failure.
    pub fn create<'a>(
        compute_type: DataType,
        inputs: BatchNormalizationBackwardWeightsInputs<impl Into<Option<&'a Tensor>>>,
        outputs: BatchNormalizationBackwardWeightsOutputs<impl Into<Option<&'a Tensor>>>,
    ) -> Result<Self> {
        let inputs: BatchNormalizationBackwardWeightsInputs<Option<&Tensor>> =
            inputs.map(Into::into);
        let outputs: BatchNormalizationBackwardWeightsOutputs<Option<&Tensor>> =
            outputs.map(Into::into);
        for (attribute, tensor) in [
            (
                BackendAttributeName::OperationBnBwdWeightsXDesc,
                inputs.input,
            ),
            (
                BackendAttributeName::OperationBnBwdWeightsDyDesc,
                inputs.output_gradient,
            ),
            (
                BackendAttributeName::OperationBnBwdWeightsMeanDesc,
                inputs.statistics.mean,
            ),
            (
                BackendAttributeName::OperationBnBwdWeightsInvStdDesc,
                inputs.statistics.inv_variance,
            ),
        ] {
            if tensor.is_none() {
                return Err(Error::DescriptorAttributeNotFound(attribute));
            }
        }
        let mut descriptor =
            BackendDescriptor::create(BackendDescriptorType::OperationBnBwdWeights)?;
        descriptor.set_attribute_enum(
            BackendAttributeName::OperationBnBwdWeightsMathPrec,
            compute_type,
        )?;
        for (attribute, tensor) in [
            (
                BackendAttributeName::OperationBnBwdWeightsMeanDesc,
                inputs.statistics.mean,
            ),
            (
                BackendAttributeName::OperationBnBwdWeightsInvStdDesc,
                inputs.statistics.inv_variance,
            ),
            (
                BackendAttributeName::OperationBnBwdWeightsBnScaleDesc,
                inputs.scale,
            ),
            (
                BackendAttributeName::OperationBnBwdWeightsXDesc,
                inputs.input,
            ),
            (
                BackendAttributeName::OperationBnBwdWeightsDyDesc,
                inputs.output_gradient,
            ),
            (
                BackendAttributeName::OperationBnBwdWeightsDbnScaleDesc,
                outputs.scale_gradient,
            ),
            (
                BackendAttributeName::OperationBnBwdWeightsDbnBiasDesc,
                outputs.bias_gradient,
            ),
            (
                BackendAttributeName::OperationBnBwdWeightsEqDyScaleDesc,
                outputs.equivalent_output_gradient_scale,
            ),
            (
                BackendAttributeName::OperationBnBwdWeightsEqXScaleDesc,
                outputs.equivalent_input_scale,
            ),
            (
                BackendAttributeName::OperationBnBwdWeightsEqBias,
                outputs.equivalent_bias,
            ),
        ] {
            if let Some(tensor) = tensor {
                descriptor.set_attribute_descriptor(attribute, tensor.descriptor())?;
            }
        }
        descriptor.finalize()?;

        Ok(Self {
            descriptor,
            metadata: BatchNormalizationBackwardWeightsOperationMetadata {
                compute_type,
                inputs: inputs.map(|tensor| tensor.map(Tensor::id)),
                outputs: outputs.map(|tensor| tensor.map(Tensor::id)),
            },
        })
    }

    /// Retained precision, saved-state identities and output roles without queries or allocation.
    #[must_use]
    pub fn metadata(&self) -> &BatchNormalizationBackwardWeightsOperationMetadata {
        &self.metadata
    }

    /// Borrows the finalized native statistics-helper descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}
