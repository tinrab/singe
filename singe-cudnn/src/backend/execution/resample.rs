mod geometry;

use std::result::Result as StdResult;

use num_enum::{IntoPrimitive, TryFromPrimitive};
use serde::{Deserialize, Serialize};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cudnn_sys as sys;

pub use crate::{backend::execution::resample::geometry::ResampleGeometry, math::Fraction};
use crate::{
    backend::{
        attribute::BackendAttributeName,
        descriptor::{BackendDescriptor, BackendDescriptorType},
        tensor::{BackendTensorUid, Tensor, TensorId},
    },
    data_type::DataType,
    error::Result,
    math::NanPropagation,
    utility::to_i64,
};

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive, Serialize, Deserialize,
)]
/// Resample mode for backend resample operations.
#[repr(u32)]
#[non_exhaustive]
pub enum ResampleMode {
    ///< Nearest-neighbor interpolation. @since cuDNN 9.0.0
    Nearest = sys::cudnnResampleMode_t::CUDNN_RESAMPLE_NEAREST as _,
    ///< Bilinear interpolation. @since cuDNN 9.0.0
    Bilinear = sys::cudnnResampleMode_t::CUDNN_RESAMPLE_BILINEAR as _,
    /// Average pooling with padded positions included in the divisor.
    AvgPoolIncludePadding = sys::cudnnResampleMode_t::CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING as _,
    ///< Average pooling excluding padding from divisor. @since cuDNN 9.0.0
    AvgPoolExcludePadding = sys::cudnnResampleMode_t::CUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING as _,
    ///< Max pooling. @since cuDNN 9.0.0
    MaxPool = sys::cudnnResampleMode_t::CUDNN_RESAMPLE_MAXPOOL as _,
}

impl_enum_conversion!(sys::cudnnResampleMode_t, ResampleMode);

impl_enum_display!(ResampleMode, {
    Self::Nearest => "CUDNN_RESAMPLE_NEAREST",
    Self::Bilinear => "CUDNN_RESAMPLE_BILINEAR",
    Self::AvgPoolIncludePadding => "CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING",
    Self::AvgPoolExcludePadding => "CUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING",
    Self::MaxPool => "CUDNN_RESAMPLE_MAXPOOL",
});

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive, Serialize, Deserialize,
)]
/// Padding mode for backend resample operations.
#[repr(u32)]
#[non_exhaustive]
pub enum PaddingMode {
    ///< Pads with zeros. @since cuDNN 9.0.0
    Zero = sys::cudnnPaddingMode_t::CUDNN_ZERO_PAD as _,
    ///< Pads with negative infinity (for max pooling). @since cuDNN 9.0.0
    NegInf = sys::cudnnPaddingMode_t::CUDNN_NEG_INF_PAD as _,
    ///< Pads by replicating edge values. @since cuDNN 9.0.0
    EdgeVal = sys::cudnnPaddingMode_t::CUDNN_EDGE_VAL_PAD as _,
}

impl_enum_conversion!(sys::cudnnPaddingMode_t, PaddingMode);

impl_enum_display!(PaddingMode, {
    Self::Zero => "CUDNN_ZERO_PAD",
    Self::NegInf => "CUDNN_NEG_INF_PAD",
    Self::EdgeVal => "CUDNN_EDGE_VAL_PAD",
});

/// Resolved native resampling attributes, retained after descriptor finalization.
/// Compute precision is explicit and independent of operand encodings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResampleDescriptorConfig {
    /// Native pooling or interpolation mode.
    pub mode: ResampleMode,
    /// Math precision selected independently of tensor storage encoding.
    pub compute_type: DataType,
    /// Native NaN propagation policy.
    pub nan_propagation: NanPropagation,
    /// Values used beyond the spatial input boundary.
    pub padding_mode: PaddingMode,
    /// Validated fractional padding, stride, and window values for every spatial axis.
    pub geometry: ResampleGeometry,
}

impl ResampleDescriptorConfig {
    /// Defaults to propagating NaNs and zero padding.
    #[must_use]
    pub fn new(mode: ResampleMode, compute_type: DataType, geometry: ResampleGeometry) -> Self {
        Self {
            mode,
            compute_type,
            nan_propagation: NanPropagation::Propagate,
            padding_mode: PaddingMode::Zero,
            geometry,
        }
    }

    /// Sets the native NaN propagation policy.
    #[must_use]
    pub fn with_nan_propagation(mut self, nan_propagation: NanPropagation) -> Self {
        self.nan_propagation = nan_propagation;
        self
    }

    /// Sets the values used beyond the spatial input boundary.
    #[must_use]
    pub fn with_padding_mode(mut self, padding_mode: PaddingMode) -> Self {
        self.padding_mode = padding_mode;
        self
    }
}

/// Finalized native descriptor and the exact configuration used to create it.
#[derive(Debug)]
pub struct ResampleDescriptor {
    descriptor: BackendDescriptor,
    config: ResampleDescriptorConfig,
}

impl ResampleDescriptor {
    /// Creates a native descriptor from validated spatial geometry and explicit math attributes.
    ///
    /// # Errors
    /// Returns an error for unrepresentable spatial rank or native attribute/finalization failure.
    pub fn create(config: ResampleDescriptorConfig) -> Result<Self> {
        let ResampleDescriptorConfig {
            mode,
            compute_type,
            nan_propagation,
            padding_mode,
            geometry,
        } = &config;
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::Resample)?;
        descriptor.set_attribute_enum(BackendAttributeName::ResampleMode, *mode)?;
        descriptor.set_attribute_enum(BackendAttributeName::ResampleCompType, *compute_type)?;
        descriptor.set_attribute_i64(
            BackendAttributeName::ResampleSpatialDims,
            to_i64(geometry.spatial_dims(), "spatial_dims")?,
        )?;
        descriptor.set_attribute_fraction_slice(
            BackendAttributeName::ResamplePrePaddings,
            geometry.pre_paddings(),
        )?;
        descriptor.set_attribute_fraction_slice(
            BackendAttributeName::ResamplePostPaddings,
            geometry.post_paddings(),
        )?;
        descriptor.set_attribute_fraction_slice(
            BackendAttributeName::ResampleStrides,
            geometry.strides(),
        )?;
        descriptor.set_attribute_fraction_slice(
            BackendAttributeName::ResampleWindowDims,
            geometry.window_dims(),
        )?;
        descriptor.set_attribute_enum(
            BackendAttributeName::ResampleNanPropagation,
            *nan_propagation,
        )?;
        descriptor.set_attribute_enum(BackendAttributeName::ResamplePaddingMode, *padding_mode)?;
        descriptor.finalize()?;

        Ok(Self { descriptor, config })
    }

    /// Retained attributes, including exact fractional geometry, without native queries or allocation.
    #[must_use]
    pub fn config(&self) -> &ResampleDescriptorConfig {
        &self.config
    }

    /// Borrows the finalized native resampling descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

/// Forward resampling value and optional saved indices.
/// Shared by frontend identities, native references and native output metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResampleOutputs<T = TensorId> {
    /// Resampled activation output.
    pub output: T,
    /// Optional saved pooling-index tensor.
    pub indices: Option<T>,
}

impl<T> ResampleOutputs<T> {
    /// Records the activation output and optional saved-index destination.
    pub fn new(output: T, indices: Option<T>) -> Self {
        Self { output, indices }
    }

    /// Borrows both results without requiring the operand type to be copyable.
    pub fn as_ref(&self) -> ResampleOutputs<&T> {
        ResampleOutputs {
            output: &self.output,
            indices: self.indices.as_ref(),
        }
    }

    /// Maps both results without changing whether saved indices are requested.
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> ResampleOutputs<U> {
        ResampleOutputs {
            output: map(self.output),
            indices: self.indices.map(map),
        }
    }

    /// Maps the value and requested indices without changing their roles.
    ///
    /// # Errors
    ///
    /// Returns the first error from the supplied tensor mapping, skipping absent optional operands.
    pub fn try_map<U, E>(
        self,
        mut map: impl FnMut(T) -> StdResult<U, E>,
    ) -> StdResult<ResampleOutputs<U>, E> {
        Ok(ResampleOutputs {
            output: map(self.output)?,
            indices: self.indices.map(map).transpose()?,
        })
    }
}

/// Retained attributes and native identities for forward resampling.
/// Does not borrow the original tensor or resampling descriptor owners.
#[derive(Debug, Clone, PartialEq)]
pub struct ResampleForwardOperationMetadata {
    /// Native identity of the forward input activation.
    pub input: BackendTensorUid,
    /// Native identities of the activation and optional saved-index outputs.
    pub outputs: ResampleOutputs<BackendTensorUid>,
    /// Retained native mode, math attributes, and exact fractional geometry.
    pub descriptor: ResampleDescriptorConfig,
    /// Double-precision multiplier for the newly computed result.
    pub alpha: f64,
    /// Double-precision multiplier for the existing destination accumulated into the result.
    pub beta: f64,
}

/// Finalized native resampling-forward operation retaining tensor identities, geometry, and scaling.
#[derive(Debug)]
pub struct ResampleForwardOperation {
    descriptor: BackendDescriptor,
    metadata: ResampleForwardOperationMetadata,
}

impl ResampleForwardOperation {
    /// Creates `y = alpha * resample(x) + beta * y`.
    /// Scaling factors use the same double-precision attributes as the C++ frontend.
    ///
    /// # Errors
    /// Returns an error if cuDNN rejects the descriptors or resampling attributes.
    pub fn create(
        resample: &ResampleDescriptor,
        x: &Tensor,
        outputs: ResampleOutputs<&Tensor>,
        alpha: f64,
        beta: f64,
    ) -> Result<Self> {
        let mut descriptor =
            BackendDescriptor::create(BackendDescriptorType::OperationResampleFwd)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationResampleFwdDesc,
            resample.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationResampleFwdXDesc,
            x.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationResampleFwdYDesc,
            outputs.output.descriptor(),
        )?;
        descriptor.set_attribute_f64(BackendAttributeName::OperationResampleFwdAlpha, alpha)?;
        descriptor.set_attribute_f64(BackendAttributeName::OperationResampleFwdBeta, beta)?;
        if let Some(indices) = outputs.indices {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationResampleFwdIdxDesc,
                indices.descriptor(),
            )?;
        }
        descriptor.finalize()?;

        Ok(Self {
            descriptor,
            metadata: ResampleForwardOperationMetadata {
                input: x.id(),
                outputs: outputs.map(Tensor::id),
                descriptor: resample.config().clone(),
                alpha,
                beta,
            },
        })
    }

    /// Retained operand identities, descriptor attributes and scaling factors.
    /// Does not query cuDNN or allocate.
    #[must_use]
    pub fn metadata(&self) -> &ResampleForwardOperationMetadata {
        &self.metadata
    }

    /// Native identities of the value and optional saved-index outputs.
    #[must_use]
    pub fn output_uids(&self) -> &ResampleOutputs<BackendTensorUid> {
        &self.metadata.outputs
    }

    /// Borrows the finalized native resampling descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

/// Operands for backward resampling.
/// Forward input/output tensors are optional at the descriptor level.
/// cuDNN requires both for NCHW layouts and may impose further mode-specific restrictions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResampleBackwardInputs<T> {
    /// Gradient with respect to the forward output.
    pub output_gradient: T,
    /// Optional saved forward input tensor.
    pub input: Option<T>,
    /// Optional saved forward output tensor.
    pub output: Option<T>,
    /// Optional saved pooling-index tensor.
    pub indices: Option<T>,
}

impl<T> ResampleBackwardInputs<T> {
    /// Creates a request without saved forward tensors or indices.
    pub fn new(output_gradient: T) -> Self {
        Self {
            output_gradient,
            input: None,
            output: None,
            indices: None,
        }
    }

    /// Borrows the gradient and any saved state while retaining their roles.
    pub fn as_ref(&self) -> ResampleBackwardInputs<&T> {
        ResampleBackwardInputs {
            output_gradient: &self.output_gradient,
            input: self.input.as_ref(),
            output: self.output.as_ref(),
            indices: self.indices.as_ref(),
        }
    }

    /// Maps the gradient and optional saved state without changing the request.
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> ResampleBackwardInputs<U> {
        ResampleBackwardInputs {
            output_gradient: map(self.output_gradient),
            input: self.input.map(&mut map),
            output: self.output.map(&mut map),
            indices: self.indices.map(map),
        }
    }

    /// Maps every operand while preserving optional saved state and its role.
    ///
    /// # Errors
    ///
    /// Returns the first error from the supplied tensor mapping, skipping absent optional operands.
    pub fn try_map<U, E>(
        self,
        mut map: impl FnMut(T) -> StdResult<U, E>,
    ) -> StdResult<ResampleBackwardInputs<U>, E> {
        Ok(ResampleBackwardInputs {
            output_gradient: map(self.output_gradient)?,
            input: self.input.map(&mut map).transpose()?,
            output: self.output.map(&mut map).transpose()?,
            indices: self.indices.map(map).transpose()?,
        })
    }
}

/// Retained attributes and native identities for backward resampling.
/// Keeps optional saved forward operands distinct from gradients and indices.
#[derive(Debug, Clone, PartialEq)]
pub struct ResampleBackwardOperationMetadata {
    /// Native identities of the output gradient and optional saved state.
    pub inputs: ResampleBackwardInputs<BackendTensorUid>,
    /// Native identity of the input-gradient destination.
    pub input_gradient: BackendTensorUid,
    /// Retained native mode, math attributes, and exact fractional geometry.
    pub descriptor: ResampleDescriptorConfig,
    /// Double-precision multiplier for the newly computed result.
    pub alpha: f64,
    /// Double-precision multiplier for the existing destination accumulated into the result.
    pub beta: f64,
}

/// Finalized native resampling-backward operation retaining gradient and saved-state identities.
#[derive(Debug)]
pub struct ResampleBackwardOperation {
    descriptor: BackendDescriptor,
    metadata: ResampleBackwardOperationMetadata,
}

impl ResampleBackwardOperation {
    /// Creates `dx = alpha * resample_backward(dy) + beta * dx`.
    /// Optional forward tensors and saved indices are passed through to cuDNN.
    ///
    /// # Errors
    /// Returns an error if cuDNN rejects the descriptors or resampling attributes.
    pub fn create(
        resample: &ResampleDescriptor,
        inputs: ResampleBackwardInputs<&Tensor>,
        dx: &Tensor,
        alpha: f64,
        beta: f64,
    ) -> Result<Self> {
        let mut descriptor =
            BackendDescriptor::create(BackendDescriptorType::OperationResampleBwd)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationResampleBwdDesc,
            resample.descriptor(),
        )?;
        if let Some(input) = inputs.input {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationResampleBwdXDesc,
                input.descriptor(),
            )?;
        }
        if let Some(output) = inputs.output {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationResampleBwdYDesc,
                output.descriptor(),
            )?;
        }
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationResampleBwdDyDesc,
            inputs.output_gradient.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationResampleBwdDxDesc,
            dx.descriptor(),
        )?;
        descriptor.set_attribute_f64(BackendAttributeName::OperationResampleBwdAlpha, alpha)?;
        descriptor.set_attribute_f64(BackendAttributeName::OperationResampleBwdBeta, beta)?;
        if let Some(indices) = inputs.indices {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationResampleBwdIdxDesc,
                indices.descriptor(),
            )?;
        }
        descriptor.finalize()?;

        Ok(Self {
            descriptor,
            metadata: ResampleBackwardOperationMetadata {
                inputs: inputs.map(Tensor::id),
                input_gradient: dx.id(),
                descriptor: resample.config().clone(),
                alpha,
                beta,
            },
        })
    }

    /// Retained gradient and saved-state identities, descriptor attributes and scaling factors.
    /// Does not query cuDNN or allocate.
    #[must_use]
    pub fn metadata(&self) -> &ResampleBackwardOperationMetadata {
        &self.metadata
    }

    /// Borrows the finalized native resampling descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}
