mod tensors;

use std::result::Result as StdResult;

use num_enum::{IntoPrimitive, TryFromPrimitive};
use serde::{Deserialize, Serialize};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cudnn_sys as sys;
pub use tensors::{
    MoeGroupedMatmulBackwardInputs, MoeGroupedMatmulBackwardTensors, MoeGroupedMatmulInputs,
    MoeGroupedMatmulTensors,
};

use crate::{
    backend::{
        attribute::BackendAttributeName,
        descriptor::{BackendDescriptor, BackendDescriptorType},
        tensor::{BackendTensorUid, Tensor},
    },
    data_type::DataType,
    error::Result,
    utility::check_range,
};

/// Retained native math precision and operand identities for grouped weight gradients.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeGroupedMatmulBackwardOperationMetadata {
    /// Retained native math precision, independent of tensor storage encodings.
    pub compute_type: DataType,
    /// Native identities of tokens, output gradients, expert offsets, and weight-gradient output.
    pub tensors: MoeGroupedMatmulBackwardTensors<BackendTensorUid>,
}

/// Native grouped weight-gradient operation for mixture-of-experts layers.
#[derive(Debug)]
pub struct MoeGroupedMatmulBackwardOperation {
    descriptor: BackendDescriptor,
    metadata: MoeGroupedMatmulBackwardOperationMetadata,
}

impl MoeGroupedMatmulBackwardOperation {
    /// Creates grouped weight gradients from named native operands and an explicit math type.
    /// cuDNN determines supported encodings, offset formats and layouts during finalization.
    ///
    /// # Errors
    ///
    /// Returns an error if native descriptor creation, attribute assignment, or finalization fails. cuDNN validates encodings, offset formats, layouts, and grouped-matmul support.
    pub fn create(
        compute_type: DataType,
        tensors: MoeGroupedMatmulBackwardTensors<&Tensor>,
    ) -> Result<Self> {
        let mut descriptor =
            BackendDescriptor::create(BackendDescriptorType::OperationMoeGroupedMatmulBackward)?;
        descriptor.set_attribute_enum(
            BackendAttributeName::OperationMoeGroupedMatmulBackwardMathPrec,
            compute_type,
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationMoeGroupedMatmulBackwardTokenDesc,
            tensors.inputs.token.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationMoeGroupedMatmulBackwardOutputGradientDesc,
            tensors.inputs.output_gradient.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationMoeGroupedMatmulBackwardFirstTokenOffsetDesc,
            tensors.inputs.first_token_offset.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationMoeGroupedMatmulBackwardWeightGradientDesc,
            tensors.weight_gradient.descriptor(),
        )?;
        descriptor.finalize()?;
        Ok(Self {
            descriptor,
            metadata: MoeGroupedMatmulBackwardOperationMetadata {
                compute_type,
                tensors: tensors.map(Tensor::id),
            },
        })
    }

    /// Retained math precision and native operand identities without queries or allocation.
    #[must_use]
    pub fn metadata(&self) -> &MoeGroupedMatmulBackwardOperationMetadata {
        &self.metadata
    }

    /// Actual native identities of the required operands and weight-gradient result.
    #[must_use]
    pub fn tensor_uids(&self) -> &MoeGroupedMatmulBackwardTensors<BackendTensorUid> {
        &self.metadata.tensors
    }

    /// Borrows the finalized native grouped-matmul operation descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

/// Mixture-of-Experts grouped matmul modes.
///
/// since: cuDNN 9.15.0
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive, Serialize, Deserialize,
)]
#[repr(u32)]
#[non_exhaustive]
pub enum MoeGroupedMatmulMode {
    ///< No gather/scatter. @since cuDNN 9.15.0
    None = sys::cudnnMoeGroupedMatmulMode_t::CUDNN_MOE_GROUPED_MATMUL_MODE_NONE as _,
    ///< Gather mode. @since cuDNN 9.15.0
    Gather = sys::cudnnMoeGroupedMatmulMode_t::CUDNN_MOE_GROUPED_MATMUL_MODE_GATHER as _,
    ///< Scatter mode. @since cuDNN 9.15.0
    Scatter = sys::cudnnMoeGroupedMatmulMode_t::CUDNN_MOE_GROUPED_MATMUL_MODE_SCATTER as _,
}

impl_enum_conversion!(sys::cudnnMoeGroupedMatmulMode_t, MoeGroupedMatmulMode);

impl_enum_display!(MoeGroupedMatmulMode, {
    Self::None => "CUDNN_MOE_GROUPED_MATMUL_MODE_NONE",
    Self::Gather => "CUDNN_MOE_GROUPED_MATMUL_MODE_GATHER",
    Self::Scatter => "CUDNN_MOE_GROUPED_MATMUL_MODE_SCATTER",
});

/// Routing operands for the native `MoE` modes.
/// Scatter carries its required expert-slot indices and top-k count together.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MoeGroupedMatmulRouting<T> {
    /// Uses already grouped operands without gather or scatter routing.
    None,
    /// Gathers token activations using the routing-index tensor.
    Gather {
        /// INT32 token-routing index tensor for gathering input activations.
        token_index: T,
    },
    /// Scatters grouped results using token and expert-slot routing tensors.
    Scatter {
        /// INT32 token-routing index tensor for scattering outputs.
        token_index: T,
        /// INT32 scatter routing-slot tensor supplied to the native token-ks attribute.
        token_ks: T,
        /// Positive number of expert slots per token.
        top_k: i32,
    },
}

impl<T> MoeGroupedMatmulRouting<T> {
    /// Reads the mode without consuming or copying tensor owners.
    pub fn mode(&self) -> MoeGroupedMatmulMode {
        match self {
            Self::None => MoeGroupedMatmulMode::None,
            Self::Gather { .. } => MoeGroupedMatmulMode::Gather,
            Self::Scatter { .. } => MoeGroupedMatmulMode::Scatter,
        }
    }

    /// Borrows routing operands while preserving the mode and top-k count.
    pub fn as_ref(&self) -> MoeGroupedMatmulRouting<&T> {
        match self {
            Self::None => MoeGroupedMatmulRouting::None,
            Self::Gather { token_index } => MoeGroupedMatmulRouting::Gather { token_index },
            Self::Scatter {
                token_index,
                token_ks,
                top_k,
            } => MoeGroupedMatmulRouting::Scatter {
                token_index,
                token_ks,
                top_k: *top_k,
            },
        }
    }

    /// Maps present routing operands without changing their mode or top-k count.
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> MoeGroupedMatmulRouting<U> {
        match self {
            Self::None => MoeGroupedMatmulRouting::None,
            Self::Gather { token_index } => MoeGroupedMatmulRouting::Gather {
                token_index: map(token_index),
            },
            Self::Scatter {
                token_index,
                token_ks,
                top_k,
            } => MoeGroupedMatmulRouting::Scatter {
                token_index: map(token_index),
                token_ks: map(token_ks),
                top_k,
            },
        }
    }

    /// Transforms routing operands while preserving their mode and top-k count.
    ///
    /// # Errors
    ///
    /// Returns the first routing-tensor mapping error, processing token indices before scatter slots.
    pub fn try_map<U, E>(
        self,
        mut map: impl FnMut(T) -> StdResult<U, E>,
    ) -> StdResult<MoeGroupedMatmulRouting<U>, E> {
        Ok(match self {
            Self::None => MoeGroupedMatmulRouting::None,
            Self::Gather { token_index } => MoeGroupedMatmulRouting::Gather {
                token_index: map(token_index)?,
            },
            Self::Scatter {
                token_index,
                token_ks,
                top_k,
            } => MoeGroupedMatmulRouting::Scatter {
                token_index: map(token_index)?,
                token_ks: map(token_ks)?,
                top_k,
            },
        })
    }
}

impl<T: Copy> MoeGroupedMatmulRouting<T> {
    /// Returns the token-routing index tensor for gather or scatter, or `None` without routing.
    pub fn token_index(self) -> Option<T> {
        match self {
            Self::None => None,
            Self::Gather { token_index } | Self::Scatter { token_index, .. } => Some(token_index),
        }
    }

    /// Returns the routing-slot tensor for scatter, or `None` for another mode.
    pub fn token_ks(self) -> Option<T> {
        match self {
            Self::Scatter { token_ks, .. } => Some(token_ks),
            _ => None,
        }
    }
}

/// Retained math precision, required operands and optional native routing choices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeGroupedMatmulOperationMetadata {
    /// Retained native math precision, independent of tensor storage encodings.
    pub compute_type: DataType,
    /// Native identities of required operands and the grouped-matmul output.
    pub tensors: MoeGroupedMatmulTensors<BackendTensorUid>,
    /// Retained gather/scatter mode, routing identities, and top-k count.
    pub routing: MoeGroupedMatmulRouting<BackendTensorUid>,
}

/// Finalized native expert-grouped matmul with retained operands, precision, and gather/scatter routing.
#[derive(Debug)]
pub struct MoeGroupedMatmulOperation {
    descriptor: BackendDescriptor,
    metadata: MoeGroupedMatmulOperationMetadata,
}

impl MoeGroupedMatmulOperation {
    /// Creates a grouped matmul from named required tensors and independent gather/scatter routing.
    /// Scatter's top-k count is validated before allocating a descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if native descriptor creation, attribute assignment, or finalization fails. cuDNN validates encodings, offset formats, layouts, and grouped-matmul support. Scatter also returns an error for a nonpositive top-k count.
    pub fn create(
        routing: MoeGroupedMatmulRouting<&Tensor>,
        compute_type: DataType,
        tensors: MoeGroupedMatmulTensors<&Tensor>,
    ) -> Result<Self> {
        if let MoeGroupedMatmulRouting::Scatter { top_k, .. } = routing {
            check_range!("moe top k", top_k > 0)?;
        }
        let mut descriptor =
            BackendDescriptor::create(BackendDescriptorType::OperationMoeGroupedMatmul)?;
        descriptor.set_attribute_enum(
            BackendAttributeName::OperationMoeGroupedMatmulMode,
            routing.mode(),
        )?;
        descriptor.set_attribute_enum(
            BackendAttributeName::OperationMoeGroupedMatmulMathPrec,
            compute_type,
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationMoeGroupedMatmulTokenDesc,
            tensors.inputs.token.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationMoeGroupedMatmulWeightDesc,
            tensors.inputs.weight.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationMoeGroupedMatmulFirstTokenOffsetDesc,
            tensors.inputs.first_token_offset.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationMoeGroupedMatmulOutputDesc,
            tensors.output.descriptor(),
        )?;
        if let Some(token_index) = routing.token_index() {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationMoeGroupedMatmulTokenIndexDesc,
                token_index.descriptor(),
            )?;
        }
        if let MoeGroupedMatmulRouting::Scatter {
            token_ks, top_k, ..
        } = routing
        {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationMoeGroupedMatmulTokenKsDesc,
                token_ks.descriptor(),
            )?;
            descriptor
                .set_attribute_i32(BackendAttributeName::OperationMoeGroupedMatmulTopK, top_k)?;
        }
        descriptor.finalize()?;

        Ok(Self {
            descriptor,
            metadata: MoeGroupedMatmulOperationMetadata {
                compute_type,
                tensors: tensors.map(Tensor::id),
                routing: routing.map(Tensor::id),
            },
        })
    }

    /// Retained math precision, native operands and routing without queries or allocation.
    #[must_use]
    pub fn metadata(&self) -> &MoeGroupedMatmulOperationMetadata {
        &self.metadata
    }

    /// Actual native identities of the required operands and output.
    #[must_use]
    pub fn tensor_uids(&self) -> &MoeGroupedMatmulTensors<BackendTensorUid> {
        &self.metadata.tensors
    }

    /// Native identities of the selected gather/scatter operands, retaining their routing mode.
    #[must_use]
    pub fn routing_uids(&self) -> &MoeGroupedMatmulRouting<BackendTensorUid> {
        &self.metadata.routing
    }

    /// Borrows the finalized native grouped-matmul operation descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}
