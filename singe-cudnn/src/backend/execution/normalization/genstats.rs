use std::result::Result as StdResult;

use num_enum::{IntoPrimitive, TryFromPrimitive};

use serde::{Deserialize, Serialize};

use singe_core::{impl_enum_conversion, impl_enum_display};

use singe_cudnn_sys as sys;

use crate::{
    backend::{
        attribute::BackendAttributeName,
        descriptor::{BackendDescriptor, BackendDescriptorType},
        tensor::{BackendTensorUid, Tensor, TensorId},
    },
    data_type::DataType,
    error::Result,
};

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive, Serialize, Deserialize,
)]
/// Statistics mode for backend statistics generation operations.
#[repr(u32)]
#[non_exhaustive]
pub enum GenStatsMode {
    /// Computes and writes the sum and sum of squares of the input tensor along the specified dimensions.
    /// Currently supported reduction dimensions are limited to per-channel reductions.
    SumSqsum = sys::cudnnGenStatsMode_t::CUDNN_GENSTATS_SUM_SQSUM as _,
}

impl_enum_conversion!(sys::cudnnGenStatsMode_t, GenStatsMode);

impl_enum_display!(GenStatsMode, {
    Self::SumSqsum => "CUDNN_GENSTATS_SUM_SQSUM",
});

/// Per-channel sum and square-sum results shared by native and frontend construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenStatsOutputs<T = TensorId> {
    /// Per-channel sum tensor.
    pub sum: T,
    /// Per-channel sum-of-squares tensor.
    pub square_sum: T,
}

impl<T> GenStatsOutputs<T> {
    /// Records the per-channel sum and sum-of-squares output tensors.
    pub fn new(sum: T, square_sum: T) -> Self {
        Self { sum, square_sum }
    }

    /// Borrows both output owners while preserving their statistic roles.
    pub fn as_ref(&self) -> GenStatsOutputs<&T> {
        GenStatsOutputs {
            sum: &self.sum,
            square_sum: &self.square_sum,
        }
    }

    /// Maps both outputs without changing their statistic roles.
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> GenStatsOutputs<U> {
        GenStatsOutputs {
            sum: map(self.sum),
            square_sum: map(self.square_sum),
        }
    }

    /// Maps both outputs while preserving their statistic roles.
    ///
    /// # Errors
    ///
    /// Returns the first error from the supplied tensor mapping; later operands are not mapped.
    pub fn try_map<U, E>(
        self,
        mut map: impl FnMut(T) -> StdResult<U, E>,
    ) -> StdResult<GenStatsOutputs<U>, E> {
        Ok(GenStatsOutputs {
            sum: map(self.sum)?,
            square_sum: map(self.square_sum)?,
        })
    }
}

/// Retained statistics mode, independent math precision and native operand identities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenStatsOperationMetadata {
    /// Native statistics mode.
    pub mode: GenStatsMode,
    /// Math precision selected independently of tensor storage encodings.
    pub compute_type: DataType,
    /// Retained native input activation identity.
    pub input: BackendTensorUid,
    /// Retained native identities of sum and sum-of-squares outputs.
    pub outputs: GenStatsOutputs<BackendTensorUid>,
}

/// Finalized native per-channel sum and sum-of-squares operation with retained tensor identities.
#[derive(Debug)]
pub struct GenStatsOperation {
    descriptor: BackendDescriptor,
    metadata: GenStatsOperationMetadata,
}

impl GenStatsOperation {
    /// Creates per-channel sums using explicit math precision independently of tensor storage.
    ///
    /// # Errors
    ///
    /// Returns an error if native descriptor creation, attribute assignment, or finalization fails. cuDNN validates the mode, math precision, and input/output tensor layouts.
    pub fn create(
        mode: GenStatsMode,
        compute_type: DataType,
        x: &Tensor,
        outputs: GenStatsOutputs<&Tensor>,
    ) -> Result<Self> {
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::OperationGenStats)?;
        descriptor.set_attribute_enum(BackendAttributeName::OperationGenStatsMode, mode)?;
        descriptor.set_attribute_enum(
            BackendAttributeName::OperationGenStatsMathPrec,
            compute_type,
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationGenStatsXDesc,
            x.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationGenStatsSumDesc,
            outputs.sum.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationGenStatsSqSumDesc,
            outputs.square_sum.descriptor(),
        )?;
        descriptor.finalize()?;

        Ok(Self {
            descriptor,
            metadata: GenStatsOperationMetadata {
                mode,
                compute_type,
                input: x.id(),
                outputs: outputs.map(Tensor::id),
            },
        })
    }

    /// Retained mode, precision and tensor UIDs without native queries or allocation.
    #[must_use]
    pub fn metadata(&self) -> &GenStatsOperationMetadata {
        &self.metadata
    }

    /// Borrows the finalized native statistics-helper descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}
