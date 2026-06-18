use std::ptr;

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cudnn_sys as sys;

use crate::{
    attribute::{BackendAttributeName, BackendAttributeType},
    context::Context,
    error::{Error, Result},
    try_ffi,
};

mod attributes;
mod cuda_graph;

/// Type tag for backend descriptors.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum BackendDescriptorType {
    Pointwise = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_POINTWISE_DESCRIPTOR as _,
    Convolution = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR as _,
    Engine = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_ENGINE_DESCRIPTOR as _,
    EngineCfg = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_ENGINECFG_DESCRIPTOR as _,
    EngineHeur = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR as _,
    ExecutionPlan = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR as _,
    IntermediateInfo =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR as _,
    KnobChoice = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR as _,
    KnobInfo = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR as _,
    LayoutInfo = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR as _,
    OperationConvolutionForward =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR as _,
    OperationConvolutionBackwardFilter = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR as _,
    OperationConvolutionBackwardData = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR as _,
    OperationPointwise =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR as _,
    OperationGenStats =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR as _,
    OperationGraph = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR as _,
    VariantPack = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR as _,
    Tensor = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_TENSOR_DESCRIPTOR as _,
    Matmul = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_MATMUL_DESCRIPTOR as _,
    OperationMatmul =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR as _,
    OperationBnFinalizeStatistics = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR as _,
    Reduction = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_REDUCTION_DESCRIPTOR as _,
    OperationReduction =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR as _,
    OperationBnBwdWeights =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR as _,
    Resample = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_RESAMPLE_DESCRIPTOR as _,
    OperationResampleFwd =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR as _,
    OperationResampleBwd =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR as _,
    OperationConcat =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR as _,
    OperationSignal =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR as _,
    OperationNormForward =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR as _,
    OperationNormBackward =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR as _,
    OperationSdpaForward =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR as _,
    OperationReshape =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR as _,
    OperationExpandBandMatrix =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_EXPAND_BAND_MATRIX_DESCRIPTOR as _,
    OperationContractBandMatrix =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_CONTRACT_BAND_MATRIX_DESCRIPTOR as _,
    Rng = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_RNG_DESCRIPTOR as _,
    OperationRng = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR as _,
    KernelCache = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_KERNEL_CACHE_DESCRIPTOR as _,
    OperationPagedCacheLoad =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR as _,
    OperationBlockScaleQuantize = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR as _,
    OperationBlockScaleDequantize = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR as _,
    DeviceProp = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_DEVICEPROP_DESCRIPTOR as _,
    OperationMoeGroupedMatmul =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR as _,
    OperationSdpaBackward =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR as _,
    OperationDiagonalBandMask =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_DIAGONAL_BAND_MASK_DESCRIPTOR as _,
    OperationSoftmax =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_SOFTMAX_DESCRIPTOR as _,
}

impl_enum_conversion!(sys::cudnnBackendDescriptorType_t, BackendDescriptorType);

impl_enum_display!(BackendDescriptorType, {
    Self::Pointwise => "CUDNN_BACKEND_POINTWISE_DESCRIPTOR",
    Self::Convolution => "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR",
    Self::Engine => "CUDNN_BACKEND_ENGINE_DESCRIPTOR",
    Self::EngineCfg => "CUDNN_BACKEND_ENGINECFG_DESCRIPTOR",
    Self::EngineHeur => "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR",
    Self::ExecutionPlan => "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR",
    Self::IntermediateInfo => "CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR",
    Self::KnobChoice => "CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR",
    Self::KnobInfo => "CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR",
    Self::LayoutInfo => "CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR",
    Self::OperationConvolutionForward => "CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR",
    Self::OperationConvolutionBackwardFilter => "CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR",
    Self::OperationConvolutionBackwardData => "CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR",
    Self::OperationPointwise => "CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR",
    Self::OperationGenStats => "CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR",
    Self::OperationGraph => "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR",
    Self::VariantPack => "CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR",
    Self::Tensor => "CUDNN_BACKEND_TENSOR_DESCRIPTOR",
    Self::Matmul => "CUDNN_BACKEND_MATMUL_DESCRIPTOR",
    Self::OperationMatmul => "CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR",
    Self::OperationBnFinalizeStatistics => "CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR",
    Self::Reduction => "CUDNN_BACKEND_REDUCTION_DESCRIPTOR",
    Self::OperationReduction => "CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR",
    Self::OperationBnBwdWeights => "CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR",
    Self::Resample => "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR",
    Self::OperationResampleFwd => "CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR",
    Self::OperationResampleBwd => "CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR",
    Self::OperationConcat => "CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR",
    Self::OperationSignal => "CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR",
    Self::OperationNormForward => "CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR",
    Self::OperationNormBackward => "CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR",
    Self::OperationSdpaForward => "CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR",
    Self::OperationReshape => "CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR",
    Self::OperationExpandBandMatrix => "CUDNN_BACKEND_OPERATION_EXPAND_BAND_MATRIX_DESCRIPTOR",
    Self::OperationContractBandMatrix => "CUDNN_BACKEND_OPERATION_CONTRACT_BAND_MATRIX_DESCRIPTOR",
    Self::Rng => "CUDNN_BACKEND_RNG_DESCRIPTOR",
    Self::OperationRng => "CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR",
    Self::KernelCache => "CUDNN_BACKEND_KERNEL_CACHE_DESCRIPTOR",
    Self::OperationPagedCacheLoad => "CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR",
    Self::OperationBlockScaleQuantize => "CUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR",
    Self::OperationBlockScaleDequantize => "CUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR",
    Self::DeviceProp => "CUDNN_BACKEND_DEVICEPROP_DESCRIPTOR",
    Self::OperationMoeGroupedMatmul => "CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR",
    Self::OperationSdpaBackward => "CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR",
    Self::OperationDiagonalBandMask => "CUDNN_BACKEND_OPERATION_DIAGONAL_BAND_MASK_DESCRIPTOR",
    Self::OperationSoftmax => "CUDNN_BACKEND_OPERATION_SOFTMAX_DESCRIPTOR",
});

#[derive(Debug)]
pub struct BackendDescriptor {
    handle: sys::cudnnBackendDescriptor_t,
    finalized: bool,
    descriptor_type: BackendDescriptorType,
    owned: bool,
}

// Backend descriptors expose mutation through &mut Self only. Shared
// references can inspect finalized descriptor state without rebinding storage.
unsafe impl Send for BackendDescriptor {}
unsafe impl Sync for BackendDescriptor {}

impl BackendDescriptor {
    /// Creates a backend descriptor of the requested type.
    ///
    /// The returned descriptor owns the cuDNN backend descriptor handle and
    /// destroys it when dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDNN does not return a valid descriptor handle, if
    /// host memory allocation fails, or if cuDNN does not support creating the
    /// requested descriptor type.
    pub fn create(descriptor_type: BackendDescriptorType) -> Result<Self> {
        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cudnnBackendCreateDescriptor(
                descriptor_type.into(),
                &raw mut handle
            ))?;
        }
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(BackendDescriptor {
            handle,
            finalized: false,
            descriptor_type,
            owned: true,
        })
    }

    pub(crate) fn from_raw_finalized_borrowed(
        handle: sys::cudnnBackendDescriptor_t,
        descriptor_type: BackendDescriptorType,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            finalized: true,
            descriptor_type,
            owned: false,
        })
    }

    /// Takes ownership of a raw cuDNN backend descriptor handle.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `cudnnBackendDescriptor_t` created by cuDNN.
    /// The returned wrapper takes ownership and will destroy it with
    /// `cudnnBackendDestroyDescriptor`; no other owner may destroy or keep
    /// using it. `finalized` and `descriptor_type` must accurately describe
    /// the descriptor state.
    pub unsafe fn from_raw(
        handle: sys::cudnnBackendDescriptor_t,
        descriptor_type: BackendDescriptorType,
        finalized: bool,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            finalized,
            descriptor_type,
            owned: true,
        })
    }

    /// Returns the raw cuDNN backend descriptor.
    ///
    /// The returned handle must not be destroyed by the caller. Mutating it
    /// directly can violate this wrapper's finalized-state tracking.
    pub fn as_raw(&self) -> sys::cudnnBackendDescriptor_t {
        self.handle
    }

    /// Releases ownership and returns the raw cuDNN backend descriptor handle.
    ///
    /// The caller becomes responsible for destroying the handle.
    pub fn into_raw(self) -> sys::cudnnBackendDescriptor_t {
        let handle = self.handle;
        std::mem::forget(self);
        handle
    }

    /// Checks if the descriptor has been finalized.
    pub fn is_finalized(&self) -> bool {
        self.finalized
    }

    /// Returns the type of the descriptor.
    pub fn descriptor_type(&self) -> BackendDescriptorType {
        self.descriptor_type
    }

    pub fn is_null(&self) -> bool {
        self.handle.is_null()
    }

    /// Finalizes this backend descriptor.
    ///
    /// Finalization checks the attributes set on the descriptor and marks the descriptor as
    /// finalized. Once finalized, attributes can be retrieved but no longer modified.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor attributes are invalid or unsupported
    /// by the current cuDNN version, or if cuDNN reports an internal failure.
    pub fn finalize(&mut self) -> Result<()> {
        if self.finalized {
            return Err(Error::DescriptorAlreadyFinalized);
        }
        unsafe {
            try_ffi!(sys::cudnnBackendFinalize(self.handle))?;
        }
        self.finalized = true;
        Ok(())
    }

    pub(crate) fn mark_finalized(&mut self) {
        self.finalized = true;
    }

    /// Helper to check if modification is allowed.
    fn check_modifiable(&self) -> Result<()> {
        if self.finalized {
            Err(Error::DescriptorCannotModifyFinalized)
        } else {
            Ok(())
        }
    }

    fn check_finalized(&self) -> Result<()> {
        if self.finalized {
            Ok(())
        } else {
            Err(Error::DescriptorRequiredFinalized)
        }
    }

    fn set_attribute_raw(
        &mut self,
        name: BackendAttributeName,
        ty: BackendAttributeType,
        element_count: i64,
        values: *const (),
    ) -> Result<()> {
        self.check_modifiable()?;
        unsafe {
            try_ffi!(sys::cudnnBackendSetAttribute(
                self.handle,
                name.into(),
                ty.into(),
                element_count,
                values.cast(),
            ))?;
        }
        Ok(())
    }

    fn set_attribute_slice_raw<T>(
        &mut self,
        name: BackendAttributeName,
        ty: BackendAttributeType,
        values: &[T],
        value_name: &'static str,
    ) -> Result<()> {
        if values.is_empty() {
            return Err(Error::EmptyList {
                name: value_name.into(),
            });
        }
        self.set_attribute_raw(name, ty, values.len() as _, values.as_ptr().cast())
    }

    fn attribute_raw<T: Default>(
        &self,
        name: BackendAttributeName,
        ty: BackendAttributeType,
    ) -> Result<T> {
        self.check_finalized()?;
        let mut value = T::default();
        let mut count_written: i64 = 0;
        unsafe {
            try_ffi!(sys::cudnnBackendGetAttribute(
                self.handle,
                name.into(),
                ty.into(),
                1,
                &raw mut count_written,
                (&raw mut value).cast(),
            ))?;
        }
        if count_written != 1 {
            return Err(Error::DescriptorAttributeNotFound(name));
        }
        Ok(value)
    }

    fn attribute_count(&self, name: BackendAttributeName, ty: BackendAttributeType) -> Result<i64> {
        self.check_finalized()?;
        let mut count = 0;
        unsafe {
            try_ffi!(sys::cudnnBackendGetAttribute(
                self.handle,
                name.into(),
                ty.into(),
                0,
                &raw mut count,
                ptr::null_mut(),
            ))?;
        }
        Ok(count)
    }

    fn attribute_raw_vec<T: Copy>(
        &self,
        name: BackendAttributeName,
        ty: BackendAttributeType,
        initial_value: T,
        value_name: &'static str,
    ) -> Result<Vec<T>> {
        let count = self.attribute_count(name, ty)?;
        if count == 0 {
            return Ok(Vec::new());
        }

        let count = usize::try_from(count).map_err(|_| Error::InvalidBackendDescriptorCount {
            name: value_name.into(),
        })?;
        let mut values = vec![initial_value; count];
        let mut count_written = 0;
        unsafe {
            try_ffi!(sys::cudnnBackendGetAttribute(
                self.handle,
                name.into(),
                ty.into(),
                i64::try_from(count).map_err(|_| Error::InvalidBackendDescriptorCount {
                    name: value_name.into(),
                })?,
                &raw mut count_written,
                values.as_mut_ptr().cast(),
            ))?;
        }

        values.truncate(usize::try_from(count_written).map_err(|_| {
            Error::InvalidBackendDescriptorCount {
                name: value_name.into(),
            }
        })?);
        Ok(values)
    }
}

impl Drop for BackendDescriptor {
    fn drop(&mut self) {
        if !self.owned || self.handle.is_null() {
            return;
        }
        unsafe {
            let _ = try_ffi!(sys::cudnnBackendDestroyDescriptor(self.handle));
        }
    }
}

/// Executes the given engine configuration plan on the [`VariantPack`](crate::execution::VariantPack)
/// and the finalized [`ExecutionPlan`](crate::execution::ExecutionPlan) on the data.
/// The data and the working space are encapsulated in the [`VariantPack`](crate::execution::VariantPack).
///
/// # Errors
///
/// Returns an error if `plan` or `varpack` is not finalized or has the wrong
/// descriptor type, if required data pointers are invalid, if plan execution
/// fails, or if cuDNN reports an internal failure.
// TODO: better types?
pub(crate) fn backend_execute(
    ctx: &Context,
    plan: &BackendDescriptor,
    varpack: &BackendDescriptor,
) -> Result<()> {
    if !plan.finalized || !varpack.finalized {
        return Err(Error::DescriptorRequiredFinalized);
    }
    if plan.descriptor_type != BackendDescriptorType::ExecutionPlan {
        return Err(Error::DescriptorExpectedExecutionPlan);
    }
    if varpack.descriptor_type != BackendDescriptorType::VariantPack {
        return Err(Error::DescriptorExpectedVariantPack);
    }

    unsafe {
        try_ffi!(sys::cudnnBackendExecute(
            ctx.as_raw(),
            plan.as_raw(),
            varpack.as_raw()
        ))?;
    }

    Ok(())
}
