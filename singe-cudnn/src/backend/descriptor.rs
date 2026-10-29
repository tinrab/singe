mod attributes;
mod cuda_graph;

use std::{mem::replace, ptr};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cudnn_sys as sys;

use crate::{
    backend::attribute::{BackendAttributeName, BackendAttributeType},
    context::Context,
    error::{Error, Result, Status},
    try_ffi,
    utility::{to_i64, to_usize},
};

/// Type tag for backend descriptors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
#[repr(u32)]
pub enum BackendDescriptorType {
    ///< Backward `MoE` grouped matmul. @since cuDNN 9.22.0
    OperationMoeGroupedMatmulBackward = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR as _,
    ///< Rotary Position Embedding (`RoPE`) forward. @since cuDNN 9.24.0
    OperationRopeForward = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_ROPE_FWD_DESCRIPTOR as _,
    ///< Rotary Position Embedding (`RoPE`) backward. @since cuDNN 9.25.0
    OperationRopeBackward = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_ROPE_BWD_DESCRIPTOR as _,

    ///< Slice operation. @since cuDNN 9.22.0
    OperationSlice = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_SLICE_DESCRIPTOR as _,
    ///< Transpose operation. @since cuDNN 9.22.0
    OperationTranspose = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_TRANSPOSE_DESCRIPTOR as _,
    ///< Pointwise op config: mode, math precision, activation params. @since cuDNN 9.0.0
    Pointwise = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_POINTWISE_DESCRIPTOR as _,
    ///< Convolution config: compute type, mode, dilation, stride, padding. @since cuDNN 9.0.0
    Convolution = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR as _,
    ///< Engine (kernel grouping) to compute an operation graph. @since cuDNN 9.0.0
    Engine = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_ENGINE_DESCRIPTOR as _,
    ///< Engine configuration: engine descriptor plus knob choices. @since cuDNN 9.0.0
    EngineCfg = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_ENGINECFG_DESCRIPTOR as _,
    ///< Engine configurations ranked by performance via cuDNN heuristics. @since cuDNN 9.0.0
    EngineHeur = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR as _,
    ///< Finalized execution plan: engine config, workspace size, optional kernel cache. @since cuDNN 9.0.0
    ExecutionPlan = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR as _,
    ///< Read-only info about a reusable execution intermediate. @since cuDNN 9.0.0
    IntermediateInfo =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR as _,
    ///< Type and value of an engine performance tuning knob. @since cuDNN 9.0.0
    KnobChoice = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR as _,
    ///< Read-only info about an engine knob: type, min/max, and stride. @since cuDNN 9.0.0
    KnobInfo = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR as _,
    ///< Read-only info on the preferred memory layout for a tensor. @since cuDNN 9.0.0
    LayoutInfo = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR as _,
    ///< Forward convolution: y = alpha * conv(w, x) + beta * y. @since cuDNN 9.0.0
    OperationConvolutionForward =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR as _,
    ///< Backward filter convolution: computes dw from x and dy. @since cuDNN 9.0.0
    OperationConvolutionBackwardFilter = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR as _,
    ///< Backward data convolution: computes dx from w and dy. @since cuDNN 9.0.0
    OperationConvolutionBackwardData = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR as _,
    ///< Pointwise operation: Y = op(alpha1*X) or Y = op(alpha1*X, alpha2*B). @since cuDNN 9.0.0
    OperationPointwise =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR as _,
    ///< Generates per-channel statistics (sum and sum-of-squares). @since cuDNN 9.0.0
    OperationGenStats =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR as _,
    ///< Operation graph: a DAG of operations connected by virtual tensors. @since cuDNN 9.0.0
    OperationGraph = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR as _,
    ///< Binds device pointers to non-virtual tensors, workspace, and intermediates. @since cuDNN 9.0.0
    VariantPack = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR as _,
    ///< Tensor: data type, dimensions, strides, alignment, unique ID, virtual flag. @since cuDNN 9.0.0
    Tensor = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_TENSOR_DESCRIPTOR as _,
    ///< Matrix multiply config: compute type and padding value. @since cuDNN 9.0.0
    Matmul = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_MATMUL_DESCRIPTOR as _,
    ///< Matrix multiplication operation: C = A * B with optional overrides. @since cuDNN 9.0.0
    OperationMatmul =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR as _,
    ///< BN finalize: computes running stats, saved mean/invstd, eq scale/bias. @since cuDNN 9.0.0
    OperationBnFinalizeStatistics = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR as _,
    ///< Reduction config: operator type (ADD/MUL/MIN/MAX/etc.) and compute type. @since cuDNN 9.0.0
    Reduction = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_REDUCTION_DESCRIPTOR as _,
    ///< Reduces input tensor values along one or more dimensions. @since cuDNN 9.0.0
    OperationReduction =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR as _,
    ///< BN backward weights: computes dScale, dBias, and equivalent gradients. @since cuDNN 9.0.0
    OperationBnBwdWeights =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR as _,
    ///< Resample config: mode, spatial dims, padding, strides, window dims. @since cuDNN 9.0.0
    Resample = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_RESAMPLE_DESCRIPTOR as _,
    ///< Forward resampling (pooling/interpolation) with alpha/beta scaling. @since cuDNN 9.0.0
    OperationResampleFwd =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR as _,
    ///< Backward resampling: computes dx from dy with alpha/beta scaling. @since cuDNN 9.0.0
    OperationResampleBwd =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR as _,
    ///< Concatenates multiple tensors along a given axis. @since cuDNN 9.0.0
    OperationConcat =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR as _,
    ///< Updates or waits on a flag variable for inter-graph signaling. @since cuDNN 9.0.0
    OperationSignal =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR as _,
    ///< Forward normalization (layer/instance/batch/RMS) with optional running stats. @since cuDNN 9.0.0
    OperationNormForward =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR as _,
    ///< Backward normalization: computes dX, dScale, dBias from dY. @since cuDNN 9.0.0
    OperationNormBackward =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR as _,
    ///< Scaled dot-product attention forward (fused flash attention). @since cuDNN 9.13.0
    OperationSdpaForward =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR as _,
    ///< Reshapes a tensor from one layout to another. @since cuDNN 9.0.0
    OperationReshape =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR as _,
    ///< Expands a band matrix into a full matrix representation. @since cuDNN 9.10.0
    OperationExpandBandMatrix =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_EXPAND_BAND_MATRIX_DESCRIPTOR as _,
    ///< Contracts a full matrix into a band matrix representation. @since cuDNN 9.10.0
    OperationContractBandMatrix =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_CONTRACT_BAND_MATRIX_DESCRIPTOR as _,
    ///< RNG config: distribution type (Bernoulli/uniform/normal) and parameters. @since cuDNN 9.0.0
    Rng = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_RNG_DESCRIPTOR as _,
    ///< Generates a tensor of random numbers from a specified distribution. @since cuDNN 9.0.0
    OperationRng = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR as _,
    ///< Caches compiled kernels to speed up plan finalization for dynamic-shape graphs. @since cuDNN 9.4.0
    KernelCache = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_KERNEL_CACHE_DESCRIPTOR as _,
    ///< Reconstructs K/V-cache pages in fused flash attention forward graphs. @since cuDNN 9.4.0
    OperationPagedCacheLoad =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR as _,
    ///< Block-scale quantization: converts float tensors to block-scaled format. @since cuDNN 9.7.0
    OperationBlockScaleQuantize = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR as _,
    ///< Block-scale dequantization: converts block-scaled tensors back to float. @since cuDNN 9.7.0
    OperationBlockScaleDequantize = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR as _,
    ///< CUDA device properties: device ID, handle, JSON representation. @since cuDNN 9.8.0
    DeviceProp = sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_DEVICEPROP_DESCRIPTOR as _,
    ///< Mixture-of-Experts grouped matmul with token routing. @since cuDNN 9.15.0
    OperationMoeGroupedMatmul =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR as _,
    ///< Scaled dot-product attention backward: computes dQ, dK, dV. @since cuDNN 9.17.0
    OperationSdpaBackward =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR as _,
    ///< Generates a diagonal band attention mask. @since cuDNN 9.20.0
    OperationDiagonalBandMask =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_DIAGONAL_BAND_MASK_DESCRIPTOR as _,
    ///< Softmax operation with optional statistics output. @since cuDNN 9.20.0
    OperationSoftmax =
        sys::cudnnBackendDescriptorType_t::CUDNN_BACKEND_OPERATION_SOFTMAX_DESCRIPTOR as _,
}

impl_enum_conversion!(sys::cudnnBackendDescriptorType_t, BackendDescriptorType);

impl_enum_display!(BackendDescriptorType, {
    Self::OperationMoeGroupedMatmulBackward => "CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR",
    Self::OperationRopeForward => "CUDNN_BACKEND_OPERATION_ROPE_FWD_DESCRIPTOR",
    Self::OperationRopeBackward => "CUDNN_BACKEND_OPERATION_ROPE_BWD_DESCRIPTOR",

    Self::OperationSlice => "CUDNN_BACKEND_OPERATION_SLICE_DESCRIPTOR",
    Self::OperationTranspose => "CUDNN_BACKEND_OPERATION_TRANSPOSE_DESCRIPTOR",
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

/// Owns a native cuDNN backend descriptor and tracks its type and finalization state.
///
/// Attributes are set before finalization and read afterward. Raw pointer and embedded-resource attributes retain their own validity and lifetime requirements.
#[derive(Debug)]
pub struct BackendDescriptor {
    handle: sys::cudnnBackendDescriptor_t,
    finalized: bool,
    descriptor_type: BackendDescriptorType,
}

// SAFETY: backend descriptors expose mutation through &mut Self only. Shared
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
        // SAFETY: `cudnnBackendCreateDescriptor` receives valid output locations.
        unsafe {
            try_ffi!(sys::cudnnBackendCreateDescriptor(
                descriptor_type.into(),
                &raw mut handle
            ))?;
        }
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            handle,
            finalized: false,
            descriptor_type,
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
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is null.
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
        })
    }

    /// Returns the raw cuDNN backend descriptor.
    ///
    /// The returned handle must not be destroyed by the caller. Mutating it
    /// directly can violate this wrapper's finalized-state tracking.
    #[must_use]
    pub fn as_raw(&self) -> sys::cudnnBackendDescriptor_t {
        self.handle
    }

    /// Releases ownership and returns the raw cuDNN backend descriptor handle.
    ///
    /// The caller becomes responsible for destroying the handle.
    #[must_use]
    pub fn into_raw(mut self) -> sys::cudnnBackendDescriptor_t {
        replace(&mut self.handle, ptr::null_mut())
    }

    /// Checks if the descriptor has been finalized.
    #[must_use]
    pub fn is_finalized(&self) -> bool {
        self.finalized
    }

    /// Returns the type of the descriptor.
    #[must_use]
    pub fn descriptor_type(&self) -> BackendDescriptorType {
        self.descriptor_type
    }

    /// Returns whether the stored native descriptor handle is null.
    #[must_use]
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
    ///
    /// Also returns [`Error::DescriptorAlreadyFinalized`] if this descriptor was already finalized.
    pub fn finalize(&mut self) -> Result<()> {
        if self.finalized {
            return Err(Error::DescriptorAlreadyFinalized);
        }
        // SAFETY: `cudnnBackendFinalize` receives live handles owned by the wrappers.
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

    /// Sets a native attribute with a caller-established payload ABI.
    ///
    /// # Safety
    /// `values` must point to `element_count` valid native payloads of `ty`, with
    /// the required size and alignment. Pointees and embedded handles must satisfy
    /// the attribute's validity and retention requirements for this descriptor,
    /// derived descriptors, queued work and replay. Preserve exclusive native mutation.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor is already finalized, `element_count` does not fit the native signed integer type, or cuDNN rejects the attribute name, payload type, count, or assignment.
    pub unsafe fn set_attribute_unchecked(
        &mut self,
        name: BackendAttributeName,
        ty: BackendAttributeType,
        element_count: usize,
        values: *const (),
    ) -> Result<()> {
        self.set_attribute_raw(
            name,
            ty,
            to_i64(element_count, "backend attribute count")?,
            values,
        )
    }

    /// Reads native attribute payloads into caller-owned host storage.
    /// Returns cuDNN's reported attribute count, which may exceed `capacity`.
    /// cuDNN writes at most `capacity` elements; zero capacity queries the full size only.
    ///
    /// # Safety
    /// `values` must describe writable storage for `capacity` payloads of `ty`,
    /// with the native size and alignment. No alias may access it during this call.
    /// Validate restricted representations before reading Rust values. Any native
    /// descriptor handles required by the queried attribute must be valid on entry.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor is not finalized, `capacity` does not fit the native signed integer type, cuDNN rejects or cannot query the attribute, or its reported count does not fit `usize`.
    pub unsafe fn attribute_unchecked(
        &self,
        name: BackendAttributeName,
        ty: BackendAttributeType,
        capacity: usize,
        values: *mut (),
    ) -> Result<usize> {
        self.check_finalized()?;
        let mut count = 0;
        // SAFETY: `cudnnBackendGetAttribute` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cudnnBackendGetAttribute(
                self.handle,
                name.into(),
                ty.into(),
                to_i64(capacity, "backend attribute count")?,
                &raw mut count,
                values.cast(),
            ))?;
        }
        to_usize(count, "backend attribute count")
    }

    fn set_attribute_raw(
        &mut self,
        name: BackendAttributeName,
        ty: BackendAttributeType,
        element_count: i64,
        values: *const (),
    ) -> Result<()> {
        self.check_modifiable()?;
        // SAFETY: `cudnnBackendSetAttribute` receives live handles owned by the wrappers.
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
        self.set_attribute_raw(
            name,
            ty,
            to_i64(values.len(), value_name)?,
            values.as_ptr().cast(),
        )
    }

    fn attribute_raw<T: Default>(
        &self,
        name: BackendAttributeName,
        ty: BackendAttributeType,
    ) -> Result<T> {
        self.check_finalized()?;
        let mut value = T::default();
        let mut count_written: i64 = 0;
        // SAFETY: `cudnnBackendGetAttribute` receives live handles owned by the wrappers and valid output locations.
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

    /// Queries an attribute's total element count without allocating or retrieving its payload.
    /// The supplied category must match the attribute's native type.
    ///
    /// # Errors
    /// Returns an error for an unfinalized descriptor, a rejected native query or an unrepresentable count.
    pub fn attribute_count(
        &self,
        name: BackendAttributeName,
        ty: BackendAttributeType,
    ) -> Result<usize> {
        self.check_finalized()?;
        let mut count = 0;
        // SAFETY: `cudnnBackendGetAttribute` receives live handles owned by the wrappers and valid output locations.
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
        usize::try_from(count).map_err(|_| Error::InvalidBackendDescriptorCount {
            name: name.to_string().into(),
        })
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

        let mut values = vec![initial_value; count];
        let mut retries = 0;
        loop {
            let mut count_written = 0;
            // SAFETY: the buffer holds the declared number of correctly typed attribute elements.
            // On a size race, cuDNN reports the new required count without writing beyond that buffer.
            let status = unsafe {
                sys::cudnnBackendGetAttribute(
                    self.handle,
                    name.into(),
                    ty.into(),
                    to_i64(values.len(), value_name)?,
                    &raw mut count_written,
                    values.as_mut_ptr().cast(),
                )
            };
            if matches!(
                Status::from(status),
                Status::Success | Status::BadParamSizeInsufficient
            ) && retries < 7
                && let Ok(required) = usize::try_from(count_written)
                && required > values.len()
            {
                values.resize(required, initial_value);
                retries += 1;
                continue;
            }
            try_ffi!(status)?;
            let written = usize::try_from(count_written).map_err(|_| {
                Error::InvalidBackendDescriptorCount {
                    name: value_name.into(),
                }
            })?;
            if written > values.len() {
                return Err(Error::InvalidBackendDescriptorCount {
                    name: value_name.into(),
                });
            }
            values.truncate(written);
            return Ok(values);
        }
    }
}

impl Drop for BackendDescriptor {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        // SAFETY: `cudnnBackendDestroyDescriptor` receives live handles owned by the wrappers.
        unsafe {
            let _ = try_ffi!(sys::cudnnBackendDestroyDescriptor(self.handle));
        }
    }
}

/// Executes the given engine configuration plan on the [`VariantPack`](crate::backend::execution::VariantPack)
/// and the finalized [`ExecutionPlan`](crate::backend::execution::ExecutionPlan) on the data.
/// The data and the working space are encapsulated in the [`VariantPack`](crate::backend::execution::VariantPack).
///
/// # Errors
///
/// Returns an error if `plan` or `varpack` is not finalized or has the wrong
/// descriptor type, if required data pointers are invalid, if plan execution
/// fails, or if cuDNN reports an internal failure.
// TODO: better types?
///
/// # Safety
/// All native accesses require live plan code and valid, initialized inputs or writable outputs,
/// with compatible layouts, permitted aliases and established producer dependencies.
/// Keep device storage and code alive through completion and every captured replay.
/// Host argument storage remains valid while cuDNN consumes it. Updates retain prior
/// storage through executable snapshots and pending launches that still use it.
pub(crate) unsafe fn backend_execute(
    context: &Context,
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

    context.bind()?;
    // SAFETY: `cudnnBackendExecute` receives live handles owned by the wrappers.
    unsafe {
        try_ffi!(sys::cudnnBackendExecute(
            context.as_raw(),
            plan.as_raw(),
            varpack.as_raw()
        ))?;
    }

    Ok(())
}
