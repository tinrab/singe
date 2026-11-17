//! Owned backend descriptors with typed attribute access and explicit finalization.

mod attributes;
mod cuda_graph;

use std::{mem::replace, ptr};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_display, report_cleanup_failure};
use singe_cudnn_sys as sys;

use crate::{
    backend::attribute::{BackendAttributeName, BackendAttributeType},
    context::Context,
    error::{Error, Result, Status},
    try_ffi,
    utility::{to_i64, to_usize},
};

singe_core::native_enum! {
    /// Type tag for backend descriptors.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[non_exhaustive]
    #[repr(u32)]
    pub enum BackendDescriptorType: sys::cudnnBackendDescriptorType_t {
        /// Backward `MoE` grouped matmul.
        OperationMoeGroupedMatmulBackward = CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR,
        /// Rotary Position Embedding (`RoPE`) forward.
        OperationRopeForward = CUDNN_BACKEND_OPERATION_ROPE_FWD_DESCRIPTOR,
        /// Rotary Position Embedding (`RoPE`) backward.
        OperationRopeBackward = CUDNN_BACKEND_OPERATION_ROPE_BWD_DESCRIPTOR,
        /// Slice operation.
        OperationSlice = CUDNN_BACKEND_OPERATION_SLICE_DESCRIPTOR,
        /// Transpose operation.
        OperationTranspose = CUDNN_BACKEND_OPERATION_TRANSPOSE_DESCRIPTOR,
        /// Pointwise op config: mode, math precision, activation params.
        Pointwise = CUDNN_BACKEND_POINTWISE_DESCRIPTOR,
        /// Convolution config: compute type, mode, dilation, stride, padding.
        Convolution = CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR,
        /// Engine (kernel grouping) to compute an operation graph.
        Engine = CUDNN_BACKEND_ENGINE_DESCRIPTOR,
        /// Engine configuration: engine descriptor plus knob choices.
        EngineCfg = CUDNN_BACKEND_ENGINECFG_DESCRIPTOR,
        /// Engine configurations ranked by performance via cuDNN heuristics.
        EngineHeur = CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR,
        /// Finalized execution plan: engine config, workspace size, optional kernel cache.
        ExecutionPlan = CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR,
        /// Read-only info about a reusable execution intermediate.
        IntermediateInfo = CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR,
        /// Type and value of an engine performance tuning knob.
        KnobChoice = CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR,
        /// Read-only info about an engine knob: type, min/max, and stride.
        KnobInfo = CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR,
        /// Read-only info on the preferred memory layout for a tensor.
        LayoutInfo = CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR,
        /// Forward convolution: y = alpha * conv(w, x) + beta * y.
        OperationConvolutionForward = CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
        /// Backward filter convolution: computes dw from x and dy.
        OperationConvolutionBackwardFilter = CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
        /// Backward data convolution: computes dx from w and dy.
        OperationConvolutionBackwardData = CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
        /// Pointwise operation: Y = op(alpha1*X) or Y = op(alpha1*X, alpha2*B).
        OperationPointwise = CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
        /// Generates per-channel statistics (sum and sum-of-squares).
        OperationGenStats = CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR,
        /// Operation graph: a DAG of operations connected by virtual tensors.
        OperationGraph = CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR,
        /// Binds device pointers to non-virtual tensors, workspace, and intermediates.
        VariantPack = CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR,
        /// Tensor: data type, dimensions, strides, alignment, unique ID, virtual flag.
        Tensor = CUDNN_BACKEND_TENSOR_DESCRIPTOR,
        /// Matrix multiply config: compute type and padding value.
        Matmul = CUDNN_BACKEND_MATMUL_DESCRIPTOR,
        /// Matrix multiplication operation: C = A * B with optional overrides.
        OperationMatmul = CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR,
        /// BN finalize: computes running stats, saved mean/invstd, eq scale/bias.
        OperationBnFinalizeStatistics = CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR,
        /// Reduction config: operator type (ADD/MUL/MIN/MAX/etc.) and compute type.
        Reduction = CUDNN_BACKEND_REDUCTION_DESCRIPTOR,
        /// Reduces input tensor values along one or more dimensions.
        OperationReduction = CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR,
        /// BN backward weights: computes dScale, dBias, and equivalent gradients.
        OperationBnBwdWeights = CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR,
        /// Resample config: mode, spatial dims, padding, strides, window dims.
        Resample = CUDNN_BACKEND_RESAMPLE_DESCRIPTOR,
        /// Forward resampling (pooling/interpolation) with alpha/beta scaling.
        OperationResampleFwd = CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR,
        /// Backward resampling: computes dx from dy with alpha/beta scaling.
        OperationResampleBwd = CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR,
        /// Concatenates multiple tensors along a given axis.
        OperationConcat = CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR,
        /// Updates or waits on a flag variable for inter-graph signaling.
        OperationSignal = CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR,
        /// Forward normalization (layer/instance/batch/RMS) with optional running stats.
        OperationNormForward = CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR,
        /// Backward normalization: computes dX, dScale, dBias from dY.
        OperationNormBackward = CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR,
        /// Scaled dot-product attention forward (fused flash attention).
        OperationSdpaForward = CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR,
        /// Reshapes a tensor from one layout to another.
        OperationReshape = CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR,
        /// Expands a band matrix into a full matrix representation.
        OperationExpandBandMatrix = CUDNN_BACKEND_OPERATION_EXPAND_BAND_MATRIX_DESCRIPTOR,
        /// Contracts a full matrix into a band matrix representation.
        OperationContractBandMatrix = CUDNN_BACKEND_OPERATION_CONTRACT_BAND_MATRIX_DESCRIPTOR,
        /// RNG config: distribution type (Bernoulli/uniform/normal) and parameters.
        Rng = CUDNN_BACKEND_RNG_DESCRIPTOR,
        /// Generates a tensor of random numbers from a specified distribution.
        OperationRng = CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR,
        /// Caches compiled kernels to speed up plan finalization for dynamic-shape graphs.
        KernelCache = CUDNN_BACKEND_KERNEL_CACHE_DESCRIPTOR,
        /// Reconstructs K/V-cache pages in fused flash attention forward graphs.
        OperationPagedCacheLoad = CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR,
        /// Block-scale quantization: converts float tensors to block-scaled format.
        OperationBlockScaleQuantize = CUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR,
        /// Block-scale dequantization: converts block-scaled tensors back to float.
        OperationBlockScaleDequantize = CUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR,
        /// CUDA device properties: device ID, handle, JSON representation.
        DeviceProp = CUDNN_BACKEND_DEVICEPROP_DESCRIPTOR,
        /// Mixture-of-Experts grouped matmul with token routing.
        OperationMoeGroupedMatmul = CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR,
        /// Scaled dot-product attention backward: computes dQ, dK, dV.
        OperationSdpaBackward = CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR,
        /// Generates a diagonal band attention mask.
        OperationDiagonalBandMask = CUDNN_BACKEND_OPERATION_DIAGONAL_BAND_MASK_DESCRIPTOR,
        /// Softmax operation with optional statistics output.
        OperationSoftmax = CUDNN_BACKEND_OPERATION_SOFTMAX_DESCRIPTOR,
    }
}

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
/// Attributes are set before finalization and read afterward.
/// Raw pointer and embedded-resource attributes retain their own validity and lifetime requirements.
///
/// The descriptor is `Sync`: cuDNN documents its functions as callable from several host threads as long as they do not share a cuDNN handle, and shared references only read finalized descriptors.
/// Execution plans, engine configurations, and device properties share descriptors through `Arc`, which needs `Sync` to cross threads.
#[derive(Debug)]
pub struct BackendDescriptor {
    handle: sys::cudnnBackendDescriptor_t,
    finalized: bool,
    descriptor_type: BackendDescriptorType,
}

// SAFETY: the wrapper exclusively owns its descriptor, and cuDNN descriptors have no thread affinity, so the owner may move between threads.
unsafe impl Send for BackendDescriptor {}

// SAFETY: every native mutation (`cudnnBackendSetAttribute`, `cudnnBackendFinalize`) takes `&mut self`.
// Shared references only pass the finalized descriptor to `cudnnBackendGetAttribute` and `cudnnBackendExecute`, and cuDNN documents its functions as thread-safe for threads that do not share a cuDNN handle; `Context` is not `Sync`, so each thread executes with its own handle.
unsafe impl Sync for BackendDescriptor {}

/// A transferred native backend descriptor with its type and finalization state.
#[derive(Debug)]
pub struct RawBackendDescriptor {
    /// The owned native backend descriptor handle.
    pub handle: sys::cudnnBackendDescriptor_t,
    /// The type `handle` was created with.
    pub descriptor_type: BackendDescriptorType,
    /// Whether `handle` has been finalized.
    pub finalized: bool,
}

// SAFETY: the record carries exclusive ownership of a descriptor with no thread affinity.
unsafe impl Send for RawBackendDescriptor {}

impl BackendDescriptor {
    /// Creates a backend descriptor of the requested type.
    ///
    /// The returned descriptor owns the cuDNN backend descriptor handle and destroys it when dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDNN does not return a valid descriptor handle, if host memory allocation fails, or if cuDNN does not support creating the requested descriptor type.
    pub fn create(descriptor_type: BackendDescriptorType) -> Result<Self> {
        let mut handle = ptr::null_mut();
        // SAFETY: `handle` is a local the call writes, and `descriptor_type` converts to a valid native enum.
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

    pub(crate) fn leak(&mut self) {
        self.handle = ptr::null_mut();
    }

    /// Takes ownership of a raw cuDNN backend descriptor handle.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be a valid [`sys::cudnnBackendDescriptor_t`] created by cuDNN.
    /// The returned wrapper takes ownership and destroys the descriptor on drop; no other owner may destroy or keep using it.
    /// `raw.finalized` and `raw.descriptor_type` must accurately describe the descriptor state.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw.handle` is null.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "the raw value transfers ownership of its handle even though its fields are `Copy`"
    )]
    pub unsafe fn from_raw(raw: RawBackendDescriptor) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: raw.handle,
            finalized: raw.finalized,
            descriptor_type: raw.descriptor_type,
        })
    }

    /// Returns the raw cuDNN backend descriptor.
    ///
    /// The returned handle must not be destroyed by the caller.
    /// Mutating it directly can violate this wrapper's finalized-state tracking.
    #[must_use]
    pub fn as_raw(&self) -> sys::cudnnBackendDescriptor_t {
        self.handle
    }

    /// Releases ownership and returns the raw cuDNN backend descriptor handle with its type and finalization state.
    ///
    /// The caller becomes responsible for destroying the handle.
    #[must_use]
    pub fn into_raw(mut self) -> RawBackendDescriptor {
        RawBackendDescriptor {
            handle: replace(&mut self.handle, ptr::null_mut()),
            descriptor_type: self.descriptor_type,
            finalized: self.finalized,
        }
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
    /// Finalization checks the attributes set on the descriptor and marks the descriptor as finalized.
    /// Once finalized, attributes can be retrieved but no longer modified.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor was already finalized, its attributes are invalid or unsupported by the current cuDNN version, or cuDNN reports an internal failure.
    pub fn finalize(&mut self) -> Result<()> {
        if self.finalized {
            return Err(Error::DescriptorAlreadyFinalized);
        }

        // SAFETY: `self.handle` is the live descriptor this wrapper owns, and `&mut self` makes the finalization exclusive.
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
    ///
    /// `values` must point to `element_count` valid native payloads of `ty`, with the required size and alignment.
    /// Pointees and embedded handles must satisfy the attribute's validity and retention requirements for this descriptor, derived descriptors, queued work and replay.
    /// Preserve exclusive native mutation.
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
    ///
    /// `values` must describe writable storage for `capacity` payloads of `ty`, with the native size and alignment.
    /// No alias may access it during this call.
    /// Validate restricted representations before reading Rust values.
    /// Any native descriptor handles required by the queried attribute must be valid on entry.
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
        // SAFETY: `self.handle` is the owned, finalized descriptor and `count` is a writable local.
        // The caller guarantees `values` holds `capacity` writable payloads of `ty`.
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
        // SAFETY: `self.handle` is the owned, unfinalized descriptor, and `&mut self` makes the update exclusive.
        // Callers pass `values` pointing to `element_count` payloads of `ty`: a borrowed slice, or the payload that `set_attribute_unchecked`'s caller vouches for.
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
            return Err(Error::Empty {
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
        // SAFETY: `self.handle` is the owned, finalized descriptor, `count_written` is a writable local, and `value` has room for the one element requested.
        // Callers pick `T` as the native representation of `ty`.
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
    ///
    /// Returns an error if the descriptor is not finalized, cuDNN rejects the query, or the count does not fit `usize`.
    pub fn attribute_count(
        &self,
        name: BackendAttributeName,
        ty: BackendAttributeType,
    ) -> Result<usize> {
        self.check_finalized()?;
        let mut count = 0;
        // SAFETY: `self.handle` is the owned, finalized descriptor and `count` is a writable local.
        // A zero capacity with a null payload asks cuDNN for the count only.
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

        // SAFETY: `self.handle` is non-null and owned by this wrapper, and nothing uses it after drop.
        if let Err(error) = unsafe { try_ffi!(sys::cudnnBackendDestroyDescriptor(self.handle)) } {
            report_cleanup_failure("destroy", "cudnn backend descriptor", &error);
        }
    }
}

/// Executes the given engine configuration plan on the [`VariantPack`](crate::backend::execution::VariantPack) and the finalized [`ExecutionPlan`](crate::backend::execution::ExecutionPlan) on the data.
/// The data and the working space are encapsulated in the [`VariantPack`](crate::backend::execution::VariantPack).
///
/// # Safety
///
/// All native accesses require live plan code and valid, initialized inputs or writable outputs, with compatible layouts, permitted aliases and established producer dependencies.
/// Keep device storage and code alive through completion and every captured replay.
/// Host argument storage remains valid while cuDNN consumes it.
/// Updates retain prior storage through executable snapshots and pending launches that still use it.
///
/// # Errors
///
/// Returns an error if `plan` or `varpack` is not finalized or has the wrong descriptor type, if required data pointers are invalid, if plan execution fails, or if cuDNN reports an internal failure.
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
    // SAFETY: `plan` and `varpack` are finalized descriptors of the checked types, and `context` is bound.
    // The caller keeps the plan code and variant-pack storage alive and ordered through completion, per `# Safety`.
    unsafe {
        try_ffi!(sys::cudnnBackendExecute(
            context.as_raw(),
            plan.as_raw(),
            varpack.as_raw()
        ))?;
    }

    Ok(())
}
