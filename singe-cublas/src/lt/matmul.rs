use core::ffi;
use std::{
    ffi::c_void,
    mem,
    mem::{MaybeUninit, size_of},
    ptr::{self, from_ref},
};

use singe_core::report_cleanup_failure;
use singe_cublas_sys as sys;
use singe_cuda::{
    data_type::DataType,
    error::Error as CudaError,
    memory::DeviceMemory,
    stream::{AsStream, Stream},
    view::{
        AnyBitPatternDeviceRepr, DeviceInputView, DeviceOutput, DeviceOutputView, DeviceRange,
        DeviceRegionProvider, DeviceRepr,
    },
};
use singe_cuda_sys::library_types::cudaDataType_t;

use crate::{
    error::{Error, Result, Status, try_ffi},
    lt::{
        context::Context,
        descriptor::{EmulationDescriptor, MatrixLayout, MatrixTransformDescriptor},
        types::{
            BatchInvariance, Epilogue, MatmulAlgorithmCapAttribute, MatmulAlgorithmConfigAttribute,
            MatmulDescriptorAttribute, MatmulPreferenceAttribute, MatrixScale, PointerMode,
            SearchMode,
        },
        utility::{attribute_enum, read_attribute, set_attribute},
    },
    scalar::Scalar,
    types::{ComputeType, FillMode, Operation},
    utility::{ensure_exact_size, to_i32, to_size_t, to_u64, to_usize},
};

/// Describes the configuration for a cuBLASLt matrix multiplication operation.
///
/// Owns a cuBLASLt matmul descriptor handle.
/// Configure transpose modes, epilogue, scaling, and other attributes through the setter methods.
#[derive(Debug)]
pub struct MatmulDescriptor {
    raw: sys::cublasLtMatmulDesc_t,
}

// SAFETY: the descriptor is host memory owned by this value; cuBLASLt allows using and destroying it from any thread, one call at a time.
unsafe impl Send for MatmulDescriptor {}

/// Stores algorithm-search preferences for cuBLASLt matmul heuristic queries.
///
/// Owns a cuBLASLt matmul preference handle.
/// Use setter methods to configure workspace limits, search mode, alignment requirements, and other tuning parameters.
#[derive(Debug)]
pub struct MatmulPreference {
    raw: sys::cublasLtMatmulPreference_t,
}

// SAFETY: the descriptor is host memory owned by this value; cuBLASLt allows using and destroying it from any thread, one call at a time.
unsafe impl Send for MatmulPreference {}

/// A configured cuBLASLt matrix multiplication algorithm.
#[derive(Debug, Clone, Copy)]
pub struct MatmulAlgorithm {
    raw: sys::cublasLtMatmulAlgo_t,
}

/// An algorithm candidate and its estimated runtime resource requirements.
#[derive(Debug, Clone)]
pub struct MatmulHeuristicResult {
    /// The algorithm configuration returned by the heuristic or compatibility check.
    pub algorithm: MatmulAlgorithm,
    /// The workspace required by this algorithm, in bytes.
    pub workspace_size: usize,
    /// The status for this candidate; the other fields are valid when this is [`Status::Success`].
    pub status: Status,
    /// Estimated device utilization; `1.0` indicates that the kernel fully occupies the GPU.
    pub waves_count: f32,
}

impl MatmulDescriptor {
    /// Requests batch-invariant matrix multiplication results.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the batch-invariance policy.
    pub fn set_batch_invariance(&mut self, policy: BatchInvariance) -> Result<()> {
        // SAFETY: `MatmulDescriptorAttribute::BatchInvarianceFlags` stores one `cublasLtMatmulBatchInvarianceFlags_t`, the type passed here.
        unsafe {
            self.set_attribute(
                MatmulDescriptorAttribute::BatchInvarianceFlags,
                &sys::cublasLtMatmulBatchInvarianceFlags_t::from(policy),
            )
        }
    }

    /// Returns the experimental batch-invariance policy.
    ///
    /// # Errors
    ///
    /// Returns an error if the attribute query fails, its size differs from `u32`, or its value is not a recognized [`BatchInvariance`] policy.
    pub fn batch_invariance(&self) -> Result<BatchInvariance> {
        let value = self.attribute::<u32>(MatmulDescriptorAttribute::BatchInvarianceFlags)?;
        attribute_enum(value, "MatmulDescriptorAttribute::BatchInvarianceFlags")
    }

    /// Creates a matrix multiplication descriptor.
    ///
    /// The descriptor owns its cuBLASLt handle and destroys it when dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt cannot allocate the descriptor or if it does not return a valid handle.
    pub fn create(compute_type: ComputeType, scale_type: DataType) -> Result<Self> {
        let mut raw = ptr::null_mut();
        // SAFETY: `cublasLtMatmulDescCreate` writes the new descriptor to `raw`, a live local, and takes its other arguments by value.
        unsafe {
            try_ffi!(sys::cublasLtMatmulDescCreate(
                &raw mut raw,
                compute_type.into(),
                scale_type.into(),
            ))?;
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self { raw })
    }

    /// Sets a matmul descriptor attribute.
    ///
    /// # Safety
    ///
    /// `T` must match the selected attribute's native representation, including its validity and initialized-byte requirements.
    /// Retain pointer targets for every native use.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute or the value size does not match the attribute storage expected by cuBLASLt.
    pub unsafe fn set_attribute<T>(
        &mut self,
        attr: MatmulDescriptorAttribute,
        value: &T,
    ) -> Result<()> {
        set_attribute(
            // SAFETY: `self.raw` is the live descriptor this value owns, and `set_attribute` passes `value` with its exact byte size.
            |value, size| unsafe {
                sys::cublasLtMatmulDescSetAttribute(self.raw, attr.into(), value, size)
            },
            from_ref::<T>(value).cast(),
            size_of::<T>(),
        )
    }

    /// Returns a matmul descriptor attribute value.
    ///
    /// `T` must accept every bit pattern; enum attributes are read as integers and validated.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the query or if the returned attribute size does not match `T`.
    pub fn attribute<T: AnyBitPatternDeviceRepr>(
        &self,
        attr: MatmulDescriptorAttribute,
    ) -> Result<T> {
        let () = T::LAYOUT;
        let mut value = MaybeUninit::<T>::uninit();
        let written = read_attribute(
            // SAFETY: `self.raw` is the live descriptor this value owns, and `read_attribute` passes `value` with its byte size and a live `written` location.
            |value, size, written| unsafe {
                sys::cublasLtMatmulDescGetAttribute(self.raw, attr.into(), value, size, written)
            },
            value.as_mut_ptr().cast(),
            size_of::<T>(),
            "matmul descriptor attribute",
        )?;
        ensure_exact_size("matmul descriptor attribute", written, size_of::<T>())?;
        // SAFETY: the library wrote exactly `size_of::<T>()` bytes, checked above, initializing `value`.
        Ok(unsafe { value.assume_init() })
    }

    /// Sets the pointer mode for alpha and beta scalars.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_pointer_mode(&mut self, pointer_mode: PointerMode) -> Result<()> {
        // SAFETY: `MatmulDescriptorAttribute::PointerMode` stores one `cublasLtPointerMode_t`, the type passed here.
        unsafe {
            self.set_attribute(
                MatmulDescriptorAttribute::PointerMode,
                &sys::cublasLtPointerMode_t::from(pointer_mode),
            )
        }
    }

    /// Returns the pointer mode for alpha and beta scalars.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt cannot report the attribute.
    pub fn pointer_mode(&self) -> Result<PointerMode> {
        let value = self.attribute::<u32>(MatmulDescriptorAttribute::PointerMode)?;
        attribute_enum(value, "MatmulDescriptorAttribute::PointerMode")
    }

    /// Sets the transpose mode applied to input matrix A.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_transpose_a(&mut self, operation: Operation) -> Result<()> {
        // SAFETY: `MatmulDescriptorAttribute::TransposeA` stores one `cublasOperation_t`, the type passed here.
        unsafe {
            self.set_attribute(
                MatmulDescriptorAttribute::TransposeA,
                &sys::cublasOperation_t::from(operation),
            )
        }
    }

    /// Returns the transpose mode for input matrix A.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt cannot report the attribute.
    pub fn transpose_a(&self) -> Result<Operation> {
        let value = self.attribute::<u32>(MatmulDescriptorAttribute::TransposeA)?;
        attribute_enum(value, "MatmulDescriptorAttribute::TransposeA")
    }

    /// Sets the transpose mode applied to input matrix B.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_transpose_b(&mut self, operation: Operation) -> Result<()> {
        // SAFETY: `MatmulDescriptorAttribute::TransposeB` stores one `cublasOperation_t`, the type passed here.
        unsafe {
            self.set_attribute(
                MatmulDescriptorAttribute::TransposeB,
                &sys::cublasOperation_t::from(operation),
            )
        }
    }

    /// Returns the transpose mode for input matrix B.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt cannot report the attribute.
    pub fn transpose_b(&self) -> Result<Operation> {
        let value = self.attribute::<u32>(MatmulDescriptorAttribute::TransposeB)?;
        attribute_enum(value, "MatmulDescriptorAttribute::TransposeB")
    }

    /// Sets the fill mode for triangular matrix multiplication.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_fill_mode(&mut self, fill_mode: FillMode) -> Result<()> {
        // SAFETY: `MatmulDescriptorAttribute::FillMode` stores one `cublasFillMode_t`, the type passed here.
        unsafe {
            self.set_attribute(
                MatmulDescriptorAttribute::FillMode,
                &sys::cublasFillMode_t::from(fill_mode),
            )
        }
    }

    /// Returns the fill mode.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt cannot report the attribute.
    pub fn fill_mode(&self) -> Result<FillMode> {
        let value = self.attribute::<u32>(MatmulDescriptorAttribute::FillMode)?;
        attribute_enum(value, "MatmulDescriptorAttribute::FillMode")
    }

    /// Sets the epilogue (post-processing) applied after the matrix multiply.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_epilogue(&mut self, epilogue: Epilogue) -> Result<()> {
        // SAFETY: `MatmulDescriptorAttribute::Epilogue` stores one `cublasLtEpilogue_t`, the type passed here.
        unsafe {
            self.set_attribute(
                MatmulDescriptorAttribute::Epilogue,
                &sys::cublasLtEpilogue_t::from(epilogue),
            )
        }
    }

    /// Returns the epilogue configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt cannot report the attribute.
    pub fn epilogue(&self) -> Result<Epilogue> {
        let value = self.attribute::<u32>(MatmulDescriptorAttribute::Epilogue)?;
        attribute_enum(value, "MatmulDescriptorAttribute::Epilogue")
    }

    /// Sets the scaling mode for matrix A.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_a_scale_mode(&mut self, scale_mode: MatrixScale) -> Result<()> {
        // SAFETY: `MatmulDescriptorAttribute::AScaleMode` stores one `cublasLtMatmulMatrixScale_t`, the type passed here.
        unsafe {
            self.set_attribute(
                MatmulDescriptorAttribute::AScaleMode,
                &sys::cublasLtMatmulMatrixScale_t::from(scale_mode),
            )
        }
    }

    /// Sets the scaling mode for matrix B.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_b_scale_mode(&mut self, scale_mode: MatrixScale) -> Result<()> {
        // SAFETY: `MatmulDescriptorAttribute::BScaleMode` stores one `cublasLtMatmulMatrixScale_t`, the type passed here.
        unsafe {
            self.set_attribute(
                MatmulDescriptorAttribute::BScaleMode,
                &sys::cublasLtMatmulMatrixScale_t::from(scale_mode),
            )
        }
    }

    /// Sets the scaling mode for matrix C.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_c_scale_mode(&mut self, scale_mode: MatrixScale) -> Result<()> {
        // SAFETY: `MatmulDescriptorAttribute::CScaleMode` stores one `cublasLtMatmulMatrixScale_t`, the type passed here.
        unsafe {
            self.set_attribute(
                MatmulDescriptorAttribute::CScaleMode,
                &sys::cublasLtMatmulMatrixScale_t::from(scale_mode),
            )
        }
    }

    /// Sets the scaling mode for matrix D.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_d_scale_mode(&mut self, scale_mode: MatrixScale) -> Result<()> {
        // SAFETY: `MatmulDescriptorAttribute::DScaleMode` stores one `cublasLtMatmulMatrixScale_t`, the type passed here.
        unsafe {
            self.set_attribute(
                MatmulDescriptorAttribute::DScaleMode,
                &sys::cublasLtMatmulMatrixScale_t::from(scale_mode),
            )
        }
    }

    /// Enables or disables FP8 fast accumulation.
    ///
    /// Fast accumulation skips the periodic promotion of intermediate results to a higher precision, trading accuracy for speed.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_fast_accum(&mut self, enabled: bool) -> Result<()> {
        // SAFETY: `MatmulDescriptorAttribute::FastAccum` stores one `i8`, the type passed here.
        unsafe { self.set_attribute(MatmulDescriptorAttribute::FastAccum, &i8::from(enabled)) }
    }

    /// Sets the bias pointer for the bias epilogue.
    ///
    /// The descriptor stores the device address of `bias` without borrowing it.
    ///
    /// # Safety
    ///
    /// `bias` must stay live, hold the values the bias epilogue reads, and not be written by other work until every matmul using this descriptor completes, or until the pointer is replaced.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub unsafe fn set_bias_pointer<T: DeviceRepr>(
        &mut self,
        bias: &(impl DeviceRange<T> + ?Sized),
    ) -> Result<()> {
        let pointer = bias.as_device_ptr();
        // SAFETY: the attribute stores one device address, and the caller keeps its target valid for each matmul.
        unsafe { self.set_attribute(MatmulDescriptorAttribute::BiasPointer, &pointer) }
    }

    /// Sets the bias data type for bias epilogues.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_bias_data_type(&mut self, data_type: DataType) -> Result<()> {
        // SAFETY: `MatmulDescriptorAttribute::BiasDataType` stores one `cudaDataType_t`, the type passed here.
        unsafe {
            self.set_attribute(
                MatmulDescriptorAttribute::BiasDataType,
                &cudaDataType_t::from(data_type),
            )
        }
    }

    /// Sets the scale pointer for matrix A.
    ///
    /// The descriptor stores the device address of `scale` without borrowing it.
    ///
    /// # Safety
    ///
    /// `scale` must stay live, hold every scale factor that the matrix A scale mode and layout read, and not be written by other work until every matmul using this descriptor completes, or until the pointer is replaced.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub unsafe fn set_a_scale_pointer<T>(&mut self, scale: &DeviceMemory<T>) -> Result<()> {
        let pointer = scale.as_ptr();
        // SAFETY: the attribute stores one device address, and the caller keeps its target valid for each matmul.
        unsafe { self.set_attribute(MatmulDescriptorAttribute::AScalePointer, &pointer) }
    }

    /// Sets the scale pointer for matrix B.
    ///
    /// The descriptor stores the device address of `scale` without borrowing it.
    ///
    /// # Safety
    ///
    /// `scale` must stay live, hold every scale factor that the matrix B scale mode and layout read, and not be written by other work until every matmul using this descriptor completes, or until the pointer is replaced.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub unsafe fn set_b_scale_pointer<T>(&mut self, scale: &DeviceMemory<T>) -> Result<()> {
        let pointer = scale.as_ptr();
        // SAFETY: the attribute stores one device address, and the caller keeps its target valid for each matmul.
        unsafe { self.set_attribute(MatmulDescriptorAttribute::BScalePointer, &pointer) }
    }

    /// Sets the scale pointer for matrix C.
    ///
    /// The descriptor stores the device address of `scale` without borrowing it.
    ///
    /// # Safety
    ///
    /// `scale` must stay live, hold every scale factor that the matrix C scale mode and layout read, and not be written by other work until every matmul using this descriptor completes, or until the pointer is replaced.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub unsafe fn set_c_scale_pointer<T>(&mut self, scale: &DeviceMemory<T>) -> Result<()> {
        let pointer = scale.as_ptr();
        // SAFETY: the attribute stores one device address, and the caller keeps its target valid for each matmul.
        unsafe { self.set_attribute(MatmulDescriptorAttribute::CScalePointer, &pointer) }
    }

    /// Sets the scale pointer for matrix D.
    ///
    /// The descriptor stores the device address of `scale` without borrowing it.
    ///
    /// # Safety
    ///
    /// `scale` must stay live, hold every scale factor that the matrix D scale mode and layout read, and not be written by other work until every matmul using this descriptor completes, or until the pointer is replaced.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub unsafe fn set_d_scale_pointer<T>(&mut self, scale: &DeviceMemory<T>) -> Result<()> {
        let pointer = scale.as_ptr();
        // SAFETY: the attribute stores one device address, and the caller keeps its target valid for each matmul.
        unsafe { self.set_attribute(MatmulDescriptorAttribute::DScalePointer, &pointer) }
    }

    /// Sets the output scale pointer for matrix D.
    ///
    /// The descriptor stores the device address of `scale` without borrowing it, and matmuls write the output scale factors through it.
    ///
    /// # Safety
    ///
    /// `scale` must stay live, have room for every scale factor that the matrix D output scale mode and layout write, and not be accessed by other work until every matmul using this descriptor completes, or until the pointer is replaced.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub unsafe fn set_d_out_scale_pointer<T>(&mut self, scale: &mut DeviceMemory<T>) -> Result<()> {
        let pointer = scale.as_mut_ptr();
        // SAFETY: the attribute stores one device address, and the caller keeps its target valid for each matmul.
        unsafe { self.set_attribute(MatmulDescriptorAttribute::DOutScalePointer, &pointer) }
    }

    /// Sets the pointer for storing the absolute maximum value of D.
    ///
    /// The descriptor stores the device address of `amax` without borrowing it, and matmuls write one value of the compute type through it.
    ///
    /// # Safety
    ///
    /// `amax` must stay live, have room for one value of the descriptor's compute type, and not be accessed by other work until every matmul using this descriptor completes, or until the pointer is replaced.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub unsafe fn set_amax_d_pointer<T>(&mut self, amax: &mut DeviceMemory<T>) -> Result<()> {
        let pointer = amax.as_mut_ptr();
        // SAFETY: the attribute stores one device address, and the caller keeps its target valid for each matmul.
        unsafe { self.set_attribute(MatmulDescriptorAttribute::AmaxDPointer, &pointer) }
    }

    /// Sets the output scaling mode for matrix D.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_d_out_scale_mode(&mut self, scale_mode: MatrixScale) -> Result<()> {
        // SAFETY: `MatmulDescriptorAttribute::DOutScaleMode` stores one `cublasLtMatmulMatrixScale_t`, the type passed here.
        unsafe {
            self.set_attribute(
                MatmulDescriptorAttribute::DOutScaleMode,
                &sys::cublasLtMatmulMatrixScale_t::from(scale_mode),
            )
        }
    }

    /// Sets the batch stride for per-batch alpha values.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_alpha_batch_stride(&mut self, stride: i64) -> Result<()> {
        // SAFETY: `MatmulDescriptorAttribute::AlphaBatchStride` stores one `i64`, the type passed here.
        unsafe { self.set_attribute(MatmulDescriptorAttribute::AlphaBatchStride, &stride) }
    }

    /// Sets the batch stride for per-batch beta values.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_beta_batch_stride(&mut self, stride: i64) -> Result<()> {
        // SAFETY: `MatmulDescriptorAttribute::BetaBatchStride` stores one `i64`, the type passed here.
        unsafe { self.set_attribute(MatmulDescriptorAttribute::BetaBatchStride, &stride) }
    }

    /// Attaches a floating-point emulation descriptor to this matmul descriptor.
    ///
    /// The descriptor stores the native handle of `emulation` without borrowing it.
    ///
    /// # Safety
    ///
    /// `emulation` must not be dropped before every algorithm query and matmul using this descriptor completes, or until the handle is replaced.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub unsafe fn set_emulation_descriptor(
        &mut self,
        emulation: &EmulationDescriptor,
    ) -> Result<()> {
        let pointer = emulation.as_raw();
        // SAFETY: the attribute stores one native handle, and the caller keeps it alive for each use.
        unsafe { self.set_attribute(MatmulDescriptorAttribute::EmulationDescriptor, &pointer) }
    }

    /// Returns the raw cuBLASLt matmul descriptor handle.
    ///
    /// The returned handle is borrowed and remains valid only while the descriptor is alive.
    #[must_use]
    pub fn as_raw(&self) -> sys::cublasLtMatmulDesc_t {
        self.raw
    }

    /// Takes ownership of a raw cuBLASLt matmul descriptor handle.
    ///
    /// # Safety
    ///
    /// `raw` must be a live cuBLASLt matmul descriptor.
    /// The returned value takes ownership and destroys it on drop; no other owner may destroy or keep using it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NullHandle`] if `raw` is null.
    pub unsafe fn from_raw(raw: sys::cublasLtMatmulDesc_t) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self { raw })
    }

    /// Releases ownership and returns the raw cuBLASLt matmul descriptor handle.
    ///
    /// The caller becomes responsible for destroying the handle.
    #[must_use]
    pub fn into_raw(self) -> sys::cublasLtMatmulDesc_t {
        let raw = self.raw;
        mem::forget(self);
        raw
    }
}

impl Drop for MatmulDescriptor {
    fn drop(&mut self) {
        // SAFETY: this value owns `self.raw`, and `drop` destroys it exactly once.
        unsafe {
            if let Err(error) = try_ffi!(sys::cublasLtMatmulDescDestroy(self.raw)) {
                report_cleanup_failure("destroy", "cublaslt matmul descriptor", &error);
            }
        }
    }
}

impl MatmulPreference {
    /// Creates a matmul heuristic preference descriptor.
    ///
    /// The preference owns its cuBLASLt handle and destroys it when dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt cannot allocate the descriptor or if it does not return a valid handle.
    pub fn create() -> Result<Self> {
        let mut raw = ptr::null_mut();
        // SAFETY: `cublasLtMatmulPreferenceCreate` writes the new descriptor to `raw`, a live local, and takes its other arguments by value.
        unsafe {
            try_ffi!(sys::cublasLtMatmulPreferenceCreate(&raw mut raw))?;
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self { raw })
    }

    /// Sets a matmul preference attribute.
    ///
    /// # Safety
    ///
    /// `T` must match the selected attribute's native representation, including its validity and initialized-byte requirements.
    /// Retain pointer targets for every native use.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute or the value size does not match the attribute storage expected by cuBLASLt.
    pub unsafe fn set_attribute<T>(
        &mut self,
        attr: MatmulPreferenceAttribute,
        value: &T,
    ) -> Result<()> {
        set_attribute(
            // SAFETY: `self.raw` is the live descriptor this value owns, and `set_attribute` passes `value` with its exact byte size.
            |value, size| unsafe {
                sys::cublasLtMatmulPreferenceSetAttribute(self.raw, attr.into(), value, size)
            },
            from_ref::<T>(value).cast(),
            size_of::<T>(),
        )
    }

    /// Returns a matmul preference attribute value.
    ///
    /// `T` must accept every bit pattern; enum attributes are read as integers and validated.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the query or if the returned attribute size does not match `T`.
    pub fn attribute<T: AnyBitPatternDeviceRepr>(
        &self,
        attr: MatmulPreferenceAttribute,
    ) -> Result<T> {
        let () = T::LAYOUT;
        let mut value = MaybeUninit::<T>::uninit();
        let written = read_attribute(
            // SAFETY: `self.raw` is the live descriptor this value owns, and `read_attribute` passes `value` with its byte size and a live `written` location.
            |value, size, written| unsafe {
                sys::cublasLtMatmulPreferenceGetAttribute(
                    self.raw,
                    attr.into(),
                    value,
                    size,
                    written,
                )
            },
            value.as_mut_ptr().cast(),
            size_of::<T>(),
            "matmul preference attribute",
        )?;
        ensure_exact_size("matmul preference attribute", written, size_of::<T>())?;
        // SAFETY: the library wrote exactly `size_of::<T>()` bytes, checked above, initializing `value`.
        Ok(unsafe { value.assume_init() })
    }

    /// Sets the heuristic search mode.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_search_mode(&mut self, mode: SearchMode) -> Result<()> {
        // SAFETY: `MatmulPreferenceAttribute::SearchMode` stores one `cublasLtMatmulSearch_t`, the type passed here.
        unsafe {
            self.set_attribute(
                MatmulPreferenceAttribute::SearchMode,
                &sys::cublasLtMatmulSearch_t::from(mode),
            )
        }
    }

    /// Returns the heuristic search mode.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt cannot report the attribute.
    pub fn search_mode(&self) -> Result<SearchMode> {
        let value = self.attribute::<u32>(MatmulPreferenceAttribute::SearchMode)?;
        attribute_enum(value, "MatmulPreferenceAttribute::SearchMode")
    }

    /// Sets the maximum workspace size cuBLASLt may use during matmul.
    ///
    /// # Errors
    ///
    /// Returns an error if the size overflows `u64` or cuBLASLt rejects the attribute.
    pub fn set_max_workspace_bytes(&mut self, size: usize) -> Result<()> {
        let size = to_u64(size, "workspace size")?;
        // SAFETY: `MatmulPreferenceAttribute::MaxWorkspaceBytes` stores one `u64`, the type passed here.
        unsafe { self.set_attribute(MatmulPreferenceAttribute::MaxWorkspaceBytes, &size) }
    }

    /// Sets the average reduction dimension for grouped matmul heuristics.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_grouped_average_reduction_dim(&mut self, average: u32) -> Result<()> {
        // SAFETY: `MatmulPreferenceAttribute::GroupedAverageReductionDim` stores one `u32`, the type passed here.
        unsafe {
            self.set_attribute(
                MatmulPreferenceAttribute::GroupedAverageReductionDim,
                &average,
            )
        }
    }

    /// Sets the average number of rows in D for grouped matmul heuristics.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_grouped_desc_d_average_rows(&mut self, average: u32) -> Result<()> {
        // SAFETY: `MatmulPreferenceAttribute::GroupedDescDAverageRows` stores one `u32`, the type passed here.
        unsafe { self.set_attribute(MatmulPreferenceAttribute::GroupedDescDAverageRows, &average) }
    }

    /// Sets the average number of columns in D for grouped matmul heuristics.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_grouped_desc_d_average_cols(&mut self, average: u32) -> Result<()> {
        // SAFETY: `MatmulPreferenceAttribute::GroupedDescDAverageCols` stores one `u32`, the type passed here.
        unsafe { self.set_attribute(MatmulPreferenceAttribute::GroupedDescDAverageCols, &average) }
    }

    /// Returns the maximum workspace size cuBLASLt may use.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt cannot report the attribute or if the returned size overflows `usize`.
    pub fn max_workspace_bytes(&self) -> Result<usize> {
        let size = self.attribute::<ffi::c_ulong>(MatmulPreferenceAttribute::MaxWorkspaceBytes)?;
        to_usize(size, "workspace size")
    }

    /// Returns the raw cuBLASLt matmul preference handle.
    ///
    /// The returned handle is borrowed and remains valid only while the preference is alive.
    #[must_use]
    pub fn as_raw(&self) -> sys::cublasLtMatmulPreference_t {
        self.raw
    }

    /// Takes ownership of a raw cuBLASLt matmul preference handle.
    ///
    /// # Safety
    ///
    /// `raw` must be a live cuBLASLt matmul preference descriptor.
    /// The returned value takes ownership and destroys it on drop; no other owner may destroy or keep using it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NullHandle`] if `raw` is null.
    pub unsafe fn from_raw(raw: sys::cublasLtMatmulPreference_t) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self { raw })
    }

    /// Releases ownership and returns the raw cuBLASLt matmul preference handle.
    ///
    /// The caller becomes responsible for destroying the handle.
    #[must_use]
    pub fn into_raw(self) -> sys::cublasLtMatmulPreference_t {
        let raw = self.raw;
        mem::forget(self);
        raw
    }
}

impl Drop for MatmulPreference {
    fn drop(&mut self) {
        // SAFETY: this value owns `self.raw`, and `drop` destroys it exactly once.
        unsafe {
            if let Err(error) = try_ffi!(sys::cublasLtMatmulPreferenceDestroy(self.raw)) {
                report_cleanup_failure("destroy", "cublaslt matmul preference", &error);
            }
        }
    }
}

impl MatmulAlgorithm {
    /// Creates an algorithm wrapper from the raw cuBLASLt algorithm representation.
    #[must_use]
    pub const fn from_raw(raw: sys::cublasLtMatmulAlgo_t) -> Self {
        Self { raw }
    }

    /// Returns the raw cuBLASLt algorithm representation.
    #[must_use]
    pub const fn into_raw(self) -> sys::cublasLtMatmulAlgo_t {
        self.raw
    }

    /// Sets a configuration attribute on an initialized [`MatmulAlgorithm`].
    ///
    /// # Safety
    ///
    /// `T` must match the selected attribute's native representation, including its validity and initialized-byte requirements.
    /// Retain pointer targets for every native use.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute or the value size does not match the attribute storage expected by cuBLASLt.
    pub unsafe fn set_config_attribute<T>(
        &mut self,
        attr: MatmulAlgorithmConfigAttribute,
        value: &T,
    ) -> Result<()> {
        set_attribute(
            // SAFETY: `self.raw` is an initialized algorithm this value owns, and `set_attribute` passes `value` with its exact byte size.
            |value, size| unsafe {
                sys::cublasLtMatmulAlgoConfigSetAttribute(
                    &raw mut self.raw,
                    attr.into(),
                    value,
                    size,
                )
            },
            from_ref::<T>(value).cast(),
            size_of::<T>(),
        )
    }

    /// Returns a configuration attribute from an initialized [`MatmulAlgorithm`].
    ///
    /// `T` must accept every bit pattern; enum attributes are read as integers and validated.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the query or if the returned attribute size does not match `T`.
    pub fn config_attribute<T: AnyBitPatternDeviceRepr>(
        &self,
        attr: MatmulAlgorithmConfigAttribute,
    ) -> Result<T> {
        let () = T::LAYOUT;
        let mut value = MaybeUninit::<T>::uninit();
        let written = read_attribute(
            // SAFETY: `self.raw` is an initialized algorithm this value owns, and `read_attribute` passes `value` with its byte size and a live `written` location.
            |value, size, written| unsafe {
                sys::cublasLtMatmulAlgoConfigGetAttribute(
                    &raw const self.raw,
                    attr.into(),
                    value,
                    size,
                    written,
                )
            },
            value.as_mut_ptr().cast(),
            size_of::<T>(),
            "matmul algorithm config attribute",
        )?;
        ensure_exact_size("matmul algorithm config attribute", written, size_of::<T>())?;
        // SAFETY: the library wrote exactly `size_of::<T>()` bytes, checked above, initializing `value`.
        Ok(unsafe { value.assume_init() })
    }

    /// Returns a capability attribute from an initialized [`MatmulAlgorithm`].
    ///
    /// `T` must accept every bit pattern; enum attributes are read as integers and validated.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the query or if the returned attribute size does not match `T`.
    pub fn cap_attribute<T: AnyBitPatternDeviceRepr>(
        &self,
        attr: MatmulAlgorithmCapAttribute,
    ) -> Result<T> {
        let () = T::LAYOUT;
        let mut value = MaybeUninit::<T>::uninit();
        let written = read_attribute(
            // SAFETY: `self.raw` is an initialized algorithm this value owns, and `read_attribute` passes `value` with its byte size and a live `written` location.
            |value, size, written| unsafe {
                sys::cublasLtMatmulAlgoCapGetAttribute(
                    &raw const self.raw,
                    attr.into(),
                    value,
                    size,
                    written,
                )
            },
            value.as_mut_ptr().cast(),
            size_of::<T>(),
            "matmul algorithm cap attribute",
        )?;
        ensure_exact_size("matmul algorithm cap attribute", written, size_of::<T>())?;
        // SAFETY: the library wrote exactly `size_of::<T>()` bytes, checked above, initializing `value`.
        Ok(unsafe { value.assume_init() })
    }

    /// Returns whether this algorithm supports split-K reduction.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt cannot query the split-K capability or reports an attribute size different from `i32`.
    pub fn supports_split_k(&self) -> Result<bool> {
        Ok(self.cap_attribute::<i32>(MatmulAlgorithmCapAttribute::SplitKSupport)? != 0)
    }

    /// Sets the split-K partition count, with checked conversion to the native integer width.
    ///
    /// # Errors
    ///
    /// Returns an error if `count` exceeds the `i32` range or cuBLASLt rejects the split-K configuration.
    pub fn set_split_k_count(&mut self, count: usize) -> Result<()> {
        let count = to_i32(count, "split-K count")?;
        // SAFETY: SplitKCount stores one native signed 32-bit integer.
        unsafe { self.set_config_attribute(MatmulAlgorithmConfigAttribute::SplitKCount, &count) }
    }

    /// Returns whether this algorithm can be used with strided-batch matrix layouts.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt cannot report the capability.
    pub fn supports_strided_batch(&self) -> Result<bool> {
        Ok(self.cap_attribute::<i32>(MatmulAlgorithmCapAttribute::StridedBatchSupport)? != 0)
    }

    /// Returns whether this algorithm can be used with pointer-array batch matrix layouts.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt cannot report the capability.
    pub fn supports_pointer_array_batch(&self) -> Result<bool> {
        Ok(self.cap_attribute::<i32>(MatmulAlgorithmCapAttribute::PointerArrayBatchSupport)? != 0)
    }

    /// Returns a reference to the raw cuBLASLt algorithm representation.
    ///
    /// The returned value is borrowed and remains valid only while the algorithm wrapper is alive.
    #[must_use]
    pub fn as_raw(&self) -> &sys::cublasLtMatmulAlgo_t {
        &self.raw
    }
}

impl MatmulHeuristicResult {
    fn try_from_raw(raw: &sys::cublasLtMatmulHeuristicResult_t) -> Result<Self> {
        let workspace_size = to_usize(raw.workspaceSize, "heuristic workspace size")?;

        Ok(Self {
            algorithm: MatmulAlgorithm::from_raw(raw.algo),
            workspace_size,
            status: raw.state.into(),
            waves_count: raw.wavesCount,
        })
    }
}

/// Enqueues `D = alpha * op(A) * op(B) + beta * C` on `stream`, or on the legacy default stream for `None`, without waiting.
///
/// The descriptor and layouts give the data types, shapes, orders, batches, scales, and epilogue.
/// `c` of `None` reads `C` from `D`'s storage, for an in-place update or a `beta` of zero.
/// A workspace, when given, must satisfy the selected algorithm's size and alignment requirements.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
/// The operands must hold every element their layouts describe, with the layouts' data types.
/// `alpha` and `beta` must have the descriptor's scale type and the location its pointer mode selects.
/// Every device address and handle stored in the descriptor and layouts, by setters such as [`MatmulDescriptor::set_bias_pointer`], [`MatmulDescriptor::set_a_scale_pointer`], [`MatmulDescriptor::set_amax_d_pointer`], [`MatmulDescriptor::set_emulation_descriptor`], and [`MatrixLayout::create_grouped`], must meet its setter's requirements until the work completes.
///
/// # Errors
///
/// Returns an error if an operand, scalar, workspace, or the stream belongs to another CUDA context, the dependencies on the operands' producers cannot be enqueued, or cuBLASLt rejects the descriptors, algorithm, or workspace.
pub unsafe fn matmul_async<TA, TB, TD, TS>(
    context: &Context,
    descriptor: &MatmulDescriptor,
    alpha: Scalar<'_, TS>,
    a: &(impl DeviceRange<TA> + ?Sized),
    a_layout: &MatrixLayout,
    b: &(impl DeviceRange<TB> + ?Sized),
    b_layout: &MatrixLayout,
    beta: Scalar<'_, TS>,
    c: Option<DeviceInputView<'_, TD>>,
    c_layout: &MatrixLayout,
    d: &mut (impl DeviceOutput<TD> + ?Sized),
    d_layout: &MatrixLayout,
    algorithm: Option<&MatmulAlgorithm>,
    mut workspace: Option<DeviceOutputView<'_, u8>>,
    stream: Option<&Stream>,
) -> Result<()>
where
    TA: DeviceRepr,
    TB: DeviceRepr,
    TD: DeviceRepr,
    TS: DeviceRepr,
{
    let native_stream = context.native_stream(stream)?;
    let regions = [a.region(), b.region(), d.region()]
        .into_iter()
        .chain(c.as_ref().map(DeviceInputView::region))
        .chain(workspace.as_ref().map(DeviceRegionProvider::region))
        .chain(alpha.region())
        .chain(beta.region())
        .collect::<Vec<_>>();
    if regions
        .iter()
        .any(|region| region.context() != context.cuda_context())
    {
        return Err(CudaError::MemoryContextMismatch.into());
    }

    context.stream_binding(stream).join_regions(&regions)?;
    let d_ptr = d.as_mut_ptr();
    let c_ptr = c.map_or_else(|| d_ptr.cast_const(), |c| c.as_ptr());
    let (workspace_ptr, workspace_size): (*mut c_void, usize) = workspace
        .as_mut()
        .map_or((ptr::null_mut(), 0), |workspace| {
            (workspace.as_mut_ptr().cast(), workspace.len())
        });
    let workspace_size = to_size_t(workspace_size, "workspace size")?;
    // SAFETY: the operands, scalars, and workspace belong to this context and are ordered after their producers on the stream; the caller upholds the layout and raw execution contracts.
    unsafe {
        try_ffi!(sys::cublasLtMatmul(
            context.as_raw(),
            descriptor.as_raw(),
            alpha.ptr().cast(),
            a.as_device_ptr().cast(),
            a_layout.as_raw(),
            b.as_device_ptr().cast(),
            b_layout.as_raw(),
            beta.ptr().cast(),
            c_ptr.cast(),
            c_layout.as_raw(),
            d_ptr.cast(),
            d_layout.as_raw(),
            algorithm.map_or(ptr::null(), |algorithm| algorithm.as_raw()),
            workspace_ptr,
            workspace_size,
            native_stream,
        ))
    }
}

/// Enqueues `C = alpha * transform(A) + beta * transform(B)` on `stream`, or on the legacy default stream for `None`, without waiting.
///
/// The descriptor selects the transpositions, scale type, and pointer mode, and the layouts may reorder the data.
/// `b` of `None` omits the second matrix.
///
/// # Safety
///
/// The caller upholds the [raw execution contract](crate#raw-execution).
/// The operands must hold every element their layouts describe, with the layouts' data types.
/// `alpha` and `beta` must have the descriptor's scale type and the location its pointer mode selects.
///
/// # Errors
///
/// Returns an error if an operand, scalar, or the stream belongs to another CUDA context, the dependencies on the operands' producers cannot be enqueued, or cuBLASLt rejects the layouts, scalars, or transform.
pub unsafe fn matrix_transform_async<TA, TC, TS>(
    context: &Context,
    descriptor: &MatrixTransformDescriptor,
    alpha: Scalar<'_, TS>,
    a: &(impl DeviceRange<TA> + ?Sized),
    a_layout: &MatrixLayout,
    beta: Scalar<'_, TS>,
    b: Option<(DeviceInputView<'_, TA>, &MatrixLayout)>,
    c: &mut (impl DeviceOutput<TC> + ?Sized),
    c_layout: &MatrixLayout,
    stream: Option<&Stream>,
) -> Result<()>
where
    TA: DeviceRepr,
    TC: DeviceRepr,
    TS: DeviceRepr,
{
    let native_stream = context.native_stream(stream)?;
    let regions = [a.region(), c.region()]
        .into_iter()
        .chain(b.as_ref().map(|(b, _)| b.region()))
        .chain(alpha.region())
        .chain(beta.region())
        .collect::<Vec<_>>();
    if regions
        .iter()
        .any(|region| region.context() != context.cuda_context())
    {
        return Err(CudaError::MemoryContextMismatch.into());
    }

    context.stream_binding(stream).join_regions(&regions)?;
    let (b_ptr, b_layout): (*const c_void, sys::cublasLtMatrixLayout_t) = b
        .map_or((ptr::null(), ptr::null_mut()), |(b, layout)| {
            (b.as_ptr().cast(), layout.as_raw())
        });
    // SAFETY: the operands and scalars belong to this context and are ordered after their producers on the stream; the caller upholds the layout and raw execution contracts.
    unsafe {
        try_ffi!(sys::cublasLtMatrixTransform(
            context.as_raw(),
            descriptor.as_raw(),
            alpha.ptr().cast(),
            a.as_device_ptr().cast(),
            a_layout.as_raw(),
            beta.ptr().cast(),
            b_ptr,
            b_layout,
            c.as_mut_ptr().cast(),
            c_layout.as_raw(),
            native_stream,
        ))
    }
}

/// Returns algorithm IDs that may be usable with [`matmul_async`] for the given data types.
///
/// # Errors
///
/// Returns an error if `requested_count` cannot be represented as `i32` or if cuBLASLt rejects the query.
pub fn matmul_algorithm_ids(
    context: &Context,
    compute_type: ComputeType,
    scale_type: DataType,
    a_type: DataType,
    b_type: DataType,
    c_type: DataType,
    d_type: DataType,
    requested_count: usize,
) -> Result<Vec<i32>> {
    context.bind()?;

    let native_count = to_i32(requested_count, "requested algorithm count")?;
    let mut ids = vec![0; requested_count];
    let mut actual_count = 0;

    // SAFETY: the bound handle is live, `ids` holds `native_count` elements, and `actual_count` is a live local.
    unsafe {
        try_ffi!(sys::cublasLtMatmulAlgoGetIds(
            context.as_raw(),
            compute_type.into(),
            scale_type.into(),
            a_type.into(),
            b_type.into(),
            c_type.into(),
            d_type.into(),
            native_count,
            ids.as_mut_ptr(),
            &raw mut actual_count,
        ))?;
    }

    ids.truncate(to_usize(actual_count, "actual algorithm count")?);
    Ok(ids)
}

/// Initializes a [`MatmulAlgorithm`] from a cuBLASLt algorithm ID.
///
/// # Errors
///
/// Returns an error if the algorithm ID is invalid or unsupported for the supplied data type combination.
pub fn matmul_algorithm(
    context: &Context,
    compute_type: ComputeType,
    scale_type: DataType,
    a_type: DataType,
    b_type: DataType,
    c_type: DataType,
    d_type: DataType,
    algorithm_id: i32,
) -> Result<MatmulAlgorithm> {
    context.bind()?;

    let mut algorithm = sys::cublasLtMatmulAlgo_t::default();
    // SAFETY: the bound handle is live, and `algorithm` is a live local the call initializes.
    unsafe {
        try_ffi!(sys::cublasLtMatmulAlgoInit(
            context.as_raw(),
            compute_type.into(),
            scale_type.into(),
            a_type.into(),
            b_type.into(),
            c_type.into(),
            d_type.into(),
            algorithm_id,
            &raw mut algorithm,
        ))?;
    }
    Ok(MatmulAlgorithm::from_raw(algorithm))
}

/// Returns heuristic algorithm candidates for [`matmul_async`].
///
/// Results are ordered by increasing estimated compute time.
///
/// # Errors
///
/// Returns an error if `requested_count` cannot be represented as `i32`, if no heuristic is available for the configuration, or if cuBLASLt rejects the query.
pub fn matmul_algorithm_heuristics(
    context: &Context,
    desc: &MatmulDescriptor,
    a_desc: &MatrixLayout,
    b_desc: &MatrixLayout,
    c_desc: &MatrixLayout,
    d_desc: &MatrixLayout,
    preference: &MatmulPreference,
    requested_count: usize,
) -> Result<Vec<MatmulHeuristicResult>> {
    context.bind()?;

    let native_count = to_i32(requested_count, "requested heuristic count")?;
    let mut actual_count = 0;
    let mut heuristics = Vec::with_capacity(requested_count);
    heuristics.resize_with(
        requested_count,
        sys::cublasLtMatmulHeuristicResult_t::default,
    );

    // SAFETY: the bound handle and descriptors are live borrows, `heuristics` holds `native_count` elements, and `actual_count` is a live local.
    unsafe {
        try_ffi!(sys::cublasLtMatmulAlgoGetHeuristic(
            context.as_raw(),
            desc.as_raw(),
            a_desc.as_raw(),
            b_desc.as_raw(),
            c_desc.as_raw(),
            d_desc.as_raw(),
            preference.as_raw(),
            native_count,
            heuristics.as_mut_ptr(),
            &raw mut actual_count,
        ))?;
    }

    heuristics.truncate(to_usize(actual_count, "actual heuristic count")?);

    heuristics
        .iter()
        .map(MatmulHeuristicResult::try_from_raw)
        .collect()
}

/// Checks whether a [`MatmulAlgorithm`] is valid for the supplied descriptors on the current device.
///
/// On success, returns the required workspace size and estimated wave count reported by cuBLASLt.
///
/// # Errors
///
/// Returns an error if the descriptors do not match the algorithm configuration or if the device does not support the requested algorithm.
pub fn check_matmul_algorithm(
    context: &Context,
    desc: &MatmulDescriptor,
    a_desc: &MatrixLayout,
    b_desc: &MatrixLayout,
    c_desc: &MatrixLayout,
    d_desc: &MatrixLayout,
    algorithm: &MatmulAlgorithm,
) -> Result<MatmulHeuristicResult> {
    context.bind()?;

    let mut heuristic = sys::cublasLtMatmulHeuristicResult_t {
        algo: *algorithm.as_raw(),
        ..Default::default()
    };
    // SAFETY: the bound handle, descriptors, and algorithm are live borrows, and `heuristic` is a live local the call writes.
    unsafe {
        try_ffi!(sys::cublasLtMatmulAlgoCheck(
            context.as_raw(),
            desc.as_raw(),
            a_desc.as_raw(),
            b_desc.as_raw(),
            c_desc.as_raw(),
            d_desc.as_raw(),
            algorithm.as_raw(),
            &raw mut heuristic,
        ))?;
    }
    MatmulHeuristicResult::try_from_raw(&heuristic)
}

/// Returns up to `requested_count` algorithm candidates for the supplied descriptors when run on `stream`.
///
/// If the stream is associated with a green context, the result reflects its SM count and resource constraints, not the full device.
///
/// # Errors
///
/// Returns an error if the stream belongs to another CUDA context, context binding fails, or `requested_count` exceeds the `i32` range.
/// Also returns an error if cuBLASLt rejects the query or a returned count or workspace size cannot be represented by its Rust integer type.
pub fn matmul_algorithm_heuristics_for_stream(
    context: &Context,
    stream: &Stream,
    desc: &MatmulDescriptor,
    a_desc: &MatrixLayout,
    b_desc: &MatrixLayout,
    c_desc: &MatrixLayout,
    d_desc: &MatrixLayout,
    preference: &MatmulPreference,
    requested_count: usize,
) -> Result<Vec<MatmulHeuristicResult>> {
    context.ensure_stream(stream)?;
    context.bind()?;

    let native_count = to_i32(requested_count, "requested heuristic count")?;
    let mut actual_count = 0;
    let mut heuristics = Vec::with_capacity(requested_count);
    heuristics.resize_with(
        requested_count,
        sys::cublasLtMatmulHeuristicResult_t::default,
    );

    // SAFETY: the bound handle and descriptors are live borrows, `heuristics` holds `native_count` elements, and `actual_count` is a live local.
    unsafe {
        try_ffi!(sys::cublasLtMatmulAlgoGetHeuristicForStream(
            context.as_raw(),
            desc.as_raw(),
            a_desc.as_raw(),
            b_desc.as_raw(),
            c_desc.as_raw(),
            d_desc.as_raw(),
            preference.as_raw(),
            native_count,
            heuristics.as_mut_ptr(),
            &raw mut actual_count,
            stream.as_raw().cast(),
        ))?;
    }

    heuristics.truncate(to_usize(actual_count, "actual heuristic count")?);

    heuristics
        .iter()
        .map(MatmulHeuristicResult::try_from_raw)
        .collect()
}

/// Checks whether a [`MatmulAlgorithm`] is valid for the supplied descriptors when run on `stream`.
///
/// If the stream is associated with a green context, the result reflects its SM count and resource constraints, not the full device.
///
/// # Errors
///
/// Returns an error if the stream belongs to another CUDA context, binding the context fails, or cuBLASLt rejects the algorithm check.
/// Also returns an error if the reported workspace size cannot be represented as `usize`.
/// An unsuccessful candidate status is returned in [`MatmulHeuristicResult::status`].
pub fn check_matmul_algorithm_for_stream(
    context: &Context,
    stream: &Stream,
    desc: &MatmulDescriptor,
    a_desc: &MatrixLayout,
    b_desc: &MatrixLayout,
    c_desc: &MatrixLayout,
    d_desc: &MatrixLayout,
    algorithm: &MatmulAlgorithm,
) -> Result<MatmulHeuristicResult> {
    context.ensure_stream(stream)?;
    context.bind()?;

    let mut heuristic = sys::cublasLtMatmulHeuristicResult_t {
        algo: *algorithm.as_raw(),
        ..Default::default()
    };
    // SAFETY: the bound handle, descriptors, and algorithm are live borrows, and `heuristic` is a live local the call writes.
    unsafe {
        try_ffi!(sys::cublasLtMatmulAlgoCheckForStream(
            context.as_raw(),
            desc.as_raw(),
            a_desc.as_raw(),
            b_desc.as_raw(),
            c_desc.as_raw(),
            d_desc.as_raw(),
            algorithm.as_raw(),
            &raw mut heuristic,
            stream.as_raw().cast(),
        ))?;
    }
    MatmulHeuristicResult::try_from_raw(&heuristic)
}
