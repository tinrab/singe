use core::ffi;
use std::{
    ffi::c_void,
    mem,
    mem::{MaybeUninit, size_of},
    ptr::{self, from_ref},
};

use singe_cublas_sys as sys;
use singe_cuda::{
    data_type::DataType,
    error::Error as CudaError,
    memory::DeviceMemory,
    stream::Stream,
    types::{Complex32, Complex64},
    view::{
        AnyBitPatternDeviceRepr, DeviceInput, DeviceInputView, DeviceOutput, DeviceOutputView,
        DeviceRepr, DeviceView, DeviceViewMut, NoUninitDeviceRepr,
    },
};
use singe_cuda_sys::library_types::cudaDataType_t;

use crate::{
    error::{Error, Result, Status},
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
    try_ffi,
    types::{ComputeType, FillMode, Operation},
    utility::{ensure_exact_size, to_i32, to_u64, to_usize},
};

/// Describes the configuration for a cuBLASLt matrix multiplication operation.
///
/// Owns a cuBLASLt matmul descriptor handle. Configure transpose modes,
/// epilogue, scaling, and other attributes through the setter methods.
#[derive(Debug)]
pub struct MatmulDescriptor {
    raw: sys::cublasLtMatmulDesc_t,
}

/// Stores algorithm-search preferences for cuBLASLt matmul heuristic queries.
///
/// Owns a cuBLASLt matmul preference handle. Use setter methods to configure
/// workspace limits, search mode, alignment requirements, and other tuning
/// parameters.
#[derive(Debug)]
pub struct MatmulPreference {
    raw: sys::cublasLtMatmulPreference_t,
}

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

/// Trait for types that can provide a const pointer to device data.
///
/// # Safety
///
/// The pointer must identify live storage for the borrow, in the
/// address space appropriate to its use. Host scalar implementations may only
/// be used where the operation accepts host scalar pointers.
/// Raw operations require valid values when each range is read; address
/// metadata alone does not establish initialization.
pub unsafe trait ConstDataPointer {
    /// Returns a borrowed pointer to the underlying data.
    fn as_const_data_ptr(&self) -> *const c_void;
}

/// Trait for types that can provide a mutable pointer to device data.
///
/// # Safety
///
/// The pointer must identify live writable storage and grant exclusive access
/// for the mutable borrow. Operation-specific bounds and GPU lifetimes remain
/// the caller's responsibility when using raw submission APIs.
pub unsafe trait MutDataPointer {
    /// Returns a borrowed writable pointer to the underlying data.
    fn as_mut_data_ptr(&mut self) -> *mut c_void;
}

unsafe impl<T: DeviceRepr> MutDataPointer for DeviceMemory<T> {
    fn as_mut_data_ptr(&mut self) -> *mut c_void {
        self.as_mut_ptr().cast()
    }
}

unsafe impl<T: NoUninitDeviceRepr> ConstDataPointer for DeviceMemory<T> {
    fn as_const_data_ptr(&self) -> *const c_void {
        self.as_ptr().cast()
    }
}

unsafe impl<T: NoUninitDeviceRepr> ConstDataPointer for DeviceView<'_, T> {
    fn as_const_data_ptr(&self) -> *const c_void {
        self.as_ptr().cast()
    }
}

unsafe impl<T: NoUninitDeviceRepr> ConstDataPointer for DeviceViewMut<'_, T> {
    fn as_const_data_ptr(&self) -> *const c_void {
        self.as_ptr().cast()
    }
}

// SAFETY: deferred views retain an aligned allocation borrow; raw calls establish validity when read.
unsafe impl<T: DeviceRepr> ConstDataPointer for DeviceInputView<'_, T> {
    fn as_const_data_ptr(&self) -> *const c_void {
        self.as_device_ptr().cast()
    }
}

unsafe impl<T: DeviceRepr> MutDataPointer for DeviceViewMut<'_, T> {
    fn as_mut_data_ptr(&mut self) -> *mut c_void {
        self.as_mut_ptr().cast()
    }
}

unsafe impl<T: DeviceRepr> MutDataPointer for DeviceOutputView<'_, T> {
    fn as_mut_data_ptr(&mut self) -> *mut c_void {
        self.as_mut_ptr().cast()
    }
}

macro_rules! impl_host_scalar_pointer {
    ($($ty:ty),+ $(,)?) => {
        $(
            unsafe impl ConstDataPointer for $ty {
                fn as_const_data_ptr(&self) -> *const c_void {
                    from_ref(self).cast()
                }
            }
        )+
    };
}

impl_host_scalar_pointer!(f32, f64, i32, Complex32, Complex64);

impl MatmulDescriptor {
    /// Requests batch-invariant matrix multiplication results.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the batch-invariance policy.
    pub fn set_batch_invariance(&mut self, policy: BatchInvariance) -> Result<()> {
        // SAFETY: this named adapter selects the attribute's documented native representation.
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
    /// Returns an error if cuBLASLt cannot allocate the descriptor or if it does not return
    /// a valid handle.
    pub fn create(compute_type: ComputeType, scale_type: DataType) -> Result<Self> {
        let mut raw = ptr::null_mut();
        // SAFETY: `cublasLtMatmulDescCreate` receives valid output locations.
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
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute or the value size does not match the
    /// attribute storage expected by cuBLASLt.
    ///
    /// # Safety
    ///
    /// `T` must match the selected attribute's native representation, including
    /// its validity and initialized-byte requirements. Retain pointer targets for every native use.
    pub unsafe fn set_attribute<T>(
        &mut self,
        attr: MatmulDescriptorAttribute,
        value: &T,
    ) -> Result<()> {
        set_attribute(
            // SAFETY: `cublasLtMatmulDescSetAttribute` receives live handles owned by the wrappers.
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
    /// Returns an error if cuBLASLt rejects the query or if the returned attribute size does not
    /// match `T`.
    pub fn attribute<T: AnyBitPatternDeviceRepr>(
        &self,
        attr: MatmulDescriptorAttribute,
    ) -> Result<T> {
        let () = T::LAYOUT;
        let mut value = MaybeUninit::<T>::uninit();
        let written = read_attribute(
            // SAFETY: `cublasLtMatmulDescGetAttribute` receives live handles owned by the wrappers.
            |value, size, written| unsafe {
                sys::cublasLtMatmulDescGetAttribute(self.raw, attr.into(), value, size, written)
            },
            value.as_mut_ptr().cast(),
            size_of::<T>(),
            "matmul descriptor attribute",
        )?;
        ensure_exact_size(written, size_of::<T>())?;
        // SAFETY: the library wrote exactly `size_of::<T>()` bytes, checked above, initializing `value`.
        Ok(unsafe { value.assume_init() })
    }

    /// Sets the pointer mode for alpha and beta scalars.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_pointer_mode(&mut self, pointer_mode: PointerMode) -> Result<()> {
        // SAFETY: this named adapter selects the attribute's documented native representation.
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
        // SAFETY: this named adapter selects the attribute's documented native representation.
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
        // SAFETY: this named adapter selects the attribute's documented native representation.
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
        // SAFETY: this named adapter selects the attribute's documented native representation.
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
        // SAFETY: this named adapter selects the attribute's documented native representation.
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
        // SAFETY: this named adapter selects the attribute's documented native representation.
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
        // SAFETY: this named adapter selects the attribute's documented native representation.
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
        // SAFETY: this named adapter selects the attribute's documented native representation.
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
        // SAFETY: this named adapter selects the attribute's documented native representation.
        unsafe {
            self.set_attribute(
                MatmulDescriptorAttribute::DScaleMode,
                &sys::cublasLtMatmulMatrixScale_t::from(scale_mode),
            )
        }
    }

    /// Enables or disables fast accumulation.
    ///
    /// Fast accumulation may use higher-precision internally for better
    /// accuracy at the cost of performance.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_fast_accum(&mut self, enabled: bool) -> Result<()> {
        // SAFETY: this named adapter selects the attribute's documented native representation.
        unsafe { self.set_attribute(MatmulDescriptorAttribute::FastAccum, &i32::from(enabled)) }
    }

    /// Sets the bias pointer for the bias epilogue.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_bias_pointer<T: DeviceRepr>(
        &mut self,
        bias: &(impl DeviceInput<T> + ?Sized),
    ) -> Result<()> {
        let pointer = bias.as_device_ptr();
        // SAFETY: this named adapter selects the attribute's documented native representation.
        unsafe { self.set_attribute(MatmulDescriptorAttribute::BiasPointer, &pointer) }
    }

    /// Sets the bias data type for bias epilogues.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_bias_data_type(&mut self, data_type: DataType) -> Result<()> {
        // SAFETY: this named adapter selects the attribute's documented native representation.
        unsafe {
            self.set_attribute(
                MatmulDescriptorAttribute::BiasDataType,
                &cudaDataType_t::from(data_type),
            )
        }
    }

    /// Sets the scale pointer for matrix A.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_a_scale_pointer<T>(&mut self, scale: &DeviceMemory<T>) -> Result<()> {
        let pointer = scale.as_ptr();
        // SAFETY: this named adapter selects the attribute's documented native representation.
        unsafe { self.set_attribute(MatmulDescriptorAttribute::AScalePointer, &pointer) }
    }

    /// Sets the scale pointer for matrix B.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_b_scale_pointer<T>(&mut self, scale: &DeviceMemory<T>) -> Result<()> {
        let pointer = scale.as_ptr();
        // SAFETY: this named adapter selects the attribute's documented native representation.
        unsafe { self.set_attribute(MatmulDescriptorAttribute::BScalePointer, &pointer) }
    }

    /// Sets the scale pointer for matrix C.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_c_scale_pointer<T>(&mut self, scale: &DeviceMemory<T>) -> Result<()> {
        let pointer = scale.as_ptr();
        // SAFETY: this named adapter selects the attribute's documented native representation.
        unsafe { self.set_attribute(MatmulDescriptorAttribute::CScalePointer, &pointer) }
    }

    /// Sets the scale pointer for matrix D.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_d_scale_pointer<T>(&mut self, scale: &DeviceMemory<T>) -> Result<()> {
        let pointer = scale.as_ptr();
        // SAFETY: this named adapter selects the attribute's documented native representation.
        unsafe { self.set_attribute(MatmulDescriptorAttribute::DScalePointer, &pointer) }
    }

    /// Sets the output scale pointer for matrix D.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_d_out_scale_pointer<T>(&mut self, scale: &DeviceMemory<T>) -> Result<()> {
        let pointer = scale.as_ptr();
        // SAFETY: this named adapter selects the attribute's documented native representation.
        unsafe { self.set_attribute(MatmulDescriptorAttribute::DOutScalePointer, &pointer) }
    }

    /// Sets the pointer for storing the absolute maximum value of D.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_amax_d_pointer<T>(&mut self, amax: &mut DeviceMemory<T>) -> Result<()> {
        let pointer = amax.as_mut_ptr();
        // SAFETY: this named adapter selects the attribute's documented native representation.
        unsafe { self.set_attribute(MatmulDescriptorAttribute::AmaxDPointer, &pointer) }
    }

    /// Sets the output scaling mode for matrix D.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_d_out_scale_mode(&mut self, scale_mode: MatrixScale) -> Result<()> {
        // SAFETY: this named adapter selects the attribute's documented native representation.
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
        // SAFETY: this named adapter selects the attribute's documented native representation.
        unsafe { self.set_attribute(MatmulDescriptorAttribute::AlphaBatchStride, &stride) }
    }

    /// Sets the batch stride for per-batch beta values.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_beta_batch_stride(&mut self, stride: i64) -> Result<()> {
        // SAFETY: this named adapter selects the attribute's documented native representation.
        unsafe { self.set_attribute(MatmulDescriptorAttribute::BetaBatchStride, &stride) }
    }

    /// Attaches a floating-point emulation descriptor to this matmul descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_emulation_descriptor(&mut self, emulation: &EmulationDescriptor) -> Result<()> {
        let pointer = emulation.as_raw();
        // SAFETY: this named adapter selects the attribute's documented native representation.
        unsafe { self.set_attribute(MatmulDescriptorAttribute::EmulationDescriptor, &pointer) }
    }

    /// Returns the raw cuBLASLt matmul descriptor handle.
    ///
    /// The returned handle is borrowed and remains valid only while the
    /// descriptor is alive.
    #[must_use]
    pub fn as_raw(&self) -> sys::cublasLtMatmulDesc_t {
        self.raw
    }

    /// Takes ownership of a raw cuBLASLt matmul descriptor handle.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid `cublasLtMatmulDesc_t` handle created by cuBLASLt.
    /// The returned wrapper takes ownership and will destroy it with
    /// `cublasLtMatmulDescDestroy`; no other owner may destroy or keep using it.
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
        // SAFETY: `cublasLtMatmulDescDestroy` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cublasLtMatmulDescDestroy(self.raw)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cublaslt matmul descriptor: {error}");
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
    /// Returns an error if cuBLASLt cannot allocate the descriptor or if it does not return
    /// a valid handle.
    pub fn create() -> Result<Self> {
        let mut raw = ptr::null_mut();
        // SAFETY: `cublasLtMatmulPreferenceCreate` receives valid output locations.
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
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute or the value size does not match the
    /// attribute storage expected by cuBLASLt.
    ///
    /// # Safety
    ///
    /// `T` must match the selected attribute's native representation, including
    /// its validity and initialized-byte requirements. Retain pointer targets for every native use.
    pub unsafe fn set_attribute<T>(
        &mut self,
        attr: MatmulPreferenceAttribute,
        value: &T,
    ) -> Result<()> {
        set_attribute(
            // SAFETY: `cublasLtMatmulPreferenceSetAttribute` receives live handles owned by the wrappers.
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
    /// Returns an error if cuBLASLt rejects the query or if the returned attribute size does not
    /// match `T`.
    pub fn attribute<T: AnyBitPatternDeviceRepr>(
        &self,
        attr: MatmulPreferenceAttribute,
    ) -> Result<T> {
        let () = T::LAYOUT;
        let mut value = MaybeUninit::<T>::uninit();
        let written = read_attribute(
            // SAFETY: `cublasLtMatmulPreferenceGetAttribute` receives live handles owned by the wrappers.
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
        ensure_exact_size(written, size_of::<T>())?;
        // SAFETY: the library wrote exactly `size_of::<T>()` bytes, checked above, initializing `value`.
        Ok(unsafe { value.assume_init() })
    }

    /// Sets the heuristic search mode.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_search_mode(&mut self, mode: SearchMode) -> Result<()> {
        // SAFETY: this named adapter selects the attribute's documented native representation.
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
        // SAFETY: this named adapter selects the attribute's documented native representation.
        unsafe { self.set_attribute(MatmulPreferenceAttribute::MaxWorkspaceBytes, &size) }
    }

    /// Sets the average reduction dimension for grouped matmul heuristics.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_grouped_average_reduction_dim(&mut self, average: i64) -> Result<()> {
        // SAFETY: this named adapter selects the attribute's documented native representation.
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
    pub fn set_grouped_desc_d_average_rows(&mut self, average: i64) -> Result<()> {
        // SAFETY: this named adapter selects the attribute's documented native representation.
        unsafe { self.set_attribute(MatmulPreferenceAttribute::GroupedDescDAverageRows, &average) }
    }

    /// Sets the average number of columns in D for grouped matmul heuristics.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute.
    pub fn set_grouped_desc_d_average_cols(&mut self, average: i64) -> Result<()> {
        // SAFETY: this named adapter selects the attribute's documented native representation.
        unsafe { self.set_attribute(MatmulPreferenceAttribute::GroupedDescDAverageCols, &average) }
    }

    /// Returns the maximum workspace size cuBLASLt may use.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt cannot report the attribute or if the
    /// returned size overflows `usize`.
    pub fn max_workspace_bytes(&self) -> Result<usize> {
        let size = self.attribute::<ffi::c_ulong>(MatmulPreferenceAttribute::MaxWorkspaceBytes)?;
        to_usize(size, "workspace size")
    }

    /// Returns the raw cuBLASLt matmul preference handle.
    ///
    /// The returned handle is borrowed and remains valid only while the
    /// preference is alive.
    #[must_use]
    pub fn as_raw(&self) -> sys::cublasLtMatmulPreference_t {
        self.raw
    }

    /// Takes ownership of a raw cuBLASLt matmul preference handle.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid `cublasLtMatmulPreference_t` handle created by
    /// cuBLASLt. The returned wrapper takes ownership and will destroy it with
    /// `cublasLtMatmulPreferenceDestroy`; no other owner may destroy or keep
    /// using it.
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
        // SAFETY: `cublasLtMatmulPreferenceDestroy` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cublasLtMatmulPreferenceDestroy(self.raw)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cublaslt matmul preference: {error}");
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
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the attribute or the value size does not match the
    /// attribute storage expected by cuBLASLt.
    ///
    /// # Safety
    ///
    /// `T` must match the selected attribute's native representation, including
    /// its validity and initialized-byte requirements. Retain pointer targets for every native use.
    pub unsafe fn set_config_attribute<T>(
        &mut self,
        attr: MatmulAlgorithmConfigAttribute,
        value: &T,
    ) -> Result<()> {
        set_attribute(
            // SAFETY: `cublasLtMatmulAlgoConfigSetAttribute` receives live handles owned by the wrappers and valid output locations.
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
    /// Returns an error if cuBLASLt rejects the query or if the returned attribute size does not
    /// match `T`.
    pub fn config_attribute<T: AnyBitPatternDeviceRepr>(
        &self,
        attr: MatmulAlgorithmConfigAttribute,
    ) -> Result<T> {
        let () = T::LAYOUT;
        let mut value = MaybeUninit::<T>::uninit();
        let written = read_attribute(
            // SAFETY: `cublasLtMatmulAlgoConfigGetAttribute` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
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
        ensure_exact_size(written, size_of::<T>())?;
        // SAFETY: the library wrote exactly `size_of::<T>()` bytes, checked above, initializing `value`.
        Ok(unsafe { value.assume_init() })
    }

    /// Returns a capability attribute from an initialized [`MatmulAlgorithm`].
    ///
    /// `T` must accept every bit pattern; enum attributes are read as integers and validated.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt rejects the query or if the returned attribute size does not
    /// match `T`.
    pub fn cap_attribute<T: AnyBitPatternDeviceRepr>(
        &self,
        attr: MatmulAlgorithmCapAttribute,
    ) -> Result<T> {
        let () = T::LAYOUT;
        let mut value = MaybeUninit::<T>::uninit();
        let written = read_attribute(
            // SAFETY: `cublasLtMatmulAlgoCapGetAttribute` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
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
        ensure_exact_size(written, size_of::<T>())?;
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
    /// The returned value is borrowed and remains valid only while the
    /// algorithm wrapper is alive.
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

/// Computes a matrix multiplication using a configured cuBLASLt algorithm.
///
/// `D = alpha*(A*B) + beta*(C),`
///
/// `alpha`, `beta`, and the matrix pointers must match the types and pointer mode configured in
/// `desc`.
///
/// When provided, `workspace` must satisfy cuBLASLt's alignment requirements and be large enough
/// for the selected algorithm.
///
/// # Safety
///
/// Matrix and scalar pointers must be valid for the configured descriptors, pointer mode, and
/// stream until the operation completes. Output pointer `d` and `workspace`, when provided, must
/// be writable for the required ranges.
///
/// # Errors
///
/// Returns an error if the descriptors, pointer mode, algorithm, workspace, or device do not
/// support the requested operation, or if execution fails.
pub unsafe fn matmul_raw(
    context: &Context,
    desc: &MatmulDescriptor,
    alpha: *const c_void,
    a: *const c_void,
    a_desc: &MatrixLayout,
    b: *const c_void,
    b_desc: &MatrixLayout,
    beta: *const c_void,
    c: *const c_void,
    c_desc: &MatrixLayout,
    d: *mut c_void,
    d_desc: &MatrixLayout,
    algorithm: Option<&MatmulAlgorithm>,
    workspace: Option<&mut dyn DeviceOutput<u8>>,
    stream: Option<&Stream>,
) -> Result<()> {
    if workspace
        .as_ref()
        .is_some_and(|workspace| workspace.context() != context.cuda_context())
    {
        return Err(CudaError::MemoryContextMismatch.into());
    }
    // SAFETY: the caller upholds this function's safety contract.
    let stream = unsafe { context.stream_raw(stream)? };
    let (workspace_ptr, workspace_size) = workspace.map_or((ptr::null_mut(), 0), |workspace| {
        (workspace.as_mut_ptr().cast(), workspace.len())
    });
    let workspace_size = to_u64(workspace_size, "workspace size")?;

    // SAFETY: `cublasLtMatmul` receives live handles owned by the wrappers.
    unsafe {
        try_ffi!(sys::cublasLtMatmul(
            context.as_raw(),
            desc.as_raw(),
            alpha,
            a,
            a_desc.as_raw(),
            b,
            b_desc.as_raw(),
            beta,
            c,
            c_desc.as_raw(),
            d,
            d_desc.as_raw(),
            algorithm.map_or(ptr::null(), |algorithm| algorithm.as_raw()),
            workspace_ptr,
            workspace_size,
            stream,
        ))?;
    }
    Ok(())
}

/// Submits a matrix multiplication using descriptor-defined pointer layouts.
///
/// # Safety
///
/// Matrix and scalar representations, accessible ranges, pointer modes, and
/// layouts must match the descriptors. Inputs, outputs, and workspace must
/// remain live and correctly ordered until the selected stream completes.
/// Output and workspace writes must not conflict with other access.
///
/// # Errors
///
/// Returns an error if the workspace or stream belongs to another CUDA context, binding the context fails, or the workspace size cannot be represented as `u64`.
/// Also returns an error if cuBLASLt rejects the descriptors, algorithm, or workspace, or cannot execute the multiplication.
pub unsafe fn matmul<A, B, C, D, Alpha, Beta>(
    context: &Context,
    desc: &MatmulDescriptor,
    alpha: &Alpha,
    a: &A,
    a_desc: &MatrixLayout,
    b: &B,
    b_desc: &MatrixLayout,
    beta: &Beta,
    c: &C,
    c_desc: &MatrixLayout,
    d: &mut D,
    d_desc: &MatrixLayout,
    algorithm: Option<&MatmulAlgorithm>,
    workspace: Option<&mut dyn DeviceOutput<u8>>,
    stream: Option<&Stream>,
) -> Result<()>
where
    A: ConstDataPointer,
    B: ConstDataPointer,
    C: ConstDataPointer,
    D: MutDataPointer,
    Alpha: ConstDataPointer,
    Beta: ConstDataPointer,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        matmul_raw(
            context,
            desc,
            alpha.as_const_data_ptr(),
            a.as_const_data_ptr(),
            a_desc,
            b.as_const_data_ptr(),
            b_desc,
            beta.as_const_data_ptr(),
            c.as_const_data_ptr(),
            c_desc,
            d.as_mut_data_ptr(),
            d_desc,
            algorithm,
            workspace,
            stream,
        )
    }
}

/// Computes a matrix transform operation.
///
/// `C = alpha*transformation(A) + beta*transformation(B),`
///
/// The transformation is defined by `desc` and may reorder matrix data or apply
/// scaling.
///
/// # Safety
///
/// Matrix and scalar pointers must be valid for the configured descriptors, pointer mode, and
/// stream until the operation completes. Output pointer `c` must be writable for the range
/// described by `c_desc`.
///
/// # Errors
///
/// Returns an error if the descriptors or parameters are incompatible with the configured
/// transform, if the device does not support it, or if execution fails.
pub(crate) unsafe fn matrix_transform_raw(
    context: &Context,
    desc: &MatrixTransformDescriptor,
    alpha: *const c_void,
    a: *const c_void,
    a_desc: &MatrixLayout,
    beta: *const c_void,
    b: *const c_void,
    b_desc: *const sys::cublasLtMatrixLayoutOpaque_t,
    c: *mut c_void,
    c_desc: &MatrixLayout,
    stream: Option<&Stream>,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    let stream = unsafe { context.stream_raw(stream)? };
    // SAFETY: `cublasLtMatrixTransform` receives live handles owned by the wrappers.
    unsafe {
        try_ffi!(sys::cublasLtMatrixTransform(
            context.as_raw(),
            desc.as_raw(),
            alpha,
            a,
            a_desc.as_raw(),
            beta,
            b,
            b_desc.cast_mut(),
            c,
            c_desc.as_raw(),
            stream,
        ))?;
    }
    Ok(())
}

/// Computes `C = alpha * op(A) + beta * op(B)` using the configured matrix layouts.
///
/// The descriptor selects transposition, scaling, and scalar pointer mode.
/// `b` supplies both the optional second matrix and its layout; omitting it passes null pointers to cuBLASLt.
///
/// # Errors
///
/// Returns an error if the stream belongs to another CUDA context, binding the context fails, or cuBLASLt rejects the layouts, scalar arguments, or transform operation.
pub fn matrix_transform<A, B, C, Alpha, Beta>(
    context: &Context,
    desc: &MatrixTransformDescriptor,
    alpha: &Alpha,
    a: &A,
    a_desc: &MatrixLayout,
    beta: &Beta,
    b: Option<(&B, &MatrixLayout)>,
    c: &mut C,
    c_desc: &MatrixLayout,
    stream: Option<&Stream>,
) -> Result<()>
where
    A: ConstDataPointer,
    B: ConstDataPointer,
    C: MutDataPointer,
    Alpha: ConstDataPointer,
    Beta: ConstDataPointer,
{
    let (b_ptr, b_desc_ptr) = b.map_or((ptr::null(), ptr::null()), |(matrix, layout)| {
        (matrix.as_const_data_ptr(), layout.as_raw().cast_const())
    });

    // SAFETY: the descriptors and matrices were validated above and stay borrowed for the call.
    unsafe {
        matrix_transform_raw(
            context,
            desc,
            alpha.as_const_data_ptr(),
            a.as_const_data_ptr(),
            a_desc,
            beta.as_const_data_ptr(),
            b_ptr,
            b_desc_ptr,
            c.as_mut_data_ptr(),
            c_desc,
            stream,
        )
    }
}

/// Returns algorithm IDs that may be usable with [`matmul`] for the given data types.
///
/// # Errors
///
/// Returns an error if `requested_count` cannot be represented as `i32` or if cuBLASLt rejects
/// the query.
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

    let requested_count = to_i32(requested_count, "requested algorithm count")?;
    let mut ids = vec![0; requested_count as usize];
    let mut actual_count = 0;

    // SAFETY: `cublasLtMatmulAlgoGetIds` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cublasLtMatmulAlgoGetIds(
            context.as_raw(),
            compute_type.into(),
            scale_type.into(),
            a_type.into(),
            b_type.into(),
            c_type.into(),
            d_type.into(),
            requested_count,
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
/// Returns an error if the algorithm ID is invalid or unsupported for the supplied data type
/// combination.
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
    // SAFETY: `cublasLtMatmulAlgoInit` receives live handles owned by the wrappers and valid output locations.
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

/// Returns heuristic algorithm candidates for [`matmul`].
///
/// Results are ordered by increasing estimated compute time.
///
/// # Errors
///
/// Returns an error if `requested_count` cannot be represented as `i32`, if no heuristic is
/// available for the configuration, or if cuBLASLt rejects the query.
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

    let requested_count = to_i32(requested_count, "requested heuristic count")?;
    let mut actual_count = 0;
    let mut heuristics = Vec::with_capacity(requested_count as usize);
    heuristics.resize_with(
        requested_count as usize,
        sys::cublasLtMatmulHeuristicResult_t::default,
    );

    // SAFETY: `cublasLtMatmulAlgoGetHeuristic` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cublasLtMatmulAlgoGetHeuristic(
            context.as_raw(),
            desc.as_raw(),
            a_desc.as_raw(),
            b_desc.as_raw(),
            c_desc.as_raw(),
            d_desc.as_raw(),
            preference.as_raw(),
            requested_count,
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

/// Checks whether a [`MatmulAlgorithm`] is valid for the supplied descriptors on the current
/// device.
///
/// On success, returns the required workspace size and estimated wave count reported by cuBLASLt.
///
/// # Errors
///
/// Returns an error if the descriptors do not match the algorithm configuration or if the device
/// does not support the requested algorithm.
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
    // SAFETY: `cublasLtMatmulAlgoCheck` receives live handles owned by the wrappers and valid output locations.
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

/// Takes a CUDA stream into account when evaluating algorithm compatibility.
///
/// If the stream is associated with a green context, the result reflects its
/// SM count and resource constraints, not the full device.
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

    let requested_count = to_i32(requested_count, "requested heuristic count")?;
    let mut actual_count = 0;
    let mut heuristics = Vec::with_capacity(requested_count as usize);
    heuristics.resize_with(
        requested_count as usize,
        sys::cublasLtMatmulHeuristicResult_t::default,
    );

    // SAFETY: `cublasLtMatmulAlgoGetHeuristicForStream` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(sys::cublasLtMatmulAlgoGetHeuristicForStream(
            context.as_raw(),
            desc.as_raw(),
            a_desc.as_raw(),
            b_desc.as_raw(),
            c_desc.as_raw(),
            d_desc.as_raw(),
            preference.as_raw(),
            requested_count,
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

/// Takes a CUDA stream into account when evaluating algorithm compatibility.
///
/// If the stream is associated with a green context, the result reflects its
/// SM count and resource constraints, not the full device.
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
    // SAFETY: `cublasLtMatmulAlgoCheckForStream` receives live handles owned by the wrappers and valid output locations.
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
