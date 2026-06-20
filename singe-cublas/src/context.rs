use std::{ffi::CString, mem::ManuallyDrop, path::Path, ptr, sync::Arc};

use singe_cublas_sys as sys;
use singe_cuda::{
    context::Context as CudaContext,
    memory::DeviceMemory,
    stream::{BorrowedStream, Stream, StreamBinding},
    types::{EmulationMantissaControl, EmulationSpecialValuesSupport},
};

use crate::{
    error::{Error, Result},
    try_ffi,
    types::{AtomicsMode, EmulationStrategy, MathMode, PointerMode},
};

/// A stateful cuBLAS handle.
///
/// Use one context per host thread or concurrent task. The handle is movable
/// between threads, but it is intentionally not `Clone` or `Sync`.
#[derive(Debug)]
pub struct Context {
    handle: Handle,
}

#[derive(Debug)]
struct Handle {
    raw: sys::cublasHandle_t,
    cuda_ctx: Arc<CudaContext>,
}

// cuBLAS handles are stateful and not internally synchronized. Moving the
// owner between threads is allowed, but shared concurrent access is not.
unsafe impl Send for Handle {}

impl Context {
    /// Initializes cuBLAS and creates a library handle.
    /// Creating the handle allocates host and device resources and must happen before using other cuBLAS operations through this context.
    ///
    /// The cuBLAS library context is tied to the current CUDA device.
    /// To use the library on multiple devices, create one cuBLAS handle for each device.
    /// For a given device, multiple cuBLAS handles with different configurations can be created.
    /// For multi-threaded applications that use the same device from different threads, create one cuBLAS handle per thread and use that handle for the thread lifetime.
    ///
    /// Because [`Context::create`] allocates some internal resources and destroying the handle may implicitly synchronize the current CUDA device, it is best to minimize how often handles are created and destroyed.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA runtime initialization fails, if cuBLAS cannot
    /// allocate the required resources, or if cuBLAS does not return a valid
    /// handle.
    pub fn create(cuda_ctx: &Arc<CudaContext>) -> Result<Self> {
        cuda_ctx.bind()?;

        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cublasCreate_v2(&raw mut handle))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: Handle {
                raw: handle,
                cuda_ctx: Arc::clone(cuda_ctx),
            },
        })
    }

    /// Wraps an existing cuBLAS handle and takes ownership of it.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid cuBLAS handle associated with `cuda_ctx`.
    /// Ownership of `handle` is transferred to the returned context, and the
    /// handle must not be destroyed elsewhere after calling this function.
    pub unsafe fn from_raw(
        handle: sys::cublasHandle_t,
        cuda_ctx: Arc<CudaContext>,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: Handle {
                raw: handle,
                cuda_ctx,
            },
        })
    }

    /// Returns the underlying CUDA context used by this cuBLAS handle.
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.handle.cuda_ctx
    }

    /// Binds the underlying CUDA context associated with this handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound.
    pub fn bind(&self) -> Result<()> {
        Ok(self.cuda_context().bind()?)
    }

    /// Ensures `stream` belongs to the same CUDA context as this handle.
    ///
    /// Returns an error if the stream belongs to a different context.
    pub fn ensure_stream(&self, stream: &Stream) -> Result<()> {
        if self.cuda_context().as_ref() != stream.context() {
            return Err(Error::StreamContextMismatch);
        }

        self.bind()
    }

    /// Returns the cuBLAS library version.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if cuBLAS
    /// cannot report the version for this handle.
    pub fn version(&self) -> Result<i32> {
        self.bind()?;

        let mut version = 0;
        unsafe {
            try_ffi!(sys::cublasGetVersion_v2(self.as_raw(), &raw mut version))?;
        }
        Ok(version)
    }

    /// Sets the cuBLAS workspace to a caller-owned device buffer.
    ///
    /// Subsequent cuBLAS calls on the currently configured stream use this buffer.
    /// If no workspace is set, kernels use the default workspace pool allocated during
    /// context creation.
    /// Use this to change the workspace between kernel launches.
    /// The workspace pointer must be aligned to at least 256 bytes; otherwise cuBLAS returns [`crate::error::Status::InvalidValue`].
    /// [`Context::set_stream`] unconditionally resets the cuBLAS workspace back to the default workspace pool.
    /// Passing `None` prevents cuBLAS from using the default workspace.
    /// Too small a workspace may cause some operations to fail with [`crate::error::Status::AllocFailed`] or cause large regressions in performance.
    /// A workspace of at least 16 KiB is enough to prevent [`crate::error::Status::AllocFailed`], while
    /// larger workspaces can improve performance for some operations.
    ///
    /// The recommended caller-provided workspace size is based on the cuBLAS
    /// default workspace pool size, which depends on the GPU architecture.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, if `workspace`
    /// exceeds cuBLAS size limits or is not aligned to at least 256 bytes, or if
    /// cuBLAS reports that the library handle is not initialized.
    pub fn set_workspace(&self, workspace: Option<&mut DeviceMemory<u8>>) -> Result<()> {
        self.bind()?;

        let (workspace_ptr, workspace_size) = workspace
            .map_or((ptr::null_mut(), 0_usize), |workspace| {
                (workspace.as_mut_ptr().cast(), workspace.byte_len())
            });
        let workspace_size = u64::try_from(workspace_size).map_err(|_| Error::OutOfRange {
            name: "workspace size".into(),
        })?;

        unsafe {
            try_ffi!(sys::cublasSetWorkspace_v2(
                self.as_raw(),
                workspace_ptr,
                workspace_size,
            ))?;
        }
        Ok(())
    }

    /// Returns the stream used for cuBLAS calls on this context.
    ///
    /// If no stream is set, kernels use CUDA's default stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or cuBLAS reports
    /// that the library handle is not initialized.
    pub fn stream(&self) -> Result<StreamBinding> {
        self.bind()?;

        let mut stream = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cublasGetStream_v2(self.as_raw(), &raw mut stream))?;
        }

        Ok(if stream.is_null() {
            StreamBinding::Default(Arc::clone(self.cuda_context()))
        } else {
            StreamBinding::Borrowed(unsafe {
                BorrowedStream::from_raw(stream, Arc::clone(self.cuda_context()))
            })
        })
    }

    /// Sets the stream used for subsequent cuBLAS calls.
    ///
    /// Passing `None` makes kernels use CUDA's default stream.
    /// Use this to change the stream between kernel launches or reset the cuBLAS library stream to the default stream.
    /// This also unconditionally resets the cuBLAS workspace back to the default workspace
    /// pool. See [`Context::set_workspace`].
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or cuBLAS reports
    /// that the library handle is not initialized.
    pub fn set_stream(&self, stream: Option<&Stream>) -> Result<()> {
        if let Some(stream) = stream {
            self.ensure_stream(stream)?;
        } else {
            self.bind()?;
        }

        unsafe {
            try_ffi!(sys::cublasSetStream_v2(
                self.as_raw(),
                match stream {
                    Some(stream) => stream.as_raw(),
                    None => ptr::null_mut(),
                },
            ))?;
        }
        Ok(())
    }

    /// Returns the context-global scalar pointer mode used by cuBLAS.
    ///
    /// See [`PointerMode`] for details.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or cuBLAS reports
    /// that the library handle is not initialized.
    pub fn scalar_pointer_mode(&self) -> Result<PointerMode> {
        self.bind()?;

        let mut mode = sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST;
        unsafe {
            try_ffi!(sys::cublasGetPointerMode_v2(self.as_raw(), &raw mut mode))?;
        }
        Ok(mode.into())
    }

    /// Sets the context-global scalar pointer mode used by cuBLAS.
    ///
    /// The default mode passes scalar values by host reference. See [`PointerMode`] for
    /// details.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, if cuBLAS rejects
    /// `mode`, or if cuBLAS reports that the library handle is not initialized.
    pub fn set_scalar_pointer_mode(&self, mode: PointerMode) -> Result<()> {
        self.bind()?;
        unsafe {
            try_ffi!(sys::cublasSetPointerMode_v2(self.as_raw(), mode.into()))?;
        }
        Ok(())
    }

    pub(crate) fn require_host_pointer_mode(&self) -> Result<()> {
        if self.scalar_pointer_mode()? != PointerMode::Host {
            return Err(Error::RequiresHostPointerMode);
        }

        Ok(())
    }

    pub(crate) fn with_pointer_mode<T>(
        &self,
        mode: PointerMode,
        operation: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let previous = self.scalar_pointer_mode()?;
        if previous != mode {
            self.set_scalar_pointer_mode(mode)?;
        }

        let result = operation();
        let restore_result = if previous != mode {
            self.set_scalar_pointer_mode(previous)
        } else {
            Ok(())
        };

        match (result, restore_result) {
            (Ok(value), Ok(())) => Ok(value),
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error),
        }
    }

    /// Returns the atomics mode for this cuBLAS context.
    ///
    /// The default atomics mode of a newly created [`Context`] is [`AtomicsMode::NotAllowed`].
    /// See [`AtomicsMode`] for details.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or cuBLAS reports
    /// that the library handle is not initialized.
    pub fn atomics_mode(&self) -> Result<AtomicsMode> {
        self.bind()?;

        let mut mode = sys::cublasAtomicsMode_t::CUBLAS_ATOMICS_NOT_ALLOWED;
        unsafe {
            try_ffi!(sys::cublasGetAtomicsMode(self.as_raw(), &raw mut mode))?;
        }
        Ok(mode.into())
    }

    /// Some symmetric and Hermitian matrix-vector operations have an alternate implementation that uses atomics to accumulate results.
    /// This atomic implementation is generally significantly faster but can
    /// generate results that are not strictly identical from one run to the next.
    /// Mathematically, those differences are not significant, but they can complicate debugging.
    ///
    /// Allows or disallows atomics in cuBLAS operations that have an alternate atomic implementation.
    /// When a cuBLAS operation does not explicitly document atomics support, it does not have an alternate implementation that uses atomics.
    /// When atomics mode is disabled, each cuBLAS operation produces the same results from one run to the next when called with identical parameters on the same hardware.
    ///
    /// The default atomics mode of a newly created [`Context`] is [`AtomicsMode::NotAllowed`].
    /// See [`AtomicsMode`] for details.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or cuBLAS reports
    /// that the library handle is not initialized.
    pub fn set_atomics_mode(&self, mode: AtomicsMode) -> Result<()> {
        self.bind()?;
        unsafe {
            try_ffi!(sys::cublasSetAtomicsMode(self.as_raw(), mode.into()))?;
        }
        Ok(())
    }

    /// Returns the math mode used by cuBLAS operations.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or cuBLAS reports
    /// that the library handle is not initialized.
    pub fn math_mode(&self) -> Result<MathMode> {
        self.bind()?;

        let mut mode = sys::cublasMath_t::CUBLAS_DEFAULT_MATH as u32;
        unsafe {
            try_ffi!(sys::cublasGetMathMode(
                self.as_raw(),
                (&raw mut mode).cast(),
            ))?;
        }
        Ok(unsafe { MathMode::from_raw_bits(mode) })
    }

    /// Sets the compute precision mode.
    ///
    /// The mode can be a logical combination of [`MathMode`] values, except for the
    /// deprecated [`MathMode::TENSOR_OP`]. The default math mode is [`MathMode::DEFAULT`].
    ///
    /// For matrix and compute precisions allowed by [`gemm_ex`](crate::blas::level3::gemm_ex),
    /// its strided variants, and cuBLASLt matmul operations, see
    /// [`gemm_ex`](crate::blas::level3::gemm_ex),
    /// [`gemm_batched_ex`](crate::blas::level3::gemm_batched_ex), and
    /// [`gemm_strided_batched_ex`](crate::blas::level3::gemm_strided_batched_ex).
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or cuBLAS reports
    /// that the library handle is not initialized.
    pub fn set_math_mode(&self, mode: MathMode) -> Result<()> {
        self.bind()?;
        unsafe {
            try_ffi!(sys::cublasSetMathMode(self.as_raw(), mode.as_raw()))?;
        }
        Ok(())
    }

    /// Returns the SM count target previously set on this handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or cuBLAS reports
    /// that the library handle is not initialized.
    pub fn sm_count_target(&self) -> Result<i32> {
        self.bind()?;

        let mut count = 0;
        unsafe {
            try_ffi!(sys::cublasGetSmCountTarget(self.as_raw(), &raw mut count))?;
        }
        Ok(count)
    }

    /// Overrides the number of multiprocessors available to cuBLAS during kernel execution.
    ///
    /// Can improve library performance when cuBLAS operations run concurrently
    /// with other work on different CUDA streams.
    /// For example, on an NVIDIA A100 GPU with 108 multiprocessors, if another kernel
    /// is running concurrently with a grid size of 8, setting `count` to `100` asks the
    /// library heuristics to optimize for the remaining 100 multiprocessors.
    ///
    /// A value of `0` restores the default behavior.
    /// The input value must not exceed the device's multiprocessor count, available from the device properties.
    /// Negative values are not accepted.
    ///
    /// Callers must synchronize concurrent handle mutation just as they would for
    /// [`Context::set_stream`].
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, if `count` is
    /// outside the range accepted by cuBLAS, or if cuBLAS reports that the
    /// library handle is not initialized.
    pub fn set_sm_count_target(&self, count: i32) -> Result<()> {
        self.bind()?;

        unsafe {
            try_ffi!(sys::cublasSetSmCountTarget(self.as_raw(), count))?;
        }
        Ok(())
    }

    /// Returns the emulation strategy configured on this handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or cuBLAS reports
    /// that the library handle is not initialized.
    pub fn emulation_strategy(&self) -> Result<EmulationStrategy> {
        self.bind()?;

        let mut strategy = sys::cublasEmulationStrategy_t::CUBLAS_EMULATION_STRATEGY_DEFAULT;
        unsafe {
            try_ffi!(sys::cublasGetEmulationStrategy(
                self.as_raw(),
                &raw mut strategy,
            ))?;
        }
        Ok(strategy.into())
    }

    /// Selects how cuBLAS uses floating-point emulation.
    ///
    /// See [`EmulationStrategy`] for details.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, if cuBLAS rejects
    /// `strategy`, or if cuBLAS reports that the library handle is not initialized.
    pub fn set_emulation_strategy(&self, strategy: EmulationStrategy) -> Result<()> {
        self.bind()?;
        unsafe {
            try_ffi!(sys::cublasSetEmulationStrategy(
                self.as_raw(),
                strategy.into(),
            ))?;
        }
        Ok(())
    }

    /// Returns the special-values support configured for emulation.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or cuBLAS reports
    /// that the library handle is not initialized.
    pub fn emulation_special_values_support(&self) -> Result<EmulationSpecialValuesSupport> {
        self.bind()?;

        let mut mask = EmulationSpecialValuesSupport::NONE.into();
        unsafe {
            try_ffi!(sys::cublasGetEmulationSpecialValuesSupport(
                self.as_raw(),
                &raw mut mask,
            ))?;
        }
        Ok(mask.into())
    }

    /// Sets the special-values support used for emulation.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, if cuBLAS rejects
    /// the support mask, or if cuBLAS reports that the library handle is not
    /// initialized.
    pub fn set_emulation_special_values_support(
        &self,
        support: EmulationSpecialValuesSupport,
    ) -> Result<()> {
        self.bind()?;
        unsafe {
            try_ffi!(sys::cublasSetEmulationSpecialValuesSupport(
                self.as_raw(),
                support.into(),
            ))?;
        }
        Ok(())
    }

    /// Returns the fixed-point emulation mantissa control.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or cuBLAS reports
    /// that the library handle is not initialized.
    pub fn emulation_mantissa_control(&self) -> Result<EmulationMantissaControl> {
        self.bind()?;

        let mut control = EmulationMantissaControl::Dynamic.into();
        unsafe {
            try_ffi!(sys::cublasGetFixedPointEmulationMantissaControl(
                self.as_raw(),
                &raw mut control,
            ))?;
        }
        Ok(control.into())
    }

    /// Sets the fixed-point emulation mantissa control.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, if cuBLAS rejects
    /// `control`, or if cuBLAS reports that the library handle is not initialized.
    pub fn set_emulation_mantissa_control(&self, control: EmulationMantissaControl) -> Result<()> {
        self.bind()?;
        unsafe {
            try_ffi!(sys::cublasSetFixedPointEmulationMantissaControl(
                self.as_raw(),
                control.into(),
            ))?;
        }
        Ok(())
    }

    /// Returns the maximum mantissa bit count for fixed-point emulation.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or cuBLAS reports
    /// that the library handle is not initialized.
    pub fn fixed_point_emulation_max_mantissa_bit_count(&self) -> Result<i32> {
        self.bind()?;

        let mut count = 0;
        unsafe {
            try_ffi!(sys::cublasGetFixedPointEmulationMaxMantissaBitCount(
                self.as_raw(),
                &raw mut count,
            ))?;
        }
        Ok(count)
    }

    /// Sets the maximum mantissa bit count for fixed-point emulation.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, if cuBLAS rejects
    /// `count`, or if cuBLAS reports that the library handle is not initialized.
    pub fn set_fixed_point_emulation_max_mantissa_bit_count(&self, count: i32) -> Result<()> {
        self.bind()?;

        unsafe {
            try_ffi!(sys::cublasSetFixedPointEmulationMaxMantissaBitCount(
                self.as_raw(),
                count,
            ))?;
        }
        Ok(())
    }

    /// Returns the mantissa bit offset for fixed-point emulation.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or cuBLAS reports
    /// that the library handle is not initialized.
    pub fn fixed_point_emulation_mantissa_bit_offset(&self) -> Result<i32> {
        self.bind()?;

        let mut offset = 0;
        unsafe {
            try_ffi!(sys::cublasGetFixedPointEmulationMantissaBitOffset(
                self.as_raw(),
                &raw mut offset,
            ))?;
        }
        Ok(offset)
    }

    /// Sets the mantissa bit offset for fixed-point emulation.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, if cuBLAS rejects
    /// `offset`, or if cuBLAS reports that the library handle is not initialized.
    pub fn set_fixed_point_emulation_mantissa_bit_offset(&self, offset: i32) -> Result<()> {
        self.bind()?;

        unsafe {
            try_ffi!(sys::cublasSetFixedPointEmulationMantissaBitOffset(
                self.as_raw(),
                offset,
            ))?;
        }
        Ok(())
    }

    /// Sets the pointer to the mantissa bit count for fixed-point emulation.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, if cuBLAS rejects
    /// `count`, or if cuBLAS reports that the library handle is not initialized.
    ///
    /// # Safety
    ///
    /// `count` must be null or point to storage that remains valid for every
    /// cuBLAS operation that may read the configured mantissa bit count.
    pub unsafe fn set_fixed_point_emulation_mantissa_bit_count_pointer(
        &self,
        count: *mut i32,
    ) -> Result<()> {
        self.bind()?;
        unsafe {
            try_ffi!(sys::cublasSetFixedPointEmulationMantissaBitCountPointer(
                self.as_raw(),
                count,
            ))?;
        }
        Ok(())
    }

    /// Returns the custom logger callback installed with [`Context::set_logger_callback`], if any.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLAS cannot report the current logger callback.
    pub fn logger_callback() -> Result<sys::cublasLogCallback> {
        let mut callback = None;
        unsafe {
            try_ffi!(sys::cublasGetLoggerCallback(&raw mut callback))?;
        }
        Ok(callback)
    }

    /// Installs a custom logger callback through cuBLAS.
    ///
    /// # Safety
    ///
    /// `callback`, if present, must remain valid for use by cuBLAS and must follow
    /// the callback ABI expected by the library.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLAS rejects the callback.
    pub unsafe fn set_logger_callback(callback: sys::cublasLogCallback) -> Result<()> {
        unsafe {
            try_ffi!(sys::cublasSetLoggerCallback(callback))?;
        }
        Ok(())
    }

    /// Configures cuBLAS logging at runtime.
    /// Logging can also be configured with environment variables checked by `libcublas`.
    ///
    /// # Errors
    ///
    /// Returns an error if `file` contains an interior NUL byte or if cuBLAS
    /// rejects the logging configuration.
    pub fn configure_logger(
        enabled: bool,
        stdout: bool,
        stderr: bool,
        file: Option<&Path>,
    ) -> Result<()> {
        let file = file
            .map(|file| CString::new(file.as_os_str().to_string_lossy().as_bytes()))
            .transpose()?;

        unsafe {
            try_ffi!(sys::cublasLoggerConfigure(
                enabled.into(),
                stdout.into(),
                stderr.into(),
                file.as_ref().map_or(ptr::null(), |file| file.as_ptr()),
            ))?;
        }
        Ok(())
    }

    /// Returns the raw cuBLAS handle.
    ///
    /// The returned handle is borrowed and remains valid only while this
    /// context and its underlying CUDA context are alive.
    pub fn as_raw(&self) -> sys::cublasHandle_t {
        self.handle.raw
    }

    /// Consumes the context and returns the raw cuBLAS handle without
    /// destroying it.
    ///
    /// The caller becomes responsible for eventually destroying the returned
    /// handle with cuBLAS.
    pub fn into_raw(self) -> sys::cublasHandle_t {
        let context = ManuallyDrop::new(self);
        context.handle.raw
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if let Err(err) = self.cuda_ctx.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind cuda context before destroying cublas handle: {err}");
        }

        unsafe {
            if let Err(err) = try_ffi!(sys::cublasDestroy_v2(self.raw)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cublas context: {err}");
            }
        }
    }
}
