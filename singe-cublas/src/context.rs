use std::{convert::Infallible, ptr, sync::Arc};

use singe_cublas_sys as sys;
use singe_cuda::{
    context::Context as CudaContext,
    error::Error as CudaError,
    handle::{CreateHandle, Handle, NativeHandle, RawHandle, StreamHandle},
    stream::{Stream, StreamBinding},
    sys::runtime,
    types::{EmulationMantissaControl, EmulationSpecialValuesSupport},
    view::{DeviceOutputView, DeviceRegion},
};

use crate::{
    error::{Error, Result},
    try_ffi,
    types::{AtomicsMode, EmulationStrategy, MathMode, PointerMode},
    utility::{to_i32, to_size_t, to_u32},
};

/// Marker type for the native cuBLAS handle owned by [`Context`].
#[derive(Debug)]
pub struct CublasHandle(Infallible);

// SAFETY: cuBLAS handles may be used and destroyed from any thread, one call at a time.
unsafe impl NativeHandle for CublasHandle {
    type Raw = sys::cublasHandle_t;

    type State = ();

    type Error = Error;

    const OBJECT: &'static str = "cublas context";

    fn null_handle() -> Error {
        Error::NullHandle
    }

    fn is_null(raw: Self::Raw) -> bool {
        raw.is_null()
    }

    unsafe fn destroy(raw: Self::Raw, (): ()) -> Result<()> {
        // SAFETY: `NativeHandle::destroy` requires a live, uniquely owned handle with its context bound, so nothing uses it afterwards.
        unsafe { try_ffi!(sys::cublasDestroy_v2(raw)) }
    }
}

// SAFETY: `cublasCreate_v2` binds to the current context.
unsafe impl CreateHandle for CublasHandle {
    unsafe fn create(raw: *mut Self::Raw) -> Result<()> {
        // SAFETY: `CreateHandle::create` requires a writable `raw` and a bound CUDA context, which is all handle creation needs.
        unsafe { try_ffi!(sys::cublasCreate_v2(raw)) }
    }
}

// SAFETY: `cublasSetStream_v2` only installs the stream; `Handle` retains it.
unsafe impl StreamHandle for CublasHandle {
    unsafe fn set_stream(raw: Self::Raw, stream: runtime::cudaStream_t) -> Result<()> {
        // SAFETY: the caller passes a live handle and a live stream of its bound context.
        unsafe { try_ffi!(sys::cublasSetStream_v2(raw, stream)) }
    }
}

/// Raw ownership of a cuBLAS handle, its CUDA context, and its installed stream.
pub type RawContext = RawHandle<CublasHandle>;

/// A stateful cuBLAS handle.
///
/// Use one context per host thread or concurrent task.
/// The handle is movable between threads, but it is not `Clone` or `Sync`.
#[derive(Debug)]
pub struct Context {
    handle: Handle<CublasHandle>,
}

impl Context {
    /// Initializes cuBLAS and creates a handle on the legacy default stream.
    ///
    /// The handle is tied to the device of `cuda_context`; create one context per device.
    /// Creating and destroying a handle allocates resources and may synchronize the device, so keep contexts for the lifetime of a thread or task.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, cuBLAS cannot allocate the handle, or cuBLAS returns a null handle.
    pub fn create(cuda_context: &Arc<CudaContext>) -> Result<Self> {
        Ok(Self {
            handle: Handle::create(cuda_context)?,
        })
    }

    /// Adopts a cuBLAS handle with its CUDA context and installed stream.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be a uniquely owned cuBLAS handle created in `raw.cuda_context`, with `raw.stream` installed on it.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle is null or the stream belongs to another CUDA context.
    pub unsafe fn from_raw(raw: RawContext) -> Result<Self> {
        Ok(Self {
            // SAFETY: the caller guarantees `raw.handle` is a uniquely owned cuBLAS handle created in `raw.cuda_context` with `raw.stream` installed.
            handle: unsafe { Handle::from_raw(raw)? },
        })
    }

    /// Transfers the native handle, its CUDA context, and its installed stream to the caller.
    ///
    /// Destroy the transferred handle before releasing its context and stream, or re-adopt the complete [`RawContext`].
    #[must_use]
    pub fn into_raw(self) -> RawContext {
        self.handle.into_raw()
    }

    /// Returns the raw cuBLAS handle.
    ///
    /// The returned handle is borrowed and remains valid only while this context is alive.
    #[must_use]
    pub fn as_raw(&self) -> sys::cublasHandle_t {
        self.handle.as_raw()
    }

    /// Returns the CUDA context of this handle.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        self.handle.cuda_context()
    }

    /// Makes the handle's CUDA context current on the calling thread.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound.
    pub fn bind(&self) -> Result<()> {
        self.handle.bind()
    }

    /// Checks that `stream` belongs to this handle's CUDA context, then binds the context.
    ///
    /// # Errors
    ///
    /// Returns [`Error::StreamContextMismatch`] if `stream` belongs to another CUDA context, or an error if binding the context fails.
    pub fn ensure_stream(&self, stream: &Stream) -> Result<()> {
        if self.cuda_context() != stream.context() {
            return Err(Error::StreamContextMismatch);
        }

        self.bind()
    }

    /// Returns the stream installed on this handle, or `None` for the legacy default stream.
    #[must_use]
    pub fn stream(&self) -> Option<&Stream> {
        self.handle.stream()
    }

    /// Returns the stream this handle's operations run on.
    #[must_use]
    pub fn stream_binding(&self) -> StreamBinding {
        self.handle.stream_binding()
    }

    /// Installs `stream` for operations on this handle and retains it, or the legacy default stream for `None`.
    ///
    /// cuBLAS also resets the workspace to its default pool, so a workspace installed with [`Context::set_workspace`] is no longer used.
    /// On an error the handle keeps its previous stream.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context, the CUDA context cannot be bound, or cuBLAS rejects the stream.
    pub fn set_stream(&mut self, stream: Option<&Stream>) -> Result<()> {
        self.handle.set_stream(stream)
    }

    /// Runs `f`, then waits for this handle's stream, even when `f` fails.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream is capturing, `f` fails, or the wait fails.
    pub fn sync_scope<R>(&self, f: impl FnOnce() -> Result<R>) -> Result<R> {
        self.handle.sync_scope(f)
    }

    /// Orders later work on this handle's stream after the producers of these regions.
    ///
    /// # Errors
    ///
    /// Returns an error if a region belongs to another CUDA context or the dependencies cannot be enqueued.
    pub fn prepare_regions(&self, regions: &[DeviceRegion<'_>]) -> Result<()> {
        self.handle.prepare_regions(regions)
    }

    // Orders the handle's stream after the producers of the operands and of the device scalars among `scalars`.
    pub(crate) fn prepare_operands(
        &self,
        regions: &[DeviceRegion<'_>],
        scalars: &[Option<DeviceRegion<'_>>],
    ) -> Result<()> {
        let regions = regions
            .iter()
            .copied()
            .chain(scalars.iter().flatten().copied())
            .collect::<Vec<_>>();
        self.prepare_regions(&regions)
    }

    // Selects the pointer mode the scalars of one operation share.
    // The mode is transient state of this handle, which is not `Sync`, so the operation that follows reads it.
    pub(crate) fn use_pointer_mode(&self, modes: &[PointerMode]) -> Result<()> {
        let Some((&mode, rest)) = modes.split_first() else {
            return Ok(());
        };
        if rest.iter().any(|&other| other != mode) {
            return Err(Error::ScalarPointerModeMismatch);
        }

        // SAFETY: the handle is live and its context is bound by the operation.
        unsafe { try_ffi!(sys::cublasSetPointerMode_v2(self.as_raw(), mode.into())) }
    }

    /// Installs a caller-owned workspace for operations on the current stream, or no workspace for `None`.
    ///
    /// Without a call, cuBLAS uses a default pool allocated with the handle.
    /// `None` keeps cuBLAS from using the default pool.
    /// [`Context::set_stream`] resets the workspace to the default pool.
    /// A workspace of at least 16 KiB avoids [`Status::AllocFailed`](crate::error::Status::AllocFailed); larger workspaces can be faster.
    ///
    /// # Safety
    ///
    /// The workspace must stay live and exclusively used by this handle until it is replaced, the stream is changed, or the handle is destroyed, and until all work and graph replays that use it complete.
    ///
    /// # Errors
    ///
    /// Returns `CudaError::MemoryContextMismatch` if the workspace belongs to another CUDA context, an error if it is not aligned to 256 bytes or too large for cuBLAS, or an error if binding the context or cuBLAS fails.
    pub unsafe fn set_workspace(
        &mut self,
        workspace: Option<DeviceOutputView<'_, u8>>,
    ) -> Result<()> {
        self.bind()?;
        let (workspace_ptr, workspace_size) = if let Some(workspace) = workspace {
            let region = workspace.into_region();
            if region.context() != self.cuda_context() {
                return Err(CudaError::MemoryContextMismatch.into());
            }

            if !region.as_ptr().addr().is_multiple_of(256) {
                return Err(CudaError::Misaligned {
                    name: "cublas workspace".into(),
                    alignment: 256,
                }
                .into());
            }
            (region.as_ptr().cast(), region.byte_len())
        } else {
            (ptr::null_mut(), 0_usize)
        };
        let workspace_size = to_size_t(workspace_size, "workspace size")?;

        // SAFETY: the handle is live in the bound context, and the caller keeps the workspace live and exclusive.
        unsafe {
            try_ffi!(sys::cublasSetWorkspace_v2(
                self.as_raw(),
                workspace_ptr,
                workspace_size,
            ))
        }
    }

    /// Returns whether operations with an alternate atomic implementation may use it.
    ///
    /// A new context returns [`AtomicsMode::NotAllowed`].
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or cuBLAS cannot report the mode.
    pub fn atomics_mode(&self) -> Result<AtomicsMode> {
        self.bind()?;

        let mut mode = sys::cublasAtomicsMode_t::CUBLAS_ATOMICS_NOT_ALLOWED;
        // SAFETY: this context owns the live handle, its CUDA context was bound above, and the output is a live local.
        unsafe {
            try_ffi!(sys::cublasGetAtomicsMode(self.as_raw(), &raw mut mode))?;
        }
        Ok(mode.try_into()?)
    }

    /// Allows or disallows atomics in operations that have an alternate atomic implementation.
    ///
    /// Atomic implementations of some symmetric and Hermitian matrix-vector products are faster but not bitwise reproducible from run to run.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or cuBLAS rejects the mode.
    pub fn set_atomics_mode(&mut self, mode: AtomicsMode) -> Result<()> {
        self.bind()?;
        // SAFETY: this context owns the live handle, its CUDA context was bound above, and the setting is passed by value.
        unsafe { try_ffi!(sys::cublasSetAtomicsMode(self.as_raw(), mode.into())) }
    }

    /// Returns the math mode used by operations on this handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or cuBLAS cannot report the mode.
    pub fn math_mode(&self) -> Result<MathMode> {
        self.bind()?;

        let mut mode = sys::cublasMath_t::CUBLAS_DEFAULT_MATH.0;
        // SAFETY: the handle is live, and the output location holds the enum's 32-bit integer.
        unsafe {
            try_ffi!(sys::cublasGetMathMode(
                self.as_raw(),
                (&raw mut mode).cast(),
            ))?;
        }
        Ok(MathMode::from_bits_retain(mode))
    }

    /// Sets the compute precision mode, a combination of [`MathMode`] flags.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or cuBLAS rejects the mode.
    pub fn set_math_mode(&mut self, mode: MathMode) -> Result<()> {
        self.bind()?;
        // SAFETY: the handle is live, and the newtype `cublasMath_t` carries any flag combination.
        unsafe {
            try_ffi!(sys::cublasSetMathMode(
                self.as_raw(),
                sys::cublasMath_t(mode.bits())
            ))
        }
    }

    /// Returns the multiprocessor count cuBLAS optimizes for, or `0` for the whole device.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or cuBLAS cannot report the count.
    pub fn sm_count_target(&self) -> Result<u32> {
        self.bind()?;

        let mut count = 0;
        // SAFETY: this context owns the live handle, its CUDA context was bound above, and the output is a live local.
        unsafe {
            try_ffi!(sys::cublasGetSmCountTarget(self.as_raw(), &raw mut count))?;
        }
        to_u32(count, "sm count target")
    }

    /// Sets the number of multiprocessors cuBLAS optimizes for, or `0` for the whole device.
    ///
    /// Use it when cuBLAS shares the device with concurrent kernels on other streams.
    /// The count must not exceed the device's multiprocessor count.
    ///
    /// # Errors
    ///
    /// Returns an error if `count` exceeds `i32`, the CUDA context cannot be bound, or cuBLAS rejects the count.
    pub fn set_sm_count_target(&mut self, count: u32) -> Result<()> {
        let count = to_i32(count, "sm count target")?;
        self.bind()?;
        // SAFETY: this context owns the live handle, its CUDA context was bound above, and the setting is passed by value.
        unsafe { try_ffi!(sys::cublasSetSmCountTarget(self.as_raw(), count)) }
    }

    /// Returns the floating-point emulation strategy of this handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or cuBLAS cannot report the strategy.
    pub fn emulation_strategy(&self) -> Result<EmulationStrategy> {
        self.bind()?;

        let mut strategy = sys::cublasEmulationStrategy_t::CUBLAS_EMULATION_STRATEGY_DEFAULT;
        // SAFETY: this context owns the live handle, its CUDA context was bound above, and the output is a live local.
        unsafe {
            try_ffi!(sys::cublasGetEmulationStrategy(
                self.as_raw(),
                &raw mut strategy,
            ))?;
        }
        Ok(strategy.try_into()?)
    }

    /// Selects how cuBLAS uses floating-point emulation.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or cuBLAS rejects `strategy`.
    pub fn set_emulation_strategy(&mut self, strategy: EmulationStrategy) -> Result<()> {
        self.bind()?;
        // SAFETY: this context owns the live handle, its CUDA context was bound above, and the setting is passed by value.
        unsafe {
            try_ffi!(sys::cublasSetEmulationStrategy(
                self.as_raw(),
                strategy.into(),
            ))
        }
    }

    /// Returns the special values that emulated operations support.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or cuBLAS cannot report the mask.
    pub fn emulation_special_values_support(&self) -> Result<EmulationSpecialValuesSupport> {
        self.bind()?;

        let mut mask = EmulationSpecialValuesSupport::NONE.into();
        // SAFETY: this context owns the live handle, its CUDA context was bound above, and the output is a live local.
        unsafe {
            try_ffi!(sys::cublasGetEmulationSpecialValuesSupport(
                self.as_raw(),
                &raw mut mask,
            ))?;
        }
        Ok(mask.into())
    }

    /// Sets the special values that emulated operations support.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or cuBLAS rejects the mask.
    pub fn set_emulation_special_values_support(
        &mut self,
        support: EmulationSpecialValuesSupport,
    ) -> Result<()> {
        self.bind()?;
        // SAFETY: this context owns the live handle, its CUDA context was bound above, and the setting is passed by value.
        unsafe {
            try_ffi!(sys::cublasSetEmulationSpecialValuesSupport(
                self.as_raw(),
                support.into(),
            ))
        }
    }

    /// Returns the fixed-point emulation mantissa control.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or cuBLAS cannot report the control.
    pub fn emulation_mantissa_control(&self) -> Result<EmulationMantissaControl> {
        self.bind()?;

        let mut control = EmulationMantissaControl::Dynamic.into();
        // SAFETY: this context owns the live handle, its CUDA context was bound above, and the output is a live local.
        unsafe {
            try_ffi!(sys::cublasGetFixedPointEmulationMantissaControl(
                self.as_raw(),
                &raw mut control,
            ))?;
        }
        Ok(control.try_into()?)
    }

    /// Sets the fixed-point emulation mantissa control.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or cuBLAS rejects `control`.
    pub fn set_emulation_mantissa_control(
        &mut self,
        control: EmulationMantissaControl,
    ) -> Result<()> {
        self.bind()?;
        // SAFETY: this context owns the live handle, its CUDA context was bound above, and the setting is passed by value.
        unsafe {
            try_ffi!(sys::cublasSetFixedPointEmulationMantissaControl(
                self.as_raw(),
                control.into(),
            ))
        }
    }

    /// Returns the maximum mantissa bit count for fixed-point emulation.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or cuBLAS cannot report the count.
    pub fn fixed_point_emulation_max_mantissa_bit_count(&self) -> Result<i32> {
        self.bind()?;

        let mut count = 0;
        // SAFETY: this context owns the live handle, its CUDA context was bound above, and the output is a live local.
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
    /// Returns an error if the CUDA context cannot be bound or cuBLAS rejects `count`.
    pub fn set_fixed_point_emulation_max_mantissa_bit_count(&mut self, count: i32) -> Result<()> {
        self.bind()?;
        // SAFETY: this context owns the live handle, its CUDA context was bound above, and the setting is passed by value.
        unsafe {
            try_ffi!(sys::cublasSetFixedPointEmulationMaxMantissaBitCount(
                self.as_raw(),
                count,
            ))
        }
    }

    /// Returns the mantissa bit offset for fixed-point emulation.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or cuBLAS cannot report the offset.
    pub fn fixed_point_emulation_mantissa_bit_offset(&self) -> Result<i32> {
        self.bind()?;

        let mut offset = 0;
        // SAFETY: this context owns the live handle, its CUDA context was bound above, and the output is a live local.
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
    /// Returns an error if the CUDA context cannot be bound or cuBLAS rejects `offset`.
    pub fn set_fixed_point_emulation_mantissa_bit_offset(&mut self, offset: i32) -> Result<()> {
        self.bind()?;
        // SAFETY: this context owns the live handle, its CUDA context was bound above, and the setting is passed by value.
        unsafe {
            try_ffi!(sys::cublasSetFixedPointEmulationMantissaBitOffset(
                self.as_raw(),
                offset,
            ))
        }
    }

    /// Installs device storage where emulated operations write the mantissa bit count they used, or none for `None`.
    ///
    /// # Safety
    ///
    /// The storage must stay live and exclusively written by this handle's operations until it is replaced or the handle is destroyed, and until all work and graph replays that write it complete.
    ///
    /// # Errors
    ///
    /// Returns `CudaError::MemoryContextMismatch` if the storage belongs to another CUDA context, or an error if it is empty, binding the context fails, or cuBLAS rejects the pointer.
    pub unsafe fn set_fixed_point_emulation_mantissa_bit_count_pointer(
        &mut self,
        count: Option<DeviceOutputView<'_, i32>>,
    ) -> Result<()> {
        self.bind()?;
        let pointer = if let Some(mut count) = count {
            if count.context() != self.cuda_context() {
                return Err(CudaError::MemoryContextMismatch.into());
            }

            if count.is_empty() {
                return Err(Error::buffer_too_small("mantissa bit count", 1, 0));
            }
            count.as_mut_ptr()
        } else {
            ptr::null_mut()
        };
        // SAFETY: the handle is live in the bound context, and the caller keeps the storage live and exclusive.
        unsafe {
            try_ffi!(sys::cublasSetFixedPointEmulationMantissaBitCountPointer(
                self.as_raw(),
                pointer,
            ))
        }
    }
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use crate::{testing::setup_context, types::MathMode};

    #[test]
    fn test_math_mode_bit_combinations_roundtrip_through_the_integer_abi() {
        let mut test = setup_context().unwrap();
        let context = test.context_mut();
        let original = context.math_mode().unwrap();
        for mode in [
            MathMode::PEDANTIC | MathMode::DISALLOW_REDUCED_PRECISION_REDUCTION,
            MathMode::TF32_TENSOR_OP | MathMode::DISALLOW_REDUCED_PRECISION_REDUCTION,
            MathMode::DEFAULT,
        ] {
            context.set_math_mode(mode).unwrap();
            assert_eq!(context.math_mode().unwrap(), mode);
        }
        context.set_math_mode(original).unwrap();
    }
}
