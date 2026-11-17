//! Native cuSOLVER handle ownership and CUDA stream binding.

use std::{convert::Infallible, sync::Arc};

use singe_cuda::{
    context::Context as CudaContext,
    handle::{CreateHandle, Handle, NativeHandle, RawHandle, StreamHandle},
    stream::{Stream, StreamBinding},
    sys::runtime,
    types::EmulationStrategy,
    view::{DeviceRegion, DeviceRegionProvider},
};

use crate::{
    error::{Error, Result},
    sys, try_ffi,
    types::{DeterministicMode, MathMode},
};

/// Marker type for the native cuSOLVER dense handle owned by [`Context`].
#[derive(Debug)]
pub struct CusolverHandle(Infallible);

// SAFETY: cuSOLVER handles may be used and destroyed from any thread, one call at a time.
unsafe impl NativeHandle for CusolverHandle {
    type Raw = sys::cusolverDnHandle_t;

    type State = ();

    type Error = Error;

    const OBJECT: &'static str = "cusolver context";

    fn null_handle() -> Error {
        Error::NullHandle
    }

    fn is_null(raw: Self::Raw) -> bool {
        raw.is_null()
    }

    unsafe fn destroy(raw: Self::Raw, (): ()) -> Result<()> {
        // SAFETY: the caller passes a live, uniquely owned handle with its context bound.
        unsafe { try_ffi!(sys::cusolverDnDestroy(raw)) }
    }
}

// SAFETY: `cusolverDnCreate` binds to the current context.
unsafe impl CreateHandle for CusolverHandle {
    unsafe fn create(raw: *mut Self::Raw) -> Result<()> {
        // SAFETY: the caller provides a writable output location and binds the context.
        unsafe { try_ffi!(sys::cusolverDnCreate(raw)) }
    }
}

// SAFETY: `cusolverDnSetStream` only installs the stream; `Handle` retains it.
unsafe impl StreamHandle for CusolverHandle {
    unsafe fn set_stream(raw: Self::Raw, stream: runtime::cudaStream_t) -> Result<()> {
        // SAFETY: the caller passes a live handle and a live stream of its bound context.
        unsafe { try_ffi!(sys::cusolverDnSetStream(raw, stream)) }
    }
}

/// Raw ownership of a cuSOLVER dense handle, its CUDA context, and its installed stream.
pub type RawContext = RawHandle<CusolverHandle>;

/// A stateful cuSOLVER handle.
///
/// Use one context per host thread or concurrent task.
/// The handle is movable between threads, but it is intentionally not `Clone` or `Sync`.
#[derive(Debug)]
pub struct Context {
    handle: Handle<CusolverHandle>,
}

impl Context {
    /// Creates a cuSOLVER dense handle for the given CUDA context, on the legacy default stream.
    ///
    /// cuSOLVER allocates the GPU-side resources it needs here.
    /// On the first application-defined stream passed to [`Context::set_stream`], cuSOLVER may also allocate an internal workspace.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, if cuSOLVER cannot create a handle, or if cuSOLVER returns a null handle.
    pub fn create(cuda_context: &Arc<CudaContext>) -> Result<Self> {
        Ok(Self {
            handle: Handle::create(cuda_context)?,
        })
    }

    /// Returns the underlying CUDA context used by this cuSOLVER handle.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        self.handle.cuda_context()
    }

    /// Binds the underlying CUDA context associated with this handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound.
    pub fn bind(&self) -> Result<()> {
        self.handle.bind()
    }

    /// Returns the stream this handle's operations run on.
    #[must_use]
    pub fn stream_binding(&self) -> StreamBinding {
        self.handle.stream_binding()
    }

    /// Completes borrowed solver work on the installed stream on every path.
    /// Capture is rejected before `f` runs.
    /// If CUDA cannot establish completion, the process aborts before returning borrowed resources to their callers.
    ///
    /// # Errors
    ///
    /// Returns an error if stream capture is active or invalidated, CUDA binding or synchronization fails, or `f` returns an error.
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

    // Orders the handle's stream after the region's producer on the device, without waiting on the host.
    pub(crate) fn prepare_region(
        &self,
        value: &(impl DeviceRegionProvider + ?Sized),
    ) -> Result<()> {
        self.prepare_regions(&[value.region()])
    }

    /// Returns the stream installed on this handle, or `None` for the legacy default stream.
    #[must_use]
    pub fn stream(&self) -> Option<&Stream> {
        self.handle.stream()
    }

    /// Installs `stream` on this handle and retains it, or the legacy default stream for `None`.
    ///
    /// On an error the handle keeps its previous stream.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context, if the CUDA context cannot be bound, or if cuSOLVER rejects the stream.
    pub fn set_stream(&mut self, stream: Option<&Stream>) -> Result<()> {
        self.handle.set_stream(stream)
    }

    /// Returns the deterministic mode currently configured on this handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if cuSOLVER cannot report the deterministic mode.
    pub fn deterministic_mode(&self) -> Result<DeterministicMode> {
        self.bind()?;

        let mut mode = sys::cusolverDeterministicMode_t::CUSOLVER_DETERMINISTIC_RESULTS;
        // SAFETY: the context is bound, `self` owns the live handle, and the call writes one value to a local.
        unsafe {
            try_ffi!(sys::cusolverDnGetDeterministicMode(
                self.as_raw(),
                &raw mut mode,
            ))?;
        }
        Ok(mode.try_into()?)
    }

    /// Sets the deterministic mode for operations executed through this handle.
    ///
    /// Allowing non-deterministic results may improve performance for some operations, including [`xgeqrf_async`](crate::dense::xgeqrf_async), [`Gesvd`](crate::svd::Gesvd), [`Gesvdr`](crate::svd::Gesvdr), and [`Gesvdp`](crate::svd::Gesvdp).
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if cuSOLVER rejects the deterministic mode.
    pub fn set_deterministic_mode(&mut self, mode: DeterministicMode) -> Result<()> {
        self.bind()?;
        // SAFETY: the context is bound, and `self` owns the live handle and is borrowed mutably, so no operation uses it concurrently.
        unsafe {
            try_ffi!(sys::cusolverDnSetDeterministicMode(
                self.as_raw(),
                mode.into(),
            ))?;
        }
        Ok(())
    }

    /// Returns the math mode currently configured on this handle.
    ///
    /// See [`MathMode`] for the supported wrapper values.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, if cuSOLVER cannot report the math mode, or if it reports a mode [`MathMode`] does not name.
    pub fn math_mode(&self) -> Result<MathMode> {
        self.bind()?;

        // cuSOLVER may report OR-combined or newer modes, so read the value as an integer instead of the bindings' enum.
        let mut mode = 0_u32;
        // SAFETY: the context is bound, and `self` owns the live handle.
        // The output points to a `u32`, which has the size and alignment of the `repr(u32)` native enum.
        unsafe {
            try_ffi!(sys::cusolverDnGetMathMode(
                self.as_raw(),
                (&raw mut mode).cast::<sys::cusolverMathMode_t>(),
            ))?;
        }
        MathMode::try_from(mode).map_err(|_| Error::UnknownEnumValue {
            name: "math mode".into(),
            value: mode.into(),
        })
    }

    /// Sets the math mode for operations executed through this handle.
    ///
    /// See [`MathMode`] for the supported wrapper values and combinations.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if cuSOLVER rejects the math mode.
    pub fn set_math_mode(&mut self, mode: MathMode) -> Result<()> {
        self.bind()?;
        // SAFETY: the context is bound, and `self` owns the live handle and is borrowed mutably, so no operation uses it concurrently.
        unsafe {
            try_ffi!(sys::cusolverDnSetMathMode(self.as_raw(), mode.into()))?;
        }
        Ok(())
    }

    /// Returns the emulation strategy configured on this handle.
    ///
    /// This only affects operations that use one of the emulated math modes described by [`MathMode`].
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if cuSOLVER cannot report the emulation strategy.
    pub fn emulation_strategy(&self) -> Result<EmulationStrategy> {
        self.bind()?;

        let mut strategy = EmulationStrategy::Default.into();
        // SAFETY: the context is bound, `self` owns the live handle, and the call writes one value to a local.
        unsafe {
            try_ffi!(sys::cusolverDnGetEmulationStrategy(
                self.as_raw(),
                &raw mut strategy,
            ))?;
        }
        Ok(strategy.try_into()?)
    }

    /// Sets the emulation strategy for operations executed through this handle.
    ///
    /// This only affects operations that use one of the emulated math modes described by [`MathMode`].
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if cuSOLVER rejects the emulation strategy.
    pub fn set_emulation_strategy(&mut self, strategy: EmulationStrategy) -> Result<()> {
        self.bind()?;
        // SAFETY: the context is bound, and `self` owns the live handle and is borrowed mutably, so no operation uses it concurrently.
        unsafe {
            try_ffi!(sys::cusolverDnSetEmulationStrategy(
                self.as_raw(),
                strategy.into(),
            ))?;
        }
        Ok(())
    }

    /// Returns the raw cuSOLVER dense handle.
    ///
    /// The returned handle is borrowed and remains valid only while this context and its underlying CUDA context are alive.
    #[must_use]
    pub fn as_raw(&self) -> sys::cusolverDnHandle_t {
        self.handle.as_raw()
    }

    /// Adopts a cuSOLVER dense handle with its CUDA context and installed stream.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be valid and uniquely owned in `raw.cuda_context`, with `raw.stream` installed on it.
    /// Complete all borrowed work before releasing or replacing its resources.
    ///
    /// # Errors
    ///
    /// Returns an error if the native handle is null or the stream belongs to another CUDA context.
    pub unsafe fn from_raw(raw: RawContext) -> Result<Self> {
        Ok(Self {
            // SAFETY: the caller guarantees that `raw.handle` is live and uniquely owned in `raw.cuda_context`, with `raw.stream` installed.
            handle: unsafe { Handle::from_raw(raw)? },
        })
    }

    /// Transfers the native handle together with its CUDA context and installed stream.
    #[must_use]
    pub fn into_raw(self) -> RawContext {
        self.handle.into_raw()
    }
}
