use std::{convert::Infallible, sync::Arc};

use singe_cuda::{
    context::Context as CudaContext,
    handle::{CreateHandle, Handle, NativeHandle, RawHandle, StreamHandle},
    stream::{Stream, StreamBinding},
    sys::runtime,
    view::{DeviceRegion, DeviceRepr},
};

use crate::{
    error::{Error, Result},
    scalar::Scalar,
    sys, try_ffi,
    types::PointerMode,
};

/// Marker type for the native cuSPARSE handle owned by [`Context`].
#[derive(Debug)]
pub struct CusparseHandle(Infallible);

// SAFETY: cuSPARSE handles may be used and destroyed from any thread, one call at a time.
unsafe impl NativeHandle for CusparseHandle {
    type Raw = sys::cusparseHandle_t;

    type State = ();

    type Error = Error;

    const OBJECT: &'static str = "cusparse context";

    fn null_handle() -> Error {
        Error::NullHandle
    }

    fn is_null(raw: Self::Raw) -> bool {
        raw.is_null()
    }

    unsafe fn destroy(raw: Self::Raw, (): ()) -> Result<()> {
        // SAFETY: the caller passes a live, uniquely owned handle with its context bound.
        unsafe { try_ffi!(sys::cusparseDestroy(raw)) }
    }
}

// SAFETY: `cusparseCreate` binds to the current context.
unsafe impl CreateHandle for CusparseHandle {
    unsafe fn create(raw: *mut Self::Raw) -> Result<()> {
        // SAFETY: the caller provides a writable output location and binds the context.
        unsafe { try_ffi!(sys::cusparseCreate(raw)) }
    }
}

// SAFETY: `cusparseSetStream` only installs the stream; `Handle` retains it.
unsafe impl StreamHandle for CusparseHandle {
    unsafe fn set_stream(raw: Self::Raw, stream: runtime::cudaStream_t) -> Result<()> {
        // SAFETY: the caller passes a live handle and a live stream of its bound context.
        unsafe { try_ffi!(sys::cusparseSetStream(raw, stream)) }
    }
}

/// Raw ownership of a cuSPARSE handle, its CUDA context, and its installed stream.
pub type RawContext = RawHandle<CusparseHandle>;

/// A stateful cuSPARSE handle.
///
/// Use one context per host thread or concurrent task.
/// The handle is movable between threads, but it is intentionally not `Clone` or `Sync`.
#[derive(Debug)]
pub struct Context {
    handle: Handle<CusparseHandle>,
}

impl Context {
    /// Initializes the cuSPARSE library and creates a cuSPARSE handle on the legacy default stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, if cuSPARSE cannot create a handle, or if cuSPARSE returns a null handle.
    pub fn create(cuda_context: &Arc<CudaContext>) -> Result<Self> {
        Ok(Self {
            handle: Handle::create(cuda_context)?,
        })
    }

    /// Adopts a cuSPARSE handle with its CUDA context and installed stream.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be a uniquely owned cuSPARSE handle created in `raw.cuda_context`, with `raw.stream` installed on it.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle is null or the stream belongs to another CUDA context.
    pub unsafe fn from_raw(raw: RawContext) -> Result<Self> {
        Ok(Self {
            // SAFETY: the caller guarantees `raw` holds a uniquely owned cuSPARSE handle created in its CUDA context with its stream installed.
            handle: unsafe { Handle::from_raw(raw)? },
        })
    }

    /// Returns the underlying CUDA context used by this cuSPARSE handle.
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

    /// Installs `stream` for cuSPARSE operations on this handle and retains it, or the legacy default stream for `None`.
    ///
    /// On an error the handle keeps its previous stream.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context, if the CUDA context cannot be bound, or if cuSPARSE rejects the stream.
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

    pub(crate) fn prepare_operands(&self, operands: &[&[Option<DeviceRegion<'_>>]]) -> Result<()> {
        let regions: Vec<DeviceRegion<'_>> = operands
            .iter()
            .flat_map(|regions| regions.iter().flatten().copied())
            .collect();
        self.prepare_regions(&regions)
    }

    /// Returns the context-global scalar pointer mode used by cuSPARSE operations on this handle.
    /// See [`PointerMode`] for scalar pointer semantics.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if cuSPARSE cannot report the pointer mode.
    pub fn scalar_pointer_mode(&self) -> Result<PointerMode> {
        self.bind()?;

        let mut mode = sys::cusparsePointerMode_t::CUSPARSE_POINTER_MODE_HOST;
        // SAFETY: `self` owns the live handle, its context is bound above, and `mode` is a writable local.
        unsafe {
            try_ffi!(sys::cusparseGetPointerMode(self.as_raw(), &raw mut mode))?;
        }
        Ok(mode.try_into()?)
    }

    /// Sets the context-global scalar pointer mode used by cuSPARSE operations on this handle.
    /// The default mode reads scalar values from host memory.
    /// See [`PointerMode`] for scalar pointer semantics.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if cuSPARSE rejects the pointer mode.
    pub fn set_scalar_pointer_mode(&mut self, mode: PointerMode) -> Result<()> {
        self.bind()?;
        // SAFETY: `self` owns the live handle, its context is bound above, and `&mut self` excludes concurrent use of it.
        unsafe {
            try_ffi!(sys::cusparseSetPointerMode(self.as_raw(), mode.into()))?;
        }
        Ok(())
    }

    pub(crate) fn require_scalar_pointer_mode<T: DeviceRepr>(
        &self,
        alpha: Scalar<'_, T>,
        beta: Scalar<'_, T>,
    ) -> Result<()> {
        alpha.validate_context(self.cuda_context())?;
        beta.validate_context(self.cuda_context())?;
        let alpha_mode = alpha.pointer_mode();
        let beta_mode = beta.pointer_mode();
        if alpha_mode != beta_mode {
            return Err(Error::ScalarPointerModeMismatch);
        }

        if self.scalar_pointer_mode()? != alpha_mode {
            // SAFETY: raw operations serialize this non-Sync handle and select matching scalar ABIs.
            unsafe {
                try_ffi!(sys::cusparseSetPointerMode(
                    self.as_raw(),
                    alpha_mode.into()
                ))?;
            }
        }
        Ok(())
    }

    /// Returns the raw cuSPARSE handle.
    ///
    /// The returned handle is borrowed and remains valid only while this context and its underlying CUDA context are alive.
    #[must_use]
    pub fn as_raw(&self) -> sys::cusparseHandle_t {
        self.handle.as_raw()
    }

    /// Consumes the context and returns the raw cuSPARSE handle, its CUDA context, and its stream without destroying the handle.
    ///
    /// Destroy the transferred handle before releasing its context and stream, or re-adopt the complete [`RawContext`].
    #[must_use]
    pub fn into_raw(self) -> RawContext {
        self.handle.into_raw()
    }
}
