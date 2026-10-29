use std::{mem, ptr, sync::Arc};

use singe_cuda::{
    context::Context as CudaContext,
    stream::{BorrowedStream, Stream, StreamBinding},
    view::DeviceRepr,
};

use crate::{
    error::{Error, Result},
    scalar::{Scalar, ScalarMut},
    sys, try_ffi,
    types::PointerMode,
};

/// A stateful cuSPARSE handle.
///
/// Use one context per host thread or concurrent task. The handle is movable
/// between threads, but it is intentionally not `Clone` or `Sync`.
#[derive(Debug)]
pub struct Context {
    handle: Handle,
}

#[derive(Debug)]
struct Handle {
    raw: sys::cusparseHandle_t,
    streams: Vec<Stream>,
    cuda_context: Arc<CudaContext>,
}

/// A transferred native handle, its actual CUDA context, and retained stream owners.
/// Destroy the handle before releasing these owners, or re-adopt the complete value.
#[derive(Debug)]
pub struct RawContext {
    /// Native cuSPARSE handle whose ownership accompanies these retained resources.
    pub handle: sys::cusparseHandle_t,
    /// Actual CUDA context required for native handle operations and destruction.
    pub cuda_context: Arc<CudaContext>,
    /// Stream owners retained while the native handle may reference them.
    pub streams: Vec<Stream>,
}

// SAFETY: cuSPARSE handles carry mutable pointer-mode and stream state. Ownership can
// move between threads, but the wrapper requires exclusive access for mutation.
unsafe impl Send for Handle {}

impl Context {
    /// Initializes the cuSPARSE library and creates a cuSPARSE handle.
    /// The handle must be created before using other cuSPARSE operations through this wrapper.
    /// It allocates hardware resources needed to access the GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, if cuSPARSE cannot
    /// create a handle, or if cuSPARSE returns a null handle.
    pub fn create(cuda_context: &Arc<CudaContext>) -> Result<Self> {
        cuda_context.bind()?;

        let mut handle = ptr::null_mut();
        // SAFETY: `cusparseCreate` receives valid output locations.
        unsafe {
            try_ffi!(sys::cusparseCreate(&raw mut handle))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle: Handle {
                raw: handle,
                streams: Vec::new(),
                cuda_context: Arc::clone(cuda_context),
            },
        })
    }

    /// Wraps an existing cuSPARSE handle and takes ownership of it.
    ///
    /// # Safety
    ///
    /// `parts.handle` must be a uniquely owned cuSPARSE handle associated with
    /// `parts.cuda_context`. Include every nondefault stream used by outstanding
    /// work, including the currently selected stream, in `parts.streams`.
    /// Complete outstanding uses before transferring ownership to a different
    /// execution protocol. No other owner may destroy the transferred handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle is null, retained streams belong to another CUDA context, or a nondefault native stream lacks a matching retained stream owner. Also propagates CUDA context-binding and native stream-query errors.
    pub unsafe fn from_raw(parts: RawContext) -> Result<Self> {
        if parts.handle.is_null() {
            return Err(Error::NullHandle);
        }
        let context = Self {
            handle: Handle {
                raw: parts.handle,
                cuda_context: parts.cuda_context,
                streams: parts.streams,
            },
        };
        for stream in &context.handle.streams {
            context.ensure_stream(stream)?;
        }
        context.bind()?;
        let mut native_stream = ptr::null_mut();
        // SAFETY: `cusparseGetStream` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cusparseGetStream(
                context.as_raw(),
                &raw mut native_stream
            ))?;
        }
        if !native_stream.is_null()
            && !context
                .handle
                .streams
                .iter()
                .any(|stream| stream.as_raw() == native_stream)
        {
            return Err(Error::StreamContextMismatch);
        }
        Ok(context)
    }

    /// Returns the underlying CUDA context used by this cuSPARSE handle.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.handle.cuda_context
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
    ///
    /// # Errors
    ///
    /// Returns an error if the stream belongs to another CUDA context or binding this context fails.
    pub fn ensure_stream(&self, stream: &Stream) -> Result<()> {
        if self.cuda_context() != stream.context() {
            return Err(Error::StreamContextMismatch);
        }

        self.bind()
    }

    /// Returns the stream used for cuSPARSE operations on this handle.
    /// If no explicit stream has been set, cuSPARSE uses the CUDA default stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if cuSPARSE
    /// cannot report the current stream.
    pub fn stream(&self) -> Result<StreamBinding> {
        self.bind()?;

        let mut stream = ptr::null_mut();
        // SAFETY: `cusparseGetStream` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cusparseGetStream(self.as_raw(), &raw mut stream))?;
        }

        if let Some(owner) = self
            .handle
            .streams
            .iter()
            .find(|owner| owner.as_raw() == stream)
        {
            return Ok(StreamBinding::Borrowed(owner.to_borrowed()));
        }
        Ok(if stream.is_null() {
            StreamBinding::Default(Arc::clone(self.cuda_context()))
        } else {
            // SAFETY: the library reports the stream bound to this handle, which the handle keeps alive.
            StreamBinding::Borrowed(unsafe {
                BorrowedStream::from_raw(stream, Arc::clone(self.cuda_context()))
            })
        })
    }

    /// Sets the stream used by cuSPARSE operations on this handle.
    /// Previous stream owners remain retained until this context is dropped,
    /// including when native stream selection returns an error.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to another CUDA context, if the CUDA
    /// context cannot be bound, or if cuSPARSE rejects the stream.
    pub fn set_stream(&mut self, stream: Option<&Stream>) -> Result<()> {
        if let Some(stream) = stream {
            self.ensure_stream(stream)?;
        } else {
            self.bind()?;
        }

        if let Some(stream) = stream
            && !self
                .handle
                .streams
                .iter()
                .any(|owner| owner.as_raw() == stream.as_raw())
        {
            self.handle.streams.push(stream.clone());
        }
        // SAFETY: `cusparseSetStream` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cusparseSetStream(
                self.as_raw(),
                stream.map_or_else(ptr::null_mut, Stream::as_raw),
            ))?;
        }
        Ok(())
    }

    /// Returns the context-global scalar pointer mode used by cuSPARSE operations on this handle.
    /// See [`PointerMode`] for scalar pointer semantics.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if cuSPARSE
    /// cannot report the pointer mode.
    pub fn scalar_pointer_mode(&self) -> Result<PointerMode> {
        self.bind()?;

        let mut mode = sys::cusparsePointerMode_t::CUSPARSE_POINTER_MODE_HOST;
        // SAFETY: `cusparseGetPointerMode` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cusparseGetPointerMode(self.as_raw(), &raw mut mode))?;
        }
        Ok(mode.into())
    }

    /// Sets the context-global scalar pointer mode used by cuSPARSE operations on this handle.
    /// The default mode reads scalar values from host memory.
    /// See [`PointerMode`] for scalar pointer semantics.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or if cuSPARSE
    /// rejects the pointer mode.
    pub fn set_scalar_pointer_mode(&mut self, mode: PointerMode) -> Result<()> {
        self.bind()?;
        // SAFETY: `cusparseSetPointerMode` receives live handles owned by the wrappers.
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

    pub(crate) fn require_output_scalar_pointer_mode<T: DeviceRepr>(
        &self,
        result: &ScalarMut<'_, T>,
    ) -> Result<()> {
        result.validate_context(self.cuda_context())?;
        if self.scalar_pointer_mode()? != result.pointer_mode() {
            // SAFETY: raw operations serialize this non-Sync handle and select the result's ABI.
            unsafe {
                try_ffi!(sys::cusparseSetPointerMode(
                    self.as_raw(),
                    result.pointer_mode().into()
                ))?;
            }
        }
        Ok(())
    }

    /// Returns the raw cuSPARSE handle.
    ///
    /// The returned handle is borrowed and remains valid only while this
    /// context and its underlying CUDA context are alive.
    #[must_use]
    pub fn as_raw(&self) -> sys::cusparseHandle_t {
        self.handle.raw
    }

    /// Consumes the context and returns the raw cuSPARSE handle without
    /// destroying it.
    ///
    /// Destroy the transferred handle before releasing its context and retained
    /// stream owners, or re-adopt the complete [`RawContext`].
    #[must_use]
    pub fn into_raw(mut self) -> RawContext {
        RawContext {
            handle: mem::replace(&mut self.handle.raw, ptr::null_mut()),
            cuda_context: Arc::clone(&self.handle.cuda_context),
            streams: mem::take(&mut self.handle.streams),
        }
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if self.raw.is_null() {
            return;
        }
        if let Err(error) = self.cuda_context.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind cuda context before destroying cusparse handle: {error}");
            return;
        }

        // SAFETY: `cusparseDestroy` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cusparseDestroy(self.raw)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cusparse context: {error}");
            }
        }
    }
}
