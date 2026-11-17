use std::{
    any::Any,
    fmt::{self, Formatter},
    mem, ptr,
    result::Result as StdResult,
    sync::Arc,
};

use singe_cuda_sys::runtime;

use super::{CaptureCleanup, PER_THREAD_STREAM, Stream, StreamIdentity, StreamInner};
use crate::{
    context::Context,
    error::{Error, Result},
    try_ffi,
};

/// Release responsibility returned with a raw native stream.
pub enum RawStreamOwnership {
    /// The caller must eventually destroy the returned native handle.
    Owned,
    /// The external owner token must remain live through every use of the native handle.
    Borrowed(Arc<dyn Any + Send + Sync>),
}

impl fmt::Debug for RawStreamOwnership {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Owned => "Owned",
            Self::Borrowed(_) => "Borrowed",
        })
    }
}

/// Raw ownership of a CUDA stream, its context, and any external owner.
///
/// Destroy owned handles explicitly, keep borrowed tokens alive through every use, or re-adopt the value with [`Stream::from_raw`].
#[derive(Debug)]
pub struct RawStream {
    /// Native stream handle whose release responsibility is described by `ownership`.
    pub handle: runtime::cudaStream_t,
    /// Actual CUDA context retained for native stream operations.
    pub context: Arc<Context>,
    /// Native destruction responsibility or external owner token.
    pub ownership: RawStreamOwnership,
}

pub(crate) struct ForeignOwner(Arc<dyn Any + Send + Sync>);

impl fmt::Debug for ForeignOwner {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ForeignOwner")
            .field("references", &Arc::strong_count(&self.0))
            .finish_non_exhaustive()
    }
}

impl Stream {
    /// Adopts a raw stream with the release responsibility recorded in `raw.ownership`.
    ///
    /// An owned handle is synchronized and destroyed when the last clone drops; a borrowed handle keeps its owner token instead.
    /// Use [`Stream::shutdown`] to surface synchronization or destruction errors explicitly.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be a valid CUDA stream of `raw.context`.
    /// An owned handle must not be destroyed elsewhere; a borrowed handle must stay valid while its owner token lives.
    ///
    /// # Errors
    ///
    /// Returns `Error::NullHandle` if `raw.handle` is null.
    pub unsafe fn from_raw(raw: RawStream) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }

        let foreign_owner = match raw.ownership {
            RawStreamOwnership::Owned => None,
            RawStreamOwnership::Borrowed(owner) => Some(ForeignOwner(owner)),
        };
        Ok(Self {
            inner: Arc::new(StreamInner {
                handle: raw.handle,
                capture_cleanup: CaptureCleanup::for_stream(StreamIdentity::new(
                    &raw.context,
                    raw.handle,
                ))?,
                context: raw.context,
                foreign_owner,
            }),
        })
    }

    /// Borrows a stream while retaining the library object that owns its native handle.
    /// Default legacy streams are accepted; per-thread default streams cannot be shared across threads.
    ///
    /// # Safety
    ///
    /// The handle must belong to `context` and remain valid while `owner` is retained.
    /// External work must obey this wrapper's access ordering and capture lifecycle.
    ///
    /// # Errors
    ///
    /// Returns `Error::ThreadLocalStreamUnsupported` for the per-thread default-stream sentinel.
    /// Also propagates context binding errors.
    /// Other handle validity and external ownership requirements remain the caller's responsibility.
    pub unsafe fn borrow_with_owner(
        handle: runtime::cudaStream_t,
        context: Arc<Context>,
        owner: Arc<dyn Any + Send + Sync>,
    ) -> Result<Self> {
        if handle.addr() == PER_THREAD_STREAM {
            return Err(Error::ThreadLocalStreamUnsupported);
        }

        context.bind()?;
        Ok(Self {
            inner: Arc::new(StreamInner {
                handle,
                capture_cleanup: CaptureCleanup::for_stream(StreamIdentity::new(&context, handle))?,
                context,
                foreign_owner: Some(ForeignOwner(owner)),
            }),
        })
    }

    #[cfg(feature = "interop_cudarc")]
    pub(crate) fn release_foreign_owner<T: Any + Send + Sync>(&self, owner: Arc<T>) -> Result<()> {
        use super::StreamCaptureStatus;
        use crate::utility::lock_for_cleanup;

        let mut cleanup = lock_for_cleanup(&self.inner.capture_cleanup);
        cleanup.foreign.push(ForeignOwner(owner));
        if self.capture_status()? == StreamCaptureStatus::None {
            cleanup.flush_frees(self.as_raw())?;
        }

        drop(cleanup);
        Ok(())
    }
}

impl Stream {
    /// Transfers the native handle, context and destruction responsibility to the caller.
    /// Destroy owned handles explicitly; retain borrowed ownership tokens through every use.
    /// Returns the unchanged stream if another clone or pending capture cleanup still retains it.
    ///
    /// # Errors
    ///
    /// Returns `Err(self)` if another stream clone retains the wrapper or shared capture cleanup still holds deferred owners, frees, or foreign tokens.
    /// The returned wrapper remains usable and retains its release responsibility.
    pub fn into_raw(self) -> StdResult<RawStream, Self> {
        let mut inner = match Arc::try_unwrap(self.inner) {
            Ok(inner) => inner,
            Err(inner) => return Err(Self { inner }),
        };
        // A poisoned lifecycle may hold deferred work, so its stream stays owned.
        if !inner
            .capture_cleanup
            .lock()
            .is_ok_and(|cleanup| cleanup.is_empty())
        {
            return Err(Self {
                inner: Arc::new(inner),
            });
        }

        let handle = mem::replace(&mut inner.handle, ptr::null_mut());
        let ownership = match inner.foreign_owner.take() {
            Some(ForeignOwner(owner)) => RawStreamOwnership::Borrowed(owner),
            None => RawStreamOwnership::Owned,
        };
        Ok(RawStream {
            handle,
            context: Arc::clone(&inner.context),
            ownership,
        })
    }

    pub(crate) fn destroy_handle(context: &Context, handle: runtime::cudaStream_t) -> Result<()> {
        context.bind()?;
        // SAFETY: the caller passes the owned stream it is destroying, live in `context`, which is bound above.
        let sync_result = unsafe { try_ffi!(runtime::cudaStreamSynchronize(handle)) };
        // SAFETY: the stream is owned here, nothing else destroys it, and the synchronization above drained its work.
        let destroy_result = unsafe { try_ffi!(runtime::cudaStreamDestroy(handle)) };

        match (sync_result, destroy_result) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(error), _) | (Ok(()), Err(error)) => Err(error),
        }
    }
}
