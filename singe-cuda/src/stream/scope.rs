use std::{marker::PhantomData, result::Result as StdResult, sync::Arc};

use singe_cuda_sys::runtime;

use super::{AsStream, Stream, StreamBinding, StreamCaptureStatus, sealed};
use crate::{
    context::Context,
    error::{Error, Result, Status},
    submission::{Access, Execution, Resources},
};

/// A borrowed stream scope that drains queued work before borrowed storage can be released.
///
/// Constructed by [`Stream::sync_scope`].
/// Cleanup also drains during unwinding and aborts the process if completion cannot be established.
#[derive(Debug)]
pub struct StreamScope<'scope, 'env> {
    stream: &'scope Stream,
    _env: PhantomData<&'env mut &'env ()>,
}

struct ScopeDrain {
    stream: StreamBinding,
    drained: bool,
}

impl Drop for ScopeDrain {
    fn drop(&mut self) {
        // Returning or unwinding would release borrowed storage that CUDA might still access.
        // Unlike owned submissions, a scope cannot retain or leak its caller's stack storage after it exits.
        if !self.drained {
            self.stream.synchronize_or_abort();
        }
    }
}

impl Stream {
    /// Runs `f` with a stream scope and synchronizes this stream before returning.
    ///
    /// Use this for scoped asynchronous operations that borrow host or device memory until stream completion.
    /// For CUDA graph capture, use [`Stream::capture`] or [`Stream::capture_executable`].
    /// The completion wait also runs when `f` unwinds.
    /// If CUDA completion cannot be established, the process aborts before returning borrowed storage to the caller.
    ///
    /// # Errors
    ///
    /// Propagates capture-state query errors, rejects active or invalidated capture, and returns errors from `f` or stream synchronization.
    /// If both `f` and synchronization fail and cleanup can drain the stream, the closure error takes precedence.
    pub fn sync_scope<'env, F, R>(&self, f: F) -> Result<R>
    where
        F: for<'scope> FnOnce(&'scope StreamScope<'scope, 'env>) -> Result<R>,
    {
        let scope = StreamScope {
            stream: self,
            _env: PhantomData,
        };
        self.complete(|| f(&scope))
    }

    /// Runs `f`, then waits for this stream even when `f` fails or unwinds, so borrowed operands outlive the work `f` enqueues.
    ///
    /// Capture is rejected before `f` runs.
    /// This is [`StreamBinding::sync_scope`] on a borrowed binding of this stream.
    ///
    /// # Errors
    ///
    /// Converts capture-query, context binding, capture rejection, and stream synchronization errors into `E`.
    /// Errors returned by `f` take precedence over a synchronization error.
    pub fn complete<R, E>(&self, f: impl FnOnce() -> StdResult<R, E>) -> StdResult<R, E>
    where
        E: From<Error>,
    {
        StreamBinding::Borrowed(self.to_borrowed()).sync_scope(f)
    }
}

impl<'scope> StreamScope<'scope, '_> {
    /// Enqueues work on the scope's stream after declaring the accesses it makes.
    ///
    /// The declared accesses order the work after conflicting tracked work and make later tracked work wait for it.
    /// Wrapper `_scoped` functions use this to record their operands' accesses.
    ///
    /// # Safety
    ///
    /// `enqueue` must only enqueue work on [`Self::stream`] that touches the declared accesses, in the declared modes.
    /// Borrowed operands must be `'env` borrows, which the scope keeps alive until it drains.
    ///
    /// # Errors
    ///
    /// Propagates access-tracking, dependency-ordering, and completion-event errors, and errors returned by `enqueue`.
    pub unsafe fn enqueue<T>(
        &self,
        accesses: &[Access],
        enqueue: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let mut resources = Resources::new();
        resources.accesses.extend_from_slice(accesses);
        let mut execution = Execution::create(self.stream)?;
        // Caller borrows stay alive until the scope drains, including error and unwind paths.
        // SAFETY: the caller guarantees `enqueue` touches only the declared accesses on this scope's stream, and the scope keeps its `'env` borrows alive until it drains.
        let output = unsafe { execution.enqueue(resources, |_| enqueue())? };
        execution.finish(())?;
        Ok(output)
    }

    /// Returns the stream borrowed by this synchronous scope.
    #[must_use]
    pub const fn stream(&self) -> &'scope Stream {
        self.stream
    }
}

impl sealed::Sealed for StreamScope<'_, '_> {}

impl AsStream for StreamScope<'_, '_> {
    fn context(&self) -> &Arc<Context> {
        self.stream.context()
    }

    fn as_raw(&self) -> runtime::cudaStream_t {
        self.stream.as_raw()
    }
}

impl StreamBinding {
    /// Completes borrowed work on this stream before returning or unwinding.
    ///
    /// Capture is rejected before calling `f`.
    /// Default and externally borrowed streams use the same completion contract as owned streams.
    /// If completion cannot be established, the process aborts before releasing borrowed storage.
    ///
    /// # Errors
    ///
    /// Converts capture-query, context binding, capture rejection, and stream synchronization errors into `E`.
    /// Errors returned by `f` pass through after cleanup drains the stream and take precedence over a synchronization error if both occur.
    pub fn sync_scope<R, E>(&self, f: impl FnOnce() -> StdResult<R, E>) -> StdResult<R, E>
    where
        E: From<Error>,
    {
        match self.capture_status()? {
            StreamCaptureStatus::None => {}
            StreamCaptureStatus::Active => {
                return Err(Error::from(Status::StreamCaptureUnsupported).into());
            }
            StreamCaptureStatus::Invalidated => {
                return Err(Error::from(Status::StreamCaptureInvalidated).into());
            }
        }

        let mut drain = ScopeDrain {
            stream: self.clone(),
            drained: false,
        };
        let result = f();
        let completion = self.synchronize();
        drain.drained = completion.is_ok();
        drop(drain);
        match (result, completion) {
            (Ok(value), Ok(())) => Ok(value),
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error.into()),
        }
    }
}
