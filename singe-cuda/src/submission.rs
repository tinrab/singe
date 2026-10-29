//! Owned CUDA operations, explicit-stream execution and range-based ordering.
//!
//! Submission retains native owners independently of outputs and futures.
//! Pending device values remain pending until the caller establishes successful completion.

mod completion;
mod operation;
mod resource;
mod tracker;
pub mod transfer;

use std::{
    future::Future,
    mem, panic,
    panic::AssertUnwindSafe,
    pin::Pin,
    result::Result as StdResult,
    sync::Arc,
    task::{Context, Poll},
};

pub use completion::Completion;
pub use operation::{DeviceOperation, Map, Operation, Pair, Shared, StreamPool, Then};
pub use resource::{Access, AccessMode, Resources};
pub(crate) use tracker::{AccessGuard, AccessTracker, lock_access};

use crate::{
    error::{Error, Result},
    event::{Event, EventFlags, EventRecordFlags},
    stream::Stream,
};

/// Retains every resource registered before an operation enqueues work.
/// On error or unwind, drains submitted work before releasing its ledger.
#[derive(Debug)]
pub struct Execution {
    stream: Stream,
    resources: Resources,
    events: Vec<Arc<Event>>,
    submitted: bool,
}

impl Execution {
    /// Starts an owned execution ledger on an explicit stream outside capture.
    ///
    /// # Errors
    /// Returns an error if capture is active or CUDA cannot query the stream.
    pub fn create(stream: &Stream) -> Result<Self> {
        stream.ensure_not_capturing()?;
        Ok(Self {
            stream: stream.clone(),
            resources: Resources::new(),
            events: Vec::new(),
            submitted: false,
        })
    }

    /// Returns the explicit stream used by this execution ledger.
    #[must_use]
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Registers all owners and accesses before invoking an asynchronous native operation.
    ///
    /// # Safety
    /// Retain every owner in the ledger or keep borrowed storage live until stream completion.
    /// Stream-ordered allocations may rely on deallocation ordered after this work.
    /// Owned [`Operation`] implementations must retain their resources independently of returned values.
    /// Declare every accessed byte range and uphold the native operation's bounds and value contracts.
    /// External raw access must participate in ordering or be excluded through completion.
    /// The closure must only enqueue work on the supplied stream and must not submit nested operations.
    ///
    /// # Errors
    ///
    /// Converts access-context mismatches, nested access, failed or poisoned tracking, dependency failures, stream-ordering errors, and completion-event creation or recording failures into `E`. Errors returned by `enqueue` pass through after completion is recorded. A later event-recording failure takes precedence over the callback result; native work may already be queued.
    ///
    /// # Panics
    ///
    /// Resumes unwinding if `enqueue` panics, after attempting to synchronize its submitted work. The execution ledger continues to retain its registered resources during cleanup.
    pub unsafe fn enqueue<T, E: From<Error>>(
        &mut self,
        mut resources: Resources,
        enqueue: impl FnOnce(&Stream) -> StdResult<T, E>,
    ) -> StdResult<T, E> {
        let context = Arc::clone(self.stream.context());
        for access in &resources.accesses {
            access.validate_context(&context)?;
        }
        let mut tracker = lock_access(&context.access_tracker, &resources.accesses)?;
        tracker.order(&self.stream, &resources.accesses)?;
        for dependency in &resources.dependencies {
            dependency.query()?;
            dependency.order_on(&self.stream)?;
        }
        let access_start = self.resources.accesses.len();
        tracker.retain(&mut resources);
        self.resources.extend(resources);
        let event = Arc::new(context.create_event_with_flags(EventFlags::DISABLE_TIMING)?);
        self.submitted = true;
        let result = match panic::catch_unwind(AssertUnwindSafe(|| enqueue(&self.stream))) {
            Ok(result) => result,
            Err(payload) => {
                if self.stream.synchronize().is_err() {
                    tracker.fail();
                }
                drop(tracker);
                panic::resume_unwind(payload);
            }
        };
        if let Err(error) = event.record(&self.stream, EventRecordFlags::DEFAULT) {
            if self.stream.synchronize().is_err() {
                tracker.fail();
            }
            return Err(error.into());
        }
        tracker.record(&self.resources.accesses[access_start..], &event);
        self.events.push(event);
        result
    }

    /// Completes current work before running host code that consumes its values.
    ///
    /// # Errors
    ///
    /// Returns an error if stream synchronization or waiting for any declared dependency fails. Dependencies are waited on in order, stopping at the first error.
    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()?;
        for dependency in &self.resources.dependencies {
            dependency.wait()?;
        }
        Ok(())
    }

    /// Transfers the ledger to checked completion independently of the returned output.
    ///
    /// # Errors
    /// Returns an error if CUDA cannot record completion or start its completion worker.
    pub fn finish<T>(self, output: T) -> Result<Submission<T>> {
        self.finish_with(output, |output| Ok(output))
    }

    fn finish_with<T, P>(
        mut self,
        pending: P,
        resolve: unsafe fn(P) -> Result<T>,
    ) -> Result<Submission<T, P>> {
        let event = if let Some(event) = self.events.last() {
            Arc::clone(event)
        } else {
            let event = Arc::new(
                self.stream
                    .context()
                    .create_event_with_flags(EventFlags::DISABLE_TIMING)?,
            );
            event.record(&self.stream, EventRecordFlags::DEFAULT)?;
            event
        };
        // Earlier events keep their access declarations live until the entire chain completes.
        for event in mem::take(&mut self.events) {
            self.resources.retain(event);
        }
        let resources = mem::take(&mut self.resources);
        let completion = Completion::start(event, self.stream.clone(), resources)?;
        self.submitted = false;
        Ok(Submission {
            completion,
            pending: Some(pending),
            resolve,
        })
    }
}

impl Drop for Execution {
    fn drop(&mut self) {
        if self.submitted
            && let Err(error) = self.stream.synchronize()
        {
            if let Ok(mut tracker) = lock_access(
                &self.stream.context().access_tracker,
                &self.resources.accesses,
            ) {
                tracker.fail();
            }
            mem::forget(mem::take(&mut self.resources));
            mem::forget(mem::take(&mut self.events));
            #[cfg(debug_assertions)]
            eprintln!("cuda submission did not complete; retaining live resources: {error}");
        }
    }
}

/// An owned result and checked completion.
/// Dropping this value never cancels CUDA work or releases its independent resource ledger.
#[derive(Debug)]
pub struct Submission<T, P = T> {
    completion: Completion,
    pending: Option<P>,
    resolve: unsafe fn(P) -> Result<T>,
}

impl<T, P> Submission<T, P> {
    /// Borrows the shared completion state without consuming the pending result.
    pub fn completion(&self) -> &Completion {
        &self.completion
    }

    /// Blocks for checked completion, then resolves and returns the operation's host result.
    ///
    /// Resolution runs on the calling thread after successful completion and resource retirement.
    ///
    /// # Errors
    ///
    /// Propagates completion, dependency, and resource-retirement errors. After successful completion, errors returned by the operation's resolver are propagated.
    ///
    /// # Panics
    ///
    /// Panics if this submission already yielded its output when polled as a future. A user-provided resolver or mapping callback can also panic.
    pub fn wait(mut self) -> Result<T> {
        self.completion.wait()?;
        // SAFETY: successful checked completion establishes the resolver's precondition.
        unsafe { (self.resolve)(self.pending.take().expect("submission already resolved")) }
    }
}

impl<T, P> Unpin for Submission<T, P> {}

impl<T, P> Future for Submission<T, P> {
    type Output = Result<T>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.completion).poll(cx) {
            Poll::Ready(Ok(())) => {
                let pending = self.pending.take().expect("submission already resolved");
                // SAFETY: successful checked completion establishes the resolver's precondition.
                Poll::Ready(unsafe { (self.resolve)(pending) })
            }
            Poll::Ready(Err(error)) => Poll::Ready(Err(error)),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Stream {
    /// Submits owned work with independent resource retention and checked completion.
    ///
    /// # Errors
    ///
    /// Propagates errors from the operation's [`Operation::submit_on`] implementation, including execution setup, enqueue, and completion registration failures. An error can occur after some native work has been queued; the execution ledger retains its resources through cleanup.
    pub fn submit<O: Operation>(&self, operation: O) -> Result<Submission<O::Output, O::Pending>> {
        operation.submit_on(self)
    }
}
