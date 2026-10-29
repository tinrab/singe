use std::{
    future::Future,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context as TaskContext, Poll, Waker},
};

use crate::{
    error::Result,
    event::{Event, EventFlags, EventRecordFlags},
    graph::ExecutableGraph,
    stream::{Stream, StreamScope},
    submission::{Completion, Execution, Resources, Submission},
};

#[derive(Debug)]
struct CompletionState {
    complete: AtomicBool,
    waker: Mutex<Option<Waker>>,
}

impl CompletionState {
    fn create() -> Self {
        Self {
            complete: AtomicBool::new(false),
            waker: Mutex::new(None),
        }
    }

    fn signal(&self) {
        self.complete.store(true, Ordering::Release);
        let waker = self.waker.lock().expect("completion waker poisoned").take();
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

/// A callback-backed notification that preceding stream work has reached a host function.
///
/// The future retains the stream and resolves to `()`, without reporting asynchronous CUDA errors. Dropping it does not cancel queued work or the callback. Use [`CheckedStreamFuture`] when completion must include CUDA status.
#[derive(Debug)]
pub struct StreamFuture {
    state: Arc<CompletionState>,
    _stream: Stream,
}

impl Future for StreamFuture {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Self::Output> {
        if self.state.complete.load(Ordering::Acquire) {
            return Poll::Ready(());
        }

        {
            let mut waker = self.state.waker.lock().expect("completion waker poisoned");
            *waker = Some(cx.waker().clone());
        }

        if self.state.complete.load(Ordering::Acquire) {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

impl Unpin for StreamFuture {}

/// Checked event-backed completion that observes CUDA errors even when callbacks do not run.
///
/// The completion worker retains the stream until the recorded work finishes.
/// Dropping the future does not cancel queued work. The future does not retain
/// owners of raw pointers or graph code; those lifetimes follow the submission API's contract.
#[derive(Debug)]
pub struct CheckedStreamFuture {
    completion: Completion,
}

impl Future for CheckedStreamFuture {
    type Output = Result<()>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Self::Output> {
        Pin::new(&mut self.completion).poll(cx)
    }
}

impl Stream {
    /// Returns a future that resolves when a host function enqueued at the current end of this stream runs.
    ///
    /// This is a notification primitive.
    /// It does not report asynchronous CUDA errors after registration.
    /// Use [`Stream::checked_completion_future`] or [`Stream::synchronize_async`] when the result must include CUDA status.
    ///
    /// # Errors
    ///
    /// Returns an error if capture is active or invalidated, stream capture state cannot be queried, the context cannot be bound, or CUDA cannot enqueue the notification callback. Errors after successful registration are not reported by the returned future.
    pub fn completion_future(&self) -> Result<StreamFuture> {
        self.ensure_not_capturing()?;

        let state = Arc::new(CompletionState::create());
        let callback_state = Arc::clone(&state);
        self.launch_host_func(move || callback_state.signal())?;

        Ok(StreamFuture {
            state,
            _stream: self.clone(),
        })
    }

    /// Returns a future that resolves with CUDA's asynchronous stream status.
    ///
    /// An event wait observes CUDA status independently of notification callbacks.
    /// Registration is rejected while stream capture is active.
    ///
    /// # Errors
    ///
    /// Returns an error if capture is active or invalidated, CUDA cannot query capture state, create or record the completion event, or bind the context, or the completion worker cannot be started or accept work. The returned future reports asynchronous CUDA completion errors.
    pub fn checked_completion_future(&self) -> Result<CheckedStreamFuture> {
        self.ensure_not_capturing()?;

        let event = Arc::new(
            self.context()
                .create_event_with_flags(EventFlags::DISABLE_TIMING)?,
        );
        event.record(self, EventRecordFlags::DEFAULT)?;
        Ok(CheckedStreamFuture {
            completion: Completion::start(event, self.clone(), Resources::new())?,
        })
    }

    /// Waits asynchronously for work preceding a newly recorded completion event.
    ///
    /// Registration occurs when the returned future is polled. Dropping the future after registration does not cancel stream work.
    ///
    /// # Errors
    ///
    /// Propagates errors from registering [`Self::checked_completion_future`] and from awaiting its checked CUDA result.
    pub async fn synchronize_async(&self) -> Result<()> {
        self.checked_completion_future()?.await
    }

    /// Submits arbitrary raw work with an explicitly supplied ownership ledger.
    ///
    /// # Safety
    /// The resources must retain every native owner and declare all accesses before `enqueue` runs.
    /// Follow [`crate::submission::Execution::enqueue`]'s native execution contract.
    ///
    /// # Errors
    ///
    /// Propagates capture-state, context, access-tracking, dependency-ordering, and completion-event errors, errors returned by `enqueue`, and failures to start or submit to the completion worker. The callback may already have enqueued work when a later step fails.
    pub unsafe fn enqueue_async_unchecked<T, F>(
        &self,
        resources: Resources,
        enqueue: F,
    ) -> Result<Submission<T>>
    where
        F: FnOnce(&Self) -> Result<T>,
    {
        let mut execution = Execution::create(self)?;
        // SAFETY: the caller upholds this function's safety contract.
        let output = unsafe { execution.enqueue(resources, enqueue)? };
        execution.finish(output)
    }
}

impl StreamScope<'_, '_> {
    /// Registers callback-backed completion on the scope's stream.
    ///
    /// The returned future does not extend the scope's storage borrows or cancel queued work when dropped.
    ///
    /// # Errors
    ///
    /// Propagates registration errors from [`Stream::completion_future`]. The returned future does not report asynchronous CUDA errors.
    pub fn completion_future(&self) -> Result<StreamFuture> {
        self.stream().completion_future()
    }

    /// Registers event-backed checked completion on the scope's stream.
    ///
    /// Dropping the future does not cancel queued work.
    ///
    /// # Errors
    ///
    /// Propagates registration errors from [`Stream::checked_completion_future`]. The returned future reports asynchronous CUDA completion errors.
    pub fn checked_completion_future(&self) -> Result<CheckedStreamFuture> {
        self.stream().checked_completion_future()
    }

    /// Waits asynchronously for preceding work on the scope's stream.
    ///
    /// Dropping the future does not cancel queued work or replace the scope's drain obligation.
    ///
    /// # Errors
    ///
    /// Propagates registration and checked completion errors from [`Stream::synchronize_async`].
    pub async fn synchronize_async(&self) -> Result<()> {
        self.stream().synchronize_async().await
    }
}

impl Event {
    /// Orders `stream` after this event, then registers callback-backed stream completion.
    ///
    /// The future also waits for work already queued on `stream`. It does not report asynchronous CUDA errors. Dropping it does not cancel the wait or queued work.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot enqueue the event wait or callback registration fails. The wait may already be queued when registration fails; see [`Stream::completion_future`].
    pub fn completion_future_on(&self, stream: &Stream) -> Result<StreamFuture> {
        stream.wait_event(self)?;
        stream.completion_future()
    }

    /// Orders `stream` after this event, then registers event-backed checked stream completion.
    ///
    /// The future also covers work already queued on `stream`. Dropping it does not cancel the wait or queued work.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot enqueue the event wait or checked completion registration fails. The wait may already be queued when registration fails; see [`Stream::checked_completion_future`]. The future reports asynchronous CUDA completion errors.
    pub fn checked_completion_future_on(&self, stream: &Stream) -> Result<CheckedStreamFuture> {
        stream.wait_event(self)?;
        stream.checked_completion_future()
    }

    /// Orders `stream` after this event and waits asynchronously for checked stream completion.
    ///
    /// The wait and completion registration occur when the future is polled. Dropping it does not cancel queued work.
    ///
    /// # Errors
    ///
    /// Propagates event-wait, completion registration, and asynchronous CUDA completion errors from [`Self::checked_completion_future_on`].
    pub async fn synchronize_async_on(&self, stream: &Stream) -> Result<()> {
        self.checked_completion_future_on(stream)?.await
    }
}

impl ExecutableGraph {
    /// Enqueues raw graph work and observes checked completion.
    ///
    /// # Safety
    /// Follow [`ExecutableGraph::launch`]'s ownership and access contract.
    ///
    /// # Errors
    ///
    /// Propagates graph launch errors and errors from registering [`Stream::checked_completion_future`]. A successful launch remains queued if completion registration subsequently fails. The returned future reports asynchronous CUDA completion errors.
    pub unsafe fn launch_async(&self, stream: &Stream) -> Result<CheckedStreamFuture> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            self.launch(stream)?;
        }
        stream.checked_completion_future()
    }

    /// Enqueues and waits for raw graph work.
    ///
    /// # Safety
    /// Follow [`ExecutableGraph::launch`]'s ownership and access contract, including future cancellation.
    ///
    /// # Errors
    ///
    /// Propagates graph launch, completion registration, and asynchronous CUDA completion errors from [`Self::launch_async`]. Dropping this future after launch does not cancel the queued graph.
    #[expect(
        clippy::future_not_send,
        reason = "raw graph mutation is thread confined; this future borrows the graph"
    )]
    pub async unsafe fn launch_and_wait(&self, stream: &Stream) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { self.launch_async(stream)? }.await
    }

    /// Enqueues this executable graph's upload and registers checked stream completion.
    ///
    /// The future covers the upload and earlier stream work. Dropping it does not cancel queued operations.
    ///
    /// # Errors
    ///
    /// Propagates graph upload errors and errors from registering [`Stream::checked_completion_future`]. The upload may already be queued when registration fails. The returned future reports asynchronous CUDA completion errors.
    pub fn upload_async(&self, stream: &Stream) -> Result<CheckedStreamFuture> {
        self.upload(stream)?;
        stream.checked_completion_future()
    }

    /// Uploads this executable graph and waits asynchronously for checked stream completion.
    ///
    /// Submission begins when the future is polled. Dropping it after submission does not cancel queued work.
    ///
    /// # Errors
    ///
    /// Propagates graph upload, completion registration, and asynchronous CUDA completion errors from [`Self::upload_async`].
    #[expect(
        clippy::future_not_send,
        reason = "raw graph mutation is thread confined; this future borrows the graph"
    )]
    pub async fn upload_and_wait(&self, stream: &Stream) -> Result<()> {
        self.upload_async(stream)?.await
    }
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use crate::{
        error::{Error, Result, Status},
        event::EventRecordFlags,
        memory::DeviceMemory,
        stream::StreamCaptureMode,
        testing,
    };

    #[tokio::test]
    async fn test_stream_future_resolves_after_empty_stream() -> Result<()> {
        let (_lock, context) = testing::bootstrap()?;
        let stream = context.create_stream()?;

        stream.completion_future()?.await;
        stream.checked_completion_future()?.await?;
        stream.synchronize_async().await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_stream_synchronize_async_waits_for_memset() -> Result<()> {
        let (_lock, context) = testing::bootstrap()?;
        let stream = context.create_stream()?;
        let mut device = DeviceMemory::<u8>::zeroed(&context, 16)?;

        // SAFETY: the test drains the stream before the memory is dropped or read.
        unsafe {
            device.set_value_async_unchecked(7, &stream)?;
        }
        stream.synchronize_async().await?;

        let mut host = vec![0; 16];
        device.copy_to_host(&mut host)?;
        assert_eq!(host, vec![7; 16]);

        Ok(())
    }

    #[tokio::test]
    async fn test_dropping_pending_future_does_not_cancel_stream_work() -> Result<()> {
        let (_lock, context) = testing::bootstrap()?;
        let stream = context.create_stream()?;
        let mut device = DeviceMemory::<u8>::zeroed(&context, 8)?;

        // SAFETY: the test drains the stream before the memory is dropped or read.
        unsafe {
            device.set_value_async_unchecked(11, &stream)?;
        }
        let future = stream.checked_completion_future()?;
        drop(future);

        stream.synchronize()?;
        let mut host = vec![0; 8];
        device.copy_to_host(&mut host)?;
        assert_eq!(host, vec![11; 8]);

        Ok(())
    }

    #[tokio::test]
    async fn test_stream_future_registration_is_rejected_during_capture() -> Result<()> {
        let (_lock, context) = testing::bootstrap()?;
        let stream = context.create_stream()?;

        stream.begin_capture(StreamCaptureMode::Relaxed)?;
        let error = stream.completion_future().unwrap_err();
        drop(stream.end_capture());

        assert!(matches!(
            error,
            Error::Cuda {
                code: Status::StreamCaptureUnsupported,
                ..
            }
        ));

        Ok(())
    }

    #[tokio::test]
    async fn test_event_future_orders_work_across_streams() -> Result<()> {
        let (_lock, context) = testing::bootstrap()?;
        let stream_a = context.create_stream()?;
        let stream_b = context.create_stream()?;
        let event = context.create_event()?;
        let mut device = DeviceMemory::<u8>::zeroed(&context, 4)?;

        // SAFETY: the test drains the stream before the memory is dropped or read.
        unsafe {
            device.set_value_async_unchecked(5, &stream_a)?;
        }
        event.record(&stream_a, EventRecordFlags::DEFAULT)?;
        event.synchronize_async_on(&stream_b).await?;

        let mut host = vec![0; 4];
        device.copy_to_host(&mut host)?;
        assert_eq!(host, vec![5; 4]);

        Ok(())
    }

    #[tokio::test]
    async fn test_graph_launch_async_waits_for_launch_completion() -> Result<()> {
        let (_lock, context) = testing::bootstrap()?;
        let stream = context.create_stream()?;
        let mut graph = context.create_graph()?;

        graph.add_empty_node(&[])?;
        let executable = graph.instantiate()?;
        // SAFETY: the buffers captured by the graph outlive the launch, which the test drains.
        unsafe { executable.launch_async(&stream) }.unwrap().await?;

        Ok(())
    }
}
