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
    event::Event,
    graph::ExecutableGraph,
    stream::{Stream, StreamScope},
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
        if let Some(waker) = self.waker.lock().expect("completion waker poisoned").take() {
            waker.wake();
        }
    }
}

#[derive(Debug)]
struct CheckedCompletionState {
    inner: Mutex<CheckedCompletionInner>,
}

#[derive(Debug)]
struct CheckedCompletionInner {
    result: Option<Result<()>>,
    waker: Option<Waker>,
}

impl CheckedCompletionState {
    fn create() -> Self {
        Self {
            inner: Mutex::new(CheckedCompletionInner {
                result: None,
                waker: None,
            }),
        }
    }

    fn signal(&self, result: Result<()>) {
        let waker = {
            let mut inner = self
                .inner
                .lock()
                .expect("checked completion waker poisoned");
            inner.result = Some(result);
            inner.waker.take()
        };

        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

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

#[derive(Debug)]
pub struct CheckedStreamFuture {
    state: Arc<CheckedCompletionState>,
    _stream: Stream,
}

impl Future for CheckedStreamFuture {
    type Output = Result<()>;

    fn poll(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Self::Output> {
        let mut inner = self
            .state
            .inner
            .lock()
            .expect("checked completion waker poisoned");

        if let Some(result) = inner.result.take() {
            return Poll::Ready(result);
        }

        inner.waker = Some(cx.waker().clone());
        Poll::Pending
    }
}

impl Unpin for CheckedStreamFuture {}

#[derive(Debug)]
pub struct CudaFuture<T> {
    completion: CheckedStreamFuture,
    output: Option<T>,
}

impl<T> Future for CudaFuture<T> {
    type Output = Result<T>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.completion).poll(cx) {
            Poll::Ready(Ok(())) => {
                let output = self
                    .output
                    .take()
                    .expect("cuda future output already consumed");
                Poll::Ready(Ok(output))
            }
            Poll::Ready(Err(err)) => Poll::Ready(Err(err)),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<T> Unpin for CudaFuture<T> {}

impl Stream {
    /// Returns a future that resolves when a host function enqueued at the current end of this stream runs.
    ///
    /// This is a notification primitive.
    /// It does not report asynchronous CUDA errors after registration.
    /// Use [`Stream::checked_completion_future`] or [`Stream::synchronize_async`] when the result must include CUDA status.
    pub fn completion_future(&self) -> Result<StreamFuture> {
        self.ensure_not_capturing_for_future()?;

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
    /// This uses CUDA's stream callback status path and is therefore rejected while stream capture is active.
    pub fn checked_completion_future(&self) -> Result<CheckedStreamFuture> {
        self.ensure_not_capturing_for_future()?;

        let state = Arc::new(CheckedCompletionState::create());
        let callback_state = Arc::clone(&state);
        self.add_callback(move |result| callback_state.signal(result))?;

        Ok(CheckedStreamFuture {
            state,
            _stream: self.clone(),
        })
    }

    pub async fn synchronize_async(&self) -> Result<()> {
        self.checked_completion_future()?.await
    }

    pub fn enqueue_async<T, F>(&self, f: F) -> Result<CudaFuture<T>>
    where
        F: FnOnce(&Stream) -> Result<T>,
    {
        let output = f(self)?;
        let completion = self.checked_completion_future()?;
        Ok(CudaFuture {
            completion,
            output: Some(output),
        })
    }
}

impl<'scope, 'env> StreamScope<'scope, 'env> {
    pub fn completion_future(&self) -> Result<StreamFuture> {
        self.stream().completion_future()
    }

    pub fn checked_completion_future(&self) -> Result<CheckedStreamFuture> {
        self.stream().checked_completion_future()
    }

    pub async fn synchronize_async(&self) -> Result<()> {
        self.stream().synchronize_async().await
    }
}

impl Event {
    pub fn completion_future_on(&self, stream: &Stream) -> Result<StreamFuture> {
        stream.wait_event(self)?;
        stream.completion_future()
    }

    pub fn checked_completion_future_on(&self, stream: &Stream) -> Result<CheckedStreamFuture> {
        stream.wait_event(self)?;
        stream.checked_completion_future()
    }

    pub async fn synchronize_async_on(&self, stream: &Stream) -> Result<()> {
        self.checked_completion_future_on(stream)?.await
    }
}

impl ExecutableGraph {
    pub fn launch_async(&self, stream: &Stream) -> Result<CheckedStreamFuture> {
        self.launch(stream)?;
        stream.checked_completion_future()
    }

    pub async fn launch_and_wait(&self, stream: &Stream) -> Result<()> {
        self.launch_async(stream)?.await
    }

    pub fn upload_async(&self, stream: &Stream) -> Result<CheckedStreamFuture> {
        self.upload(stream)?;
        stream.checked_completion_future()
    }

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
    async fn stream_future_resolves_after_empty_stream() -> Result<()> {
        let (_lock, ctx) = testing::bootstrap()?;
        let stream = ctx.create_stream()?;

        stream.completion_future()?.await;
        stream.checked_completion_future()?.await?;
        stream.synchronize_async().await?;

        Ok(())
    }

    #[tokio::test]
    async fn stream_synchronize_async_waits_for_memset() -> Result<()> {
        let (_lock, ctx) = testing::bootstrap()?;
        let stream = ctx.create_stream()?;
        let mut device = DeviceMemory::<u8>::zeroes(16)?;

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
    async fn dropping_pending_future_does_not_cancel_stream_work() -> Result<()> {
        let (_lock, ctx) = testing::bootstrap()?;
        let stream = ctx.create_stream()?;
        let mut device = DeviceMemory::<u8>::zeroes(8)?;

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
    async fn stream_future_registration_is_rejected_during_capture() -> Result<()> {
        let (_lock, ctx) = testing::bootstrap()?;
        let stream = ctx.create_stream()?;

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
    async fn event_future_orders_work_across_streams() -> Result<()> {
        let (_lock, ctx) = testing::bootstrap()?;
        let stream_a = ctx.create_stream()?;
        let stream_b = ctx.create_stream()?;
        let event = ctx.create_event()?;
        let mut device = DeviceMemory::<u8>::zeroes(4)?;

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
    async fn graph_launch_async_waits_for_launch_completion() -> Result<()> {
        let (_lock, ctx) = testing::bootstrap()?;
        let stream = ctx.create_stream()?;
        let mut graph = ctx.create_graph()?;

        graph.add_empty_node(&[])?;
        let executable = graph.instantiate()?;
        executable.launch_async(&stream).unwrap().await?;

        Ok(())
    }
}
