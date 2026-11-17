//! Checked stream completion futures.

use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context as TaskContext, Poll},
};

use crate::{
    error::Result,
    event::{EventFlags, EventRecordFlags},
    stream::Stream,
    submission::{Completion, Resources},
};

/// Checked event-backed completion that observes CUDA errors even when callbacks do not run.
///
/// The completion worker retains the stream until the recorded work finishes.
/// Dropping the future does not cancel queued work.
/// The future does not retain owners of raw pointers or graph code; those lifetimes follow the submission API's contract.
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
    /// Returns a future that resolves with CUDA's status once the work queued so far on this stream completes.
    ///
    /// An event wait observes CUDA status independently of notification callbacks.
    /// Registration is rejected while stream capture is active.
    ///
    /// # Errors
    ///
    /// Returns an error if capture is active or invalidated, CUDA cannot query capture state, create or record the completion event, or bind the context, or the completion worker cannot be started or accept work.
    /// The returned future reports asynchronous CUDA completion errors.
    pub fn completion(&self) -> Result<CheckedStreamFuture> {
        self.ensure_not_capturing()?;

        let event = Arc::new(self.context().create_event(EventFlags::DISABLE_TIMING)?);
        event.record(self, EventRecordFlags::DEFAULT)?;
        Ok(CheckedStreamFuture {
            completion: Completion::start(event, self.clone(), Resources::new())?,
        })
    }
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use std::sync::Arc;

    use crate::{
        error::{Error, Result, Status},
        event::{EventFlags, EventRecordFlags, EventWaitFlags},
        graph::GraphInstantiateFlags,
        memory::DeviceMemory,
        stream::{StreamCaptureMode, StreamOptions},
        testing,
    };

    #[tokio::test]
    async fn test_stream_completion_waits_for_memset() -> Result<()> {
        let test_device = testing::bootstrap()?;
        let context = Arc::clone(test_device.context());
        let stream = context.create_stream(StreamOptions::default())?;
        let mut device = DeviceMemory::<u8>::zeroed(&context, 16)?;

        // SAFETY: the test drains the stream before the memory is dropped or read.
        unsafe {
            device.set_value_async(7, &stream)?;
        }
        stream.completion()?.await?;

        let mut host = vec![0; 16];
        device.copy_to_host(&mut host)?;
        assert_eq!(host, vec![7; 16]);

        Ok(())
    }

    #[tokio::test]
    async fn test_dropping_pending_future_does_not_cancel_stream_work() -> Result<()> {
        let test_device = testing::bootstrap()?;
        let context = Arc::clone(test_device.context());
        let stream = context.create_stream(StreamOptions::default())?;
        let mut device = DeviceMemory::<u8>::zeroed(&context, 8)?;

        // SAFETY: the test drains the stream before the memory is dropped or read.
        unsafe {
            device.set_value_async(11, &stream)?;
        }
        let future = stream.completion()?;
        drop(future);

        stream.synchronize()?;
        let mut host = vec![0; 8];
        device.copy_to_host(&mut host)?;
        assert_eq!(host, vec![11; 8]);

        Ok(())
    }

    #[tokio::test]
    async fn test_stream_completion_is_rejected_during_capture() -> Result<()> {
        let test_device = testing::bootstrap()?;
        let context = Arc::clone(test_device.context());
        let stream = context.create_stream(StreamOptions::default())?;

        stream.begin_capture(StreamCaptureMode::Relaxed)?;
        let error = stream.completion().unwrap_err();
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
    async fn test_event_wait_orders_completion_across_streams() -> Result<()> {
        let test_device = testing::bootstrap()?;
        let context = Arc::clone(test_device.context());
        let stream_a = context.create_stream(StreamOptions::default())?;
        let stream_b = context.create_stream(StreamOptions::default())?;
        let event = context.create_event(EventFlags::DEFAULT)?;
        let mut device = DeviceMemory::<u8>::zeroed(&context, 4)?;

        // SAFETY: the test drains the stream before the memory is dropped or read.
        unsafe {
            device.set_value_async(5, &stream_a)?;
        }
        event.record(&stream_a, EventRecordFlags::DEFAULT)?;
        stream_b.wait_event(&event, EventWaitFlags::DEFAULT)?;
        stream_b.completion()?.await?;

        let mut host = vec![0; 4];
        device.copy_to_host(&mut host)?;
        assert_eq!(host, vec![5; 4]);

        Ok(())
    }

    #[tokio::test]
    async fn test_graph_launch_completion_resolves() -> Result<()> {
        let test_device = testing::bootstrap()?;
        let context = Arc::clone(test_device.context());
        let stream = context.create_stream(StreamOptions::default())?;
        let mut graph = context.create_graph()?;

        graph.add_empty_node(&[])?;
        let executable = graph.instantiate(GraphInstantiateFlags::empty())?;
        // SAFETY: the graph has no operands, and the test drains the stream.
        unsafe { executable.launch_async(&stream)? };
        stream.completion()?.await?;

        Ok(())
    }
}
