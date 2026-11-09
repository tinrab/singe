use singe_core::ErrorChain;
use singe_cuda::{
    graph::{ExecutableGraph as NativeExecutableGraph, Graph as NativeCudaGraph},
    stream::{StreamCaptureMode, StreamCaptureStatus},
};

use crate::{
    backend::cuda::{self, CudaStream},
    error::Result,
};

#[derive(Debug)]
struct CudaStreamCaptureGuard<'stream> {
    stream: &'stream CudaStream,
    active: bool,
}

impl CudaStream {
    /// Begins CUDA graph capture on this stream.
    ///
    /// Capture must be ended on the same stream with [`CudaStream::end_capture`].
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot begin capture, the stream is already capturing, or the requested capture mode is invalid for the current thread state.
    pub fn begin_capture(&self, mode: StreamCaptureMode) -> Result<()> {
        self.stream.begin_capture(mode)?;
        Ok(())
    }

    /// Ends CUDA graph capture on this stream and returns ownership of the captured graph.
    ///
    /// # Errors
    ///
    /// Returns an error if capture is not active on this stream or the capture was invalidated by an unsupported operation.
    pub fn end_capture(&self) -> Result<CudaGraph> {
        Ok(CudaGraph::new(self.stream.end_capture()?))
    }

    /// Returns the current CUDA graph capture status for this stream.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query the stream capture state.
    pub fn capture_status(&self) -> Result<StreamCaptureStatus> {
        Ok(self.stream.capture_status()?)
    }

    /// Runs `f` while this stream is current and returns the captured CUDA graph.
    ///
    /// The previous current stream for this stream's CUDA device is restored if `f` succeeds, returns an error, or unwinds.
    /// If `f` returns an error or unwinds, the in-progress capture is ended and discarded.
    ///
    /// # Errors
    ///
    /// Returns an error if capture setup fails, setting the current stream fails, `f` returns an error, or ending the capture fails.
    /// If `f` returns an error, that error is returned after a best-effort attempt to discard the in-progress capture.
    pub fn capture<F, R>(&self, mode: StreamCaptureMode, f: F) -> Result<(CudaGraph, R)>
    where
        F: FnOnce() -> Result<R>,
    {
        self.begin_capture(mode)?;
        let capture_guard = CudaStreamCaptureGuard {
            stream: self,
            active: true,
        };
        let stream_guard = cuda::set_current_stream(self)?;
        let result = f();
        drop(stream_guard);

        match result {
            Ok(value) => Ok((capture_guard.finish()?, value)),
            Err(error) => {
                // The body's error is the one to report; ending the abandoned capture can fail only as a consequence of it.
                capture_guard.discard().ok();
                Err(error)
            }
        }
    }
}

impl CudaStreamCaptureGuard<'_> {
    fn finish(mut self) -> Result<CudaGraph> {
        self.active = false;
        self.stream.end_capture()
    }

    fn discard(mut self) -> Result<()> {
        self.active = false;
        self.stream.end_capture().map(|_| ())
    }
}

impl Drop for CudaStreamCaptureGuard<'_> {
    fn drop(&mut self) {
        if self.active
            && let Err(error) = self.stream.end_capture()
        {
            eprintln!("failed to end cuda stream capture: {}", ErrorChain(&error));
        }
    }
}

/// Captured native CUDA graph.
#[derive(Debug)]
pub struct CudaGraph {
    graph: NativeCudaGraph,
}

/// Instantiated native CUDA graph executable.
#[derive(Debug)]
pub struct CudaExecutableGraph {
    executable: NativeExecutableGraph,
}

impl CudaGraph {
    pub(crate) fn new(graph: NativeCudaGraph) -> Self {
        Self { graph }
    }

    pub fn into_native(self) -> NativeCudaGraph {
        self.graph
    }

    /// Instantiates this captured CUDA graph into an executable graph.
    ///
    /// The captured graph remains owned by `self`; the executable owns the instantiated CUDA graph state.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects graph instantiation.
    pub fn instantiate(&self) -> Result<CudaExecutableGraph> {
        Ok(CudaExecutableGraph {
            executable: self.graph.instantiate()?,
        })
    }
}

impl CudaExecutableGraph {
    /// Uploads this executable graph to `stream`.
    ///
    /// Upload is ordered on `stream` and must complete before a dependent launch can use the uploaded state.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the upload for the executable graph or target stream.
    pub fn upload(&self, stream: &CudaStream) -> Result<()> {
        self.executable.upload(stream.raw())?;
        Ok(())
    }

    /// Launches this executable graph on `stream`.
    ///
    /// The launch is asynchronous with respect to the host and ordered by `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot enqueue the executable graph on `stream`.
    /// # Safety
    /// Keep every captured allocation live and exclude conflicting access until completion.
    pub unsafe fn launch(&self, stream: &CudaStream) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            self.executable.launch(stream.raw())?;
        }
        Ok(())
    }
}
