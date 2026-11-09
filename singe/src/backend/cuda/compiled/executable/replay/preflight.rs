use singe_cuda::{error::Error as CudaError, stream::StreamCaptureStatus};

use crate::{
    backend::cuda::{
        compiled::executable::CudaCompiledGraph, nvtx::CudaNvtxRange, stream::CudaStream,
    },
    error::{Error, Result},
    utility::Locked,
};

impl CudaCompiledGraph {
    pub(crate) fn prepare_replay_on_stream(&mut self, stream: &CudaStream) -> Result<()> {
        if stream.device() != self.device {
            self.record_replay_preflight_failure();
            return Err(Error::device_mismatch(
                "cuda graph replay stream device",
                self.device,
                stream.device(),
            ));
        }

        if stream.raw().context() != self.runtime.context() {
            return Err(CudaError::StreamContextMismatch.into());
        }

        if stream.raw().capture_status()? != StreamCaptureStatus::None {
            return Err(Error::unsupported_operation(
                "cuda graph replay during stream capture",
            ));
        }

        let _sync_range = CudaNvtxRange::create("singe cuda graph replay synchronize");
        if let Err(error) = self.runtime.context().bind() {
            self.record_replay_preflight_failure();
            return Err(Error::from(error));
        }

        if let Err(error) = self.pending_runs.prune_completed() {
            self.record_replay_preflight_failure();
            return Err(error);
        }

        if let Err(error) = self.pending_runs.wait_for_pending(stream) {
            self.record_replay_preflight_failure();
            return Err(error);
        }
        Ok(())
    }

    /// Uploads native executable graph state to `stream` when the graph has native nodes.
    ///
    /// Warm-up operation for the compiled CUDA graph executable.
    /// It binds the graph's runtime context and validates that `stream` belongs to the graph device, but it does not validate replay inputs, apply initializer overrides, launch graph work, or create a pending async replay.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` belongs to a different device, context binding fails, or CUDA rejects the executable graph upload.
    pub fn upload(&self, stream: &CudaStream) -> Result<()> {
        if stream.device() != self.device {
            return Err(Error::device_mismatch(
                "cuda graph upload stream device",
                self.device,
                stream.device(),
            ));
        }

        self.runtime.context().bind()?;
        if let Some(executable) = &self.executable {
            executable
                .locked("cuda runtime cache")?
                .upload(stream.raw())?;
        }
        Ok(())
    }

    fn record_replay_preflight_failure(&mut self) {
        self.replay_stats.failed_replays = self.replay_stats.failed_replays.saturating_add(1);
    }
}
