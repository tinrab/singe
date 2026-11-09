use std::collections::HashMap;

use crate::{
    backend::cuda::{
        compiled::executable::CudaCompiledGraph, graph_value::CudaGraphValue, kernel_plan,
        stream::CudaStream,
    },
    error::Result,
    tensor::Tensor,
};

impl CudaCompiledGraph {
    /// Runs replay on an internal stream and returns independent completed tensors.
    ///
    /// # Errors
    ///
    /// Returns an error if internal stream creation, replay preparation, execution, output copying, synchronization, or promotion to uniquely owned tensors fails.
    pub fn run(&mut self, inputs: &HashMap<String, Tensor>) -> Result<Vec<Tensor>> {
        self.replay_stats.synchronous_runs = self.replay_stats.synchronous_runs.saturating_add(1);
        let stream = CudaStream::create(self.device)?;
        self.run_on(inputs, &stream)
    }

    /// Waits for all submitted replay work and releases its retained handles.
    ///
    /// # Errors
    ///
    /// Returns an error if waiting for a retained replay completion fails, or a checking kernel such as an index lookup reported bad data.
    pub fn synchronize(&mut self) -> Result<()> {
        self.pending_runs.synchronize()?;
        self.take_status()
    }

    /// Waits for replays on one stream and releases their retained submission handles.
    ///
    /// # Errors
    ///
    /// Returns an error if waiting for a retained replay completion on `stream` fails, or a checking kernel reported bad data.
    pub fn synchronize_on(&mut self, stream: &CudaStream) -> Result<()> {
        self.pending_runs.synchronize_on(stream)?;
        self.take_status()
    }

    /// Returns the data error a checking kernel reported during completed replays, clearing it.
    fn take_status(&self) -> Result<()> {
        self.private
            .status
            .as_ref()
            .map_or(Ok(()), kernel_plan::take_status)
    }

    /// Runs replay and completes copies into caller-provided buffers.
    ///
    /// # Safety
    /// Destinations must not overlap inputs or graph-private storage.
    /// Exclude conflicting access through all aliases throughout this call.
    ///
    /// # Errors
    ///
    /// Returns an error if stream creation, input or destination validation, replay submission, copies, or synchronization fails.
    pub unsafe fn run_into(
        &mut self,
        inputs: &HashMap<String, Tensor>,
        outputs: &[CudaGraphValue],
    ) -> Result<()> {
        self.replay_stats.synchronous_runs = self.replay_stats.synchronous_runs.saturating_add(1);
        let stream = CudaStream::create(self.device)?;
        // SAFETY: caller establishes destination exclusivity.
        unsafe {
            self.enqueue_on_into(inputs, outputs, &stream)?;
        }
        self.synchronize()
    }

    /// Runs replay and returns completed graph-owned regions without copy-out.
    /// Later replay may overwrite them; raw readers must establish ordering.
    ///
    /// # Errors
    ///
    /// Returns an error if stream creation, replay preparation or submission, output lookup, or synchronization fails.
    pub fn run_graph_owned_outputs(
        &mut self,
        inputs: &HashMap<String, Tensor>,
    ) -> Result<Vec<CudaGraphValue>> {
        self.replay_stats.synchronous_runs = self.replay_stats.synchronous_runs.saturating_add(1);
        let stream = CudaStream::create(self.device)?;
        let outputs = self.enqueue_on_graph_owned_outputs(inputs, &stream)?;
        self.synchronize()?;
        Ok(outputs)
    }
}
