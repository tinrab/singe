use std::collections::HashMap;

use crate::{
    backend::cuda::{compiled::executable::CudaCompiledGraph, kernel_plan, stream::CudaStream},
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

    /// Returns the data error a checking kernel reported during completed replays, clearing it.
    fn take_status(&self) -> Result<()> {
        self.private
            .status
            .as_ref()
            .map_or(Ok(()), kernel_plan::take_status)
    }
}
