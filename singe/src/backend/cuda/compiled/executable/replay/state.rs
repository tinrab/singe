use singe_core::ErrorChain;
use singe_cuda::submission::{Completion, Execution, Resources};

use crate::{
    backend::cuda::{
        graph_value::{CudaGraphValue, GraphOutputBuffers},
        stream::CudaStream,
    },
    error::Result,
    tensor::Tensor,
};

/// Replay statistics collected after CUDA graph compilation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GraphReplayStats {
    /// Number of direct `run` calls submitted to this executable.
    pub(crate) synchronous_runs: usize,
    /// Number of `run_on` calls submitted to this executable.
    pub(crate) asynchronous_runs: usize,
    /// Number of `run_on` calls that returned output tensors successfully.
    pub(crate) successful_replays: usize,
    /// Number of `run_on` calls that returned an error.
    pub(crate) failed_replays: usize,
    /// Number of runtime input tensors copied into graph-private placeholders.
    pub(crate) input_copies: usize,
    /// Number of output tensors copied into per-run output tensors.
    pub(crate) output_copies: usize,
    /// Number of native executable CUDA graph launches.
    pub(crate) native_launches: usize,
    /// Number of replays that only used metadata/input/output copy work.
    pub(crate) metadata_only_replays: usize,
    /// Number of pending-run completion events recorded successfully.
    pub(crate) completion_events: usize,
    /// Number of replay submissions synchronized immediately because event creation or recording failed.
    pub(crate) submission_sync_fallbacks: usize,
}

impl GraphReplayStats {
    /// Returns direct `run` calls submitted to this executable.
    #[must_use]
    pub fn synchronous_runs(&self) -> usize {
        self.synchronous_runs
    }

    /// Returns `run_on` calls submitted to this executable.
    #[must_use]
    pub fn asynchronous_runs(&self) -> usize {
        self.asynchronous_runs
    }

    /// Returns `run_on` calls that returned output tensors successfully.
    #[must_use]
    pub fn successful_replays(&self) -> usize {
        self.successful_replays
    }

    /// Returns `run_on` calls that returned an error.
    #[must_use]
    pub fn failed_replays(&self) -> usize {
        self.failed_replays
    }

    /// Returns runtime input tensors copied into graph-private placeholders.
    #[must_use]
    pub fn input_copies(&self) -> usize {
        self.input_copies
    }

    /// Returns output tensors copied into per-run output tensors.
    #[must_use]
    pub fn output_copies(&self) -> usize {
        self.output_copies
    }

    /// Returns native executable CUDA graph launches.
    #[must_use]
    pub fn native_launches(&self) -> usize {
        self.native_launches
    }

    /// Returns replays that only used metadata/input/output copy work.
    #[must_use]
    pub fn metadata_only_replays(&self) -> usize {
        self.metadata_only_replays
    }

    /// Returns pending-run completion events recorded successfully.
    #[must_use]
    pub fn completion_events(&self) -> usize {
        self.completion_events
    }

    /// Returns replay submissions synchronized immediately after event creation or record failure.
    #[must_use]
    pub fn submission_sync_fallbacks(&self) -> usize {
        self.submission_sync_fallbacks
    }
}

#[derive(Debug, Default)]
pub struct PendingRuns {
    runs: Vec<PendingRun>,
}

#[derive(Debug)]
struct PendingRun {
    completion: Completion,
    inputs: Vec<Tensor>,
    outputs: Vec<CudaGraphValue>,
}

impl Drop for PendingRun {
    fn drop(&mut self) {
        // Owners cannot be released when CUDA may still access their allocations.
        if let Err(error) = self.completion.wait() {
            eprintln!(
                "failed to wait for cuda replay completion: {}",
                ErrorChain(&error)
            );
        }
    }
}

impl PendingRuns {
    pub fn len(&self) -> usize {
        self.runs.len()
    }

    pub fn held_output_count(&self) -> usize {
        self.runs.iter().map(|r| r.outputs.len()).sum()
    }

    pub fn held_input_count(&self) -> usize {
        self.runs.iter().map(|r| r.inputs.len()).sum()
    }

    pub fn clear(&mut self) {
        self.runs.clear();
    }

    pub fn push(
        &mut self,
        completion: Completion,
        inputs: Vec<Tensor>,
        outputs: Vec<CudaGraphValue>,
    ) {
        self.runs.push(PendingRun {
            completion,
            inputs,
            outputs,
        });
    }

    pub fn prune_completed(&mut self) -> Result<()> {
        let mut index = 0;
        while index < self.runs.len() {
            if self.runs[index].completion.query()? {
                self.runs.swap_remove(index);
            } else {
                index += 1;
            }
        }
        Ok(())
    }

    pub fn wait_for_pending(&self, stream: &CudaStream) -> Result<()> {
        for run in &self.runs {
            run.completion.order_on(stream.raw())?;
        }
        Ok(())
    }

    pub fn synchronize_on(&mut self, stream: &CudaStream) -> Result<()> {
        let mut index = 0;
        while index < self.runs.len() {
            if self.runs[index].completion.stream() == stream.raw() {
                self.runs[index].completion.wait()?;
                self.runs.swap_remove(index);
            } else {
                index += 1;
            }
        }
        Ok(())
    }

    pub fn synchronize(&mut self) -> Result<()> {
        for run in &self.runs {
            run.completion.wait()?;
        }

        self.clear();
        Ok(())
    }

    pub fn synchronize_before_drop(&mut self) {
        self.clear();
    }
}

#[derive(Debug)]
pub struct ReplaySubmission {
    execution: Option<Execution>,
    pub(crate) inputs: Vec<Tensor>,
    pub(crate) outputs: Vec<CudaGraphValue>,
    pub(crate) input_copies: usize,
    pub(crate) output_copies: usize,
    pub(crate) native_launches: usize,
}

#[derive(Clone, Copy)]
pub enum ReplayOutputTarget<'a> {
    Allocate,
    CallerOwned(GraphOutputBuffers<'a>),
    GraphOwned,
}

impl ReplaySubmission {
    pub(crate) fn create(stream: &CudaStream) -> Result<Self> {
        Ok(Self {
            execution: Some(Execution::create(stream.raw())?),
            inputs: Vec::new(),
            outputs: Vec::new(),
            input_copies: 0,
            output_copies: 0,
            native_launches: 0,
        })
    }

    pub(crate) unsafe fn enqueue<T>(
        &mut self,
        resources: Resources,
        enqueue: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            self.execution
                .as_mut()
                .expect("active replay submission")
                .enqueue(resources, |_| enqueue())
        }
    }

    pub(crate) fn finish(&mut self) -> Result<Completion> {
        let submission = self
            .execution
            .take()
            .expect("active replay submission")
            .finish(())?;
        Ok(submission.completion().clone())
    }

    pub(crate) fn has_submitted_work(&self) -> bool {
        self.execution.is_some()
            && (self.input_copies != 0 || self.output_copies != 0 || self.native_launches != 0)
    }

    pub(crate) fn record_input_copy(&mut self, input: Tensor) {
        self.input_copies += 1;
        self.inputs.push(input);
    }

    pub(crate) fn record_output_copy(&mut self, output: CudaGraphValue) {
        self.output_copies += 1;
        self.outputs.push(output);
    }

    pub(crate) fn record_native_launch(&mut self) {
        self.native_launches += 1;
    }

    pub(crate) fn clear_retained_values(&mut self) {
        self.inputs.clear();
        self.outputs.clear();
    }
}
