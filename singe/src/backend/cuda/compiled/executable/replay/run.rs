use std::{collections::HashMap, sync::Arc};

use singe_cuda::submission::{AccessMode, Resources};

use crate::{
    backend::cuda::{
        compiled::executable::{
            CudaCompiledGraph,
            replay::{ReplayOutputTarget, ReplaySubmission},
        },
        graph_value::{CudaGraphValue, GraphOutputBuffers},
        nvtx::CudaNvtxRange,
        stream::CudaStream,
    },
    error::Result,
    graph::{ir::ValueBindingKind, narrow_counted},
    tensor::Tensor,
    utility::Locked,
};

impl CudaCompiledGraph {
    /// Runs replay on `stream` and returns independent, completed tensor snapshots.
    ///
    /// # Errors
    ///
    /// Returns an error if replay preparation or submission fails, waiting for completion fails, or an output allocation cannot be promoted with unique ownership.
    pub fn run_on(
        &mut self,
        inputs: &HashMap<String, Tensor>,
        stream: &CudaStream,
    ) -> Result<Vec<Tensor>> {
        self.run_on_then(inputs, stream, |_| Ok(()))
    }

    // Runs replay on `stream` as `run_on` does, calling `enqueued` once all of its work is on the stream and before waiting for any of it.
    pub fn run_on_then(
        &mut self,
        inputs: &HashMap<String, Tensor>,
        stream: &CudaStream,
        enqueued: impl FnOnce(&CudaStream) -> Result<()>,
    ) -> Result<Vec<Tensor>> {
        let outputs = self.enqueue_on(inputs, stream)?;
        enqueued(stream)?;
        self.synchronize()?;
        // SAFETY: fresh contiguous copy-out allocations are fully produced and the completed submission has released its retained handles.
        let outputs = outputs
            .into_iter()
            // SAFETY: the completed producer wrote every element of this tightly packed output.
            .map(|output| unsafe { output.assume_init() })
            .collect::<Result<Vec<_>>>()?;
        narrow_counted(self.ir.counted_outputs(), outputs)
    }

    /// Enqueues replay and returns independent pending output allocations.
    ///
    /// Results expose metadata only.
    /// Call `synchronize` before promoting fully produced outputs.
    /// This executable retains all submitted owners meanwhile.
    ///
    /// # Errors
    ///
    /// Returns an error if stream or input validation fails, required graph values are missing, output buffers are incompatible, a cache or executable mutex is poisoned, or parameter updates, copies, graph launch, or completion tracking fails.
    pub fn enqueue_on(
        &mut self,
        inputs: &HashMap<String, Tensor>,
        stream: &CudaStream,
    ) -> Result<Vec<CudaGraphValue>> {
        self.enqueue_on_with_output_target(inputs, stream, ReplayOutputTarget::Allocate)
    }

    /// Enqueues replay and copies logical outputs into caller-provided buffers.
    ///
    /// # Safety
    /// Exclude conflicting access through every destination alias until replay completes.
    /// Destinations must not overlap inputs or graph-private storage.
    ///
    /// # Errors
    ///
    /// Returns an error if stream or input validation fails, required graph values are missing, output buffers are incompatible, a cache or executable mutex is poisoned, or parameter updates, copies, graph launch, or completion tracking fails.
    pub unsafe fn enqueue_on_into(
        &mut self,
        inputs: &HashMap<String, Tensor>,
        outputs: &[CudaGraphValue],
        stream: &CudaStream,
    ) -> Result<()> {
        // SAFETY: caller establishes the destination request's access contract.
        let outputs = unsafe { GraphOutputBuffers::create(outputs) };
        self.enqueue_on_with_output_target(inputs, stream, ReplayOutputTarget::CallerOwned(outputs))
            .map(|_| ())
    }

    /// Enqueues replay and returns retained graph-owned regions without copy-out.
    ///
    /// Values expose no initialized reads.
    /// Later replay may overwrite their contents.
    /// Raw access requires caller-established completion and ordering.
    ///
    /// # Errors
    ///
    /// Returns an error if stream or input validation fails, required graph values are missing, output buffers are incompatible, a cache or executable mutex is poisoned, or parameter updates, copies, graph launch, or completion tracking fails.
    pub fn enqueue_on_graph_owned_outputs(
        &mut self,
        inputs: &HashMap<String, Tensor>,
        stream: &CudaStream,
    ) -> Result<Vec<CudaGraphValue>> {
        self.enqueue_on_with_output_target(inputs, stream, ReplayOutputTarget::GraphOwned)
    }

    pub(crate) fn enqueue_on_with_output_target(
        &mut self,
        inputs: &HashMap<String, Tensor>,
        stream: &CudaStream,
        output_target: ReplayOutputTarget<'_>,
    ) -> Result<Vec<CudaGraphValue>> {
        self.replay_stats.asynchronous_runs = self.replay_stats.asynchronous_runs.saturating_add(1);
        let _replay_range = CudaNvtxRange::create("singe cuda graph replay");
        self.prepare_replay_on_stream(stream)?;
        let mut submission = ReplaySubmission::create(stream)?;
        let result = run_replay_submission(self, inputs, stream, output_target, &mut submission);
        match result {
            Ok(outputs) => {
                self.record_replay_success(&submission);
                Ok(outputs)
            }
            Err(error) => {
                if submission.has_submitted_work()
                    && let Err(finish_error) =
                        self.finish_replay_submission(&mut submission, stream)
                {
                    self.record_replay_failure(&submission);
                    return Err(finish_error);
                }

                self.record_replay_failure(&submission);
                Err(error)
            }
        }
    }
}

pub fn run_replay_submission(
    graph: &mut CudaCompiledGraph,
    inputs: &HashMap<String, Tensor>,
    stream: &CudaStream,
    output_target: ReplayOutputTarget<'_>,
    submission: &mut ReplaySubmission,
) -> Result<Vec<CudaGraphValue>> {
    let cache = {
        let _input_range = CudaNvtxRange::create("singe cuda graph replay prepare inputs");
        graph.runtime_cache(inputs, stream, submission)?
    };
    if let Some(executable_owner) = &graph.executable {
        let mut executable = executable_owner.locked("cuda runtime cache")?;
        {
            let _update_range = CudaNvtxRange::create("singe cuda graph replay update parameters");
            for step in &mut graph.cudnn_steps {
                if !step.update_on_replay() {
                    continue;
                }

                let mut cudnn = graph.runtime.cudnn().locked("cuda runtime cache")?;
                step.update_graph(&mut cudnn, &cache)?;
                executable.set_child_graph_node(&step.graph_node(), step.child_graph())?;
            }
        }

        let _launch_range = CudaNvtxRange::create("singe cuda graph replay launch");
        submission.record_native_launch();
        let mut resources = Resources::new();
        resources.retain(Arc::clone(executable_owner));
        resources.retain(Arc::clone(&graph.runtime));
        for (&id, value) in &graph.values {
            let binding = id;
            let mode = match graph.ir.required_value_binding_kind(binding)? {
                ValueBindingKind::StaticInput | ValueBindingKind::Constant => AccessMode::Read,
                _ => AccessMode::ReadWrite,
            };
            resources.extend(value.submission_resources(mode)?);
        }

        for step in &graph.cudnn_steps {
            resources.extend(step.submission_resources()?);
        }

        for step in &graph.cutensor_steps {
            resources.extend(step.submission_resources()?);
        }

        // SAFETY: the declared resources retain every buffer and the stream orders all access to them.
        unsafe {
            submission.enqueue(resources, || Ok(executable.launch(stream.raw())?))?;
        }

        drop(executable);
    }

    let outputs = {
        let _output_range = CudaNvtxRange::create("singe cuda graph replay copy outputs");
        match output_target {
            ReplayOutputTarget::Allocate => {
                graph.copy_outputs_for_run(&cache, stream, submission)?
            }
            ReplayOutputTarget::CallerOwned(outputs) => {
                graph.copy_outputs_into(&cache, outputs, stream, submission)?
            }
            ReplayOutputTarget::GraphOwned => graph.graph_owned_outputs(&cache)?,
        }
    };
    graph.finish_replay_submission(submission, stream)?;
    Ok(outputs)
}
