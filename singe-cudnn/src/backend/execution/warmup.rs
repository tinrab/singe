use singe_cuda::stream::{
    AsStream, StreamCaptureMode, StreamCaptureStatus, StreamFlags, StreamOptions,
};

use crate::{
    backend::{
        descriptor::{BackendDescriptor, backend_execute},
        execution::{ExecutionPlan, VariantPack},
    },
    context::Context,
    error::{Error, Result},
    utility::validate_context_pair,
};

/// Outcome of a throwaway capture used to prepare a plan's native kernel state.
#[derive(Debug)]
pub enum WarmupOutcome {
    /// Capture completed and its graph was discarded without launching any kernels.
    /// C++ treats cuDNN execution rejection during this capture as nonfatal.
    Captured {
        /// Native execution rejection observed inside the completed throwaway capture, or `None` on acceptance.
        execution_error: Option<Error>,
    },
    /// The handle's current stream was already capturing, so warmup left it alone.
    SkippedActiveCapture,
}

impl ExecutionPlan {
    /// Warms native kernel state by capturing this plan into a graph and discarding it.
    /// Uses the supplied pointers and runtime overrides without executing the recorded kernels.
    /// A private nonblocking stream permits warmup when the handle uses CUDA's default stream.
    /// The original stream is restored before returning, including on errors and unwinding.
    ///
    /// # Safety
    ///
    /// Host by-value pointers must contain initialized values in their declared native encodings and remain live until this call returns.
    /// Keep borrowed device buffers and workspace valid in the execution context through capture and graph disposal.
    /// Captured device reads and writes are never launched, so device inputs need not be initialized for this call.
    /// Deviceless plans retain the live context through native plan destruction, including when capture or the native warmup execution fails.
    ///
    /// # Errors
    ///
    /// Returns an error if the context, workspace, stream configuration, or capture fails.
    /// A cuDNN rejection inside a completed capture is returned in [`WarmupOutcome::Captured`].
    pub unsafe fn warmup(
        &self,
        context: &mut Context,
        variant_pack: &VariantPack,
    ) -> Result<WarmupOutcome> {
        if context.stream_binding().capture_status()? != StreamCaptureStatus::None {
            return Ok(WarmupOutcome::SkippedActiveCapture);
        }

        self.validate_workspace(context, variant_pack)?;
        validate_context_pair(
            Some(context.cuda_context()),
            Some(variant_pack.cuda_context()),
            "variant pack",
        )?;
        // SAFETY: the pack is used only inside the discarded capture, and the caller keeps its host scalar pointees initialized and live, per `# Safety`.
        unsafe {
            capture_plan_warmup(self, context, variant_pack.descriptor(), |context| {
                variant_pack.initialize_workspace(context)
            })
        }
    }
}

/// Captures native initialization and discards every recorded operation before returning.
///
/// # Safety
///
/// Host scalar pointees must be initialized, correctly encoded and live until this call returns.
/// Device addresses may refer to undersized or overlapping placeholder storage because no graph is launched.
/// This pack must never escape into execution, graph population, instantiation or replay.
pub(crate) unsafe fn capture_plan_warmup(
    plan: &ExecutionPlan,
    context: &mut Context,
    pack: &BackendDescriptor,
    initialize: impl FnOnce(&Context) -> Result<()>,
) -> Result<WarmupOutcome> {
    if context.stream_binding().capture_status()? != StreamCaptureStatus::None {
        return Ok(WarmupOutcome::SkippedActiveCapture);
    }

    validate_context_pair(
        Some(context.cuda_context()),
        plan.cuda_context(),
        "execution plan",
    )?;
    plan.retain_execution_context(context.cuda_context())?;
    let stream = context.cuda_context().create_stream(StreamOptions {
        flags: StreamFlags::NON_BLOCKING,
        ..StreamOptions::default()
    })?;
    context.with_stream(Some(&stream), |context| {
        let mut execution_error = None;
        let mut initialization_result = Ok(());
        let graph = stream.capture(StreamCaptureMode::Relaxed, |_| {
            initialization_result = initialize(context);
            if initialization_result.is_ok() {
                // SAFETY: the work is only recorded into a private capture that is dropped without launching, so device addresses are never accessed.
                // The caller keeps the host scalar pointees initialized and live until this call returns, per `# Safety`.
                execution_error =
                    unsafe { backend_execute(context, plan.descriptor(), pack) }.err();
            }
            Ok(())
        });
        initialization_result?;
        drop(graph?);
        Ok(WarmupOutcome::Captured { execution_error })
    })
}
