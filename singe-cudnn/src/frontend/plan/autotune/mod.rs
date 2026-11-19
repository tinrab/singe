mod config;
mod sampling;
mod session;
mod state;

use std::{collections::HashSet, sync::Arc};

use singe_cuda::{
    error::{Error as CudaError, Status as CudaStatus},
    event::EventFlags,
    stream::{AsStream, StreamCaptureStatus},
    submission::{Access, AccessMode},
    view::{DeviceOutputView, DeviceRegionProvider},
};

pub use crate::frontend::plan::autotune::{
    config::{AutotuneConfig, AutotuneResult, AutotuneSampling, PlanTiming},
    session::AutotuneSession,
};
use crate::{
    context::Context,
    error::{Error, Result},
    frontend::{
        bindings::{Bindings, RuntimeOverrides},
        error::Error as FrontendError,
        plan::{
            CompiledGraph, PlanFailure, PlanFailureStage, TunedPlan,
            autotune::{sampling::normalize_elapsed, state::AutotuneState},
        },
    },
};

impl CompiledGraph {
    /// Measures plans for the supplied runtime shapes without changing default selection.
    /// Timings and recoverable failures retain their stable plan identities.
    /// The returned winner retains this compilation and its measured layouts; executing it leaves this handle's default unchanged.
    /// Failed timed executions consume sampling attempts without discarding earlier successful measurements.
    /// Writable bindings use private scratch; caller outputs and read/write operands remain unchanged.
    /// Overlapping bindings preserve their relative addresses, and each invocation restores its initial read state.
    /// Fully produced internal values need no caller baseline, even when later operations read them.
    /// Private bindings retain their actual capacity, encoding and context metadata.
    /// Copies, frontend workspace initialization, native workspace queries and resource preparation occur outside timing events.
    /// Typed bindings and workspace use the CUDA range tracker to wait for producers and prevent conflicting tracked submissions until tuning completes.
    ///
    /// # Safety
    ///
    /// Bound memory and workspace must remain valid and satisfy native access and aliasing rules through completion.
    /// Untracked raw inputs must be ready on the context's stream, including producer dependencies.
    /// Concurrent untracked access must not modify inputs or workspace during tuning.
    /// Ragged spans require a tensor's declared byte capacity or allocation metadata from its binding.
    /// Use `Bindings::set_raw_buffer` when supplying capacity for externally managed allocations.
    ///
    /// # Errors
    ///
    /// Returns an error if the sampling parameters are invalid, the stream is capturing or its capture was invalidated, overrides or bindings are invalid, the workspace is too small, preparing private trial buffers or timing events fails, a CUDA call or synchronization fails, or every candidate fails without a valid timing.
    pub unsafe fn autotune(
        &self,
        context: &Context,
        bindings: &Bindings<'_>,
        workspace: Option<DeviceOutputView<'_, u8>>,
        overrides: &RuntimeOverrides,
        config: &AutotuneConfig,
    ) -> Result<AutotuneResult> {
        let sample_limit = config.sampling.sample_limit()?;
        // SAFETY: `autotune_with` requires the binding, workspace and ordering contract of `autotune`, which this caller upholds, and the sampling strategy only runs plans through the session's stream, which completes before it returns.
        unsafe {
            self.autotune_with(context, bindings, workspace, overrides, |session| {
                session.sample(config, sample_limit)
            })
        }
    }

    /// Runs a caller-defined tuning strategy without changing stored plan order or default selection.
    /// The strategy chooses trial order, sampling, aggregation and which plans enter the ranking.
    /// Return one timing per ranked plan.
    /// Lower elapsed times rank first, with ties preserving the returned order.
    /// Positive and negative zero timings compare equally and are reported as positive zero.
    /// The session exposes private trial bindings, native plans and runtime layouts for custom measurements.
    /// Warmup and measurement helpers restore the input baseline before each invocation.
    /// Caller outputs and read/write operands remain unchanged by those trials.
    /// Recoverable trial failures remain in the result even when later measurements succeed.
    ///
    /// # Safety
    ///
    /// Bindings and workspace must satisfy the lifetime, initialization, aliasing and producer-ordering contracts of [`Self::autotune`].
    /// Custom native execution must obey the variant pack and execution plan contracts.
    /// Report custom native execution failures through [`AutotuneSession::record_failure`].
    /// Use the session's stream or complete work on other streams before the strategy returns or unwinds.
    /// Do not retain trial pointers, captured graphs or other users of private trial storage beyond this call.
    ///
    /// # Errors
    ///
    /// Returns an error if preparation, the workspace, CUDA, or the strategy fails, timings are invalid or duplicated, or the strategy returns no timings.
    pub unsafe fn autotune_with(
        &self,
        context: &Context,
        bindings: &Bindings<'_>,
        mut workspace: Option<DeviceOutputView<'_, u8>>,
        overrides: &RuntimeOverrides,
        strategy: impl FnOnce(&mut AutotuneSession<'_, '_>) -> Result<Vec<PlanTiming>>,
    ) -> Result<AutotuneResult> {
        let stream = context.stream_binding();
        match stream.capture_status()? {
            StreamCaptureStatus::None => {}
            StreamCaptureStatus::Invalidated => {
                return Err(CudaError::from(CudaStatus::StreamCaptureInvalidated).into());
            }
            _ => return Err(CudaError::from(CudaStatus::StreamCaptureUnsupported).into()),
        }

        let pack = self.variant_pack_with_overrides(
            context.cuda_context(),
            bindings,
            workspace.as_mut().map(DeviceOutputView::reborrow),
            overrides,
        )?;
        let mut failures = Vec::new();
        let mut required = 0_usize;
        let plans = self
            .ordered_entries()
            .filter_map(
                |entry| match entry.execution_plan.workspace_size_for(context, &pack) {
                    Ok(bytes) => {
                        required = required.max(bytes);
                        Some(entry.candidate.id.clone())
                    }
                    Err(error) => {
                        failures.push(PlanFailure {
                            plan: entry.candidate.id.clone(),
                            stage: PlanFailureStage::Requirements,
                            source: Arc::new(error),
                        });
                        None
                    }
                },
            )
            .collect::<Vec<_>>();
        if plans.is_empty() {
            return Err(FrontendError::AutotuneFailed { failures }.into());
        }

        if pack.workspace_capacity() < required {
            let required = required
                .checked_add(pack.workspace_prefix_bytes())
                .ok_or_else(|| Error::out_of_range("autotune workspace bytes"))?;
            return Err(Error::BufferTooSmall {
                name: "workspace".into(),
                required,
                actual: pack.total_workspace_capacity(),
            });
        }

        let mut state = AutotuneState::create(self, context, bindings, &pack)?;
        if let Some(workspace) = workspace.as_ref() {
            state
                .accesses
                .push(Access::new(workspace.region(), AccessMode::ReadWrite));
        }

        let pack = self.variant_pack_with_overrides(
            context.cuda_context(),
            &state.bindings,
            workspace,
            overrides,
        )?;
        let start = context.cuda_context().create_event(EventFlags::DEFAULT)?;
        let stop = context.cuda_context().create_event(EventFlags::DEFAULT)?;
        // Owners outlive the completion scope, including failed enqueues and unwinding strategies.
        let run_trials = || {
            stream.sync_scope(|| {
                // SAFETY: `with_synchronized_accesses` waited for tracked producers, this function's contract orders untracked inputs before the stream, and `state` outlives this `sync_scope`, which waits even when an enqueue fails.
                unsafe { state.snapshot(&stream) }?;
                let mut session = AutotuneSession {
                    graph: self,
                    context,
                    overrides,
                    state: &state,
                    pack: &pack,
                    stream: &stream,
                    start: &start,
                    stop: &stop,
                    plans,
                    failures,
                    terminal_error: None,
                };
                let timings = strategy(&mut session);
                session.ensure_active()?;
                let mut timings = timings?;
                let mut seen = HashSet::new();
                for timing in &mut timings {
                    session.execution_plan(&timing.plan)?;
                    timing.elapsed_ms = normalize_elapsed(timing.elapsed_ms)?;
                    if timing.samples == 0 {
                        return Err(Error::out_of_range("autotune samples"));
                    }
                    if !seen.insert(timing.plan.clone()) {
                        return Err(FrontendError::AutotuneDuplicatePlan {
                            plan: timing.plan.clone(),
                        }
                        .into());
                    }
                }

                timings.sort_by(|lhs, rhs| lhs.elapsed_ms.total_cmp(&rhs.elapsed_ms));
                let Some(winner) = timings.first().map(|timing| timing.plan.clone()) else {
                    return Err(FrontendError::AutotuneFailed {
                        failures: session.failures,
                    }
                    .into());
                };
                Ok(AutotuneResult {
                    winner: TunedPlan::new(
                        self.clone().with_default_plan(&winner)?,
                        Arc::clone(context.cuda_context()),
                        overrides.clone(),
                    ),
                    timings,
                    failures: session.failures,
                })
            })
        };
        // SAFETY: this function's contract keeps the bound storage live, `state.accesses` declares its access modes, and `run_trials` runs inside `sync_scope`, so all trial work completes before the callback returns.
        unsafe {
            context
                .cuda_context()
                .with_synchronized_accesses(&state.accesses, run_trials)
        }
    }
}
