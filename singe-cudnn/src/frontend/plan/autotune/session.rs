use std::sync::Arc;

use singe_cuda::{
    event::{Event, EventRecordFlags},
    stream::StreamBinding,
};

use crate::{
    backend::execution::{ExecutionPlan, VariantPack},
    context::Context,
    error::{Error, Result},
    frontend::{
        bindings::{Bindings, RuntimeOverrides},
        error::Error as FrontendError,
        plan::{
            AutotuneConfig, CompiledGraph, PlanFailure, PlanFailureStage, PlanId, PlanTiming,
            autotune::{
                AutotuneState,
                sampling::{TimingSamples, normalize_elapsed},
            },
        },
    },
};

/// Temporary execution environment for a caller-defined tuning strategy.
/// Trial helpers restore private writable storage before every invocation and wait for completion.
/// The enclosing tuning call retains trial resources and locks tracked caller accesses through completion or unwinding.
/// This session and its trial storage are valid only during the strategy callback.
#[derive(Debug)]
pub struct AutotuneSession<'a, 'bindings> {
    pub(crate) graph: &'a CompiledGraph,
    pub(crate) context: &'a Context,
    pub(crate) overrides: &'a RuntimeOverrides,
    pub(crate) state: &'a AutotuneState<'bindings>,
    pub(crate) pack: &'a VariantPack,
    pub(crate) stream: &'a StreamBinding,
    pub(crate) start: &'a Event,
    pub(crate) stop: &'a Event,
    pub(crate) plans: Vec<PlanId>,
    pub(crate) failures: Vec<PlanFailure>,
    pub(crate) terminal_error: Option<Arc<Error>>,
}

impl<'a> AutotuneSession<'a, '_> {
    /// Plans whose workspace requirements were resolved for this session's runtime layouts, in this handle's order.
    #[must_use]
    pub fn plans(&self) -> &[PlanId] {
        &self.plans
    }

    /// Retained compilation and metadata used to choose the trial order or exclude plans.
    #[must_use]
    pub fn compiled_graph(&self) -> &CompiledGraph {
        self.graph
    }

    /// Borrows the cuDNN context used for this tuning callback and its trial executions.
    #[must_use]
    pub fn context(&self) -> &Context {
        self.context
    }

    /// Stream used by baseline restoration, workspace initialization and trial helpers.
    #[must_use]
    pub fn stream(&self) -> &StreamBinding {
        self.stream
    }

    /// Borrows the checked runtime layout request shared by every trial in this session.
    #[must_use]
    pub fn runtime_overrides(&self) -> &RuntimeOverrides {
        self.overrides
    }

    /// Private trial bindings, including writable scratch allocations and retained scalar owners.
    #[must_use]
    pub fn bindings(&self) -> &Bindings<'_> {
        &self.state.bindings
    }

    /// Native bindings, concrete layouts and workspace for custom native execution.
    /// Pointer inspection does not grant permission to access pointees or retain them beyond the callback.
    #[must_use]
    pub fn variant_pack(&self) -> &VariantPack {
        self.pack
    }

    /// Finds a plan admitted to this session without changing the compiled graph's selection.
    ///
    /// # Errors
    ///
    /// Returns an error if the identity is absent from the compiled collection or the plan was excluded from this tuning session during workspace-requirement preparation.
    pub fn execution_plan(&self, plan: &PlanId) -> Result<&'a ExecutionPlan> {
        let execution_plan = self.graph.execution_plan_for(plan)?;
        if !self.plans.contains(plan) {
            return Err(FrontendError::AutotunePlanUnavailable { plan: plan.clone() }.into());
        }
        Ok(execution_plan)
    }

    /// Preparation and recoverable execution failures recorded so far.
    #[must_use]
    pub fn failures(&self) -> &[PlanFailure] {
        &self.failures
    }

    /// Enqueues input-state restoration and frontend workspace initialization on this session's stream.
    /// Call this before custom native trials and record timing events after these copies.
    /// Custom work on another stream must wait for this stream's restoration before reading the trial buffers.
    ///
    /// # Errors
    ///
    /// Returns an error if the session has aborted, restoring the saved input baseline fails, or frontend workspace initialization fails. CUDA failures mark the session aborted.
    pub fn prepare_trial(&mut self) -> Result<()> {
        self.ensure_active()?;
        let result = (|| {
            // SAFETY: the enclosing tuning scope retains the saved baseline and private buffers through stream completion.
            unsafe { self.state.reset(self.stream) }?;
            self.pack.initialize_workspace(self.context)
        })();
        self.finish_call(result)
    }

    /// Runs one unmeasured invocation and waits for completion.
    /// Returns false after a recoverable failure, which remains available through [`Self::failures`].
    /// CUDA and trial-preparation failures abort tuning.
    ///
    /// # Errors
    ///
    /// Returns an error if the session has aborted, the plan is unknown or unavailable, trial-state restoration or workspace initialization fails, or a CUDA operation fails. Recoverable native preparation or execution failures are recorded and returned as an unsuccessful trial.
    pub fn warmup(&mut self, plan: &PlanId) -> Result<bool> {
        self.ensure_active()?;
        let result = self.run_trial(plan, false);
        Ok(self.finish_call(result)?.is_some())
    }

    /// Measures one invocation in milliseconds, excluding state restoration, workspace initialization and native preparation.
    /// Workspace queries and native context/code retention complete before recording the start event.
    /// Returns None after a recoverable failure, which remains available through [`Self::failures`].
    /// CUDA and trial-preparation failures abort tuning.
    ///
    /// # Errors
    ///
    /// Returns an error if the session has aborted, the plan is unknown or unavailable, trial-state restoration or workspace initialization fails, or a CUDA operation fails. Recoverable native preparation or execution failures are recorded and returned as an unsuccessful trial.
    pub fn measure(&mut self, plan: &PlanId) -> Result<Option<f32>> {
        self.ensure_active()?;
        let result = self.run_trial(plan, true);
        self.finish_call(result)
    }

    /// Records a failure from custom trial execution after draining partial work on this session's stream.
    /// CUDA errors abort tuning instead of allowing another trial to use a failed context.
    /// Complete any custom work on other streams before calling this method.
    ///
    /// # Errors
    ///
    /// Returns an error if the session has aborted, the plan is unknown or unavailable, draining the trial stream fails, or the supplied failure is a CUDA error. CUDA failures mark the session aborted.
    pub fn record_failure(
        &mut self,
        plan: &PlanId,
        stage: PlanFailureStage,
        error: Error,
    ) -> Result<()> {
        self.ensure_active()?;
        self.execution_plan(plan)?;
        let completion = self.stream.synchronize().map_err(Error::from);
        self.finish_call(completion)?;
        if matches!(error, Error::Cuda(_)) {
            return self.finish_call(Err(error));
        }
        self.failures.push(PlanFailure {
            plan: plan.clone(),
            stage,
            source: Arc::new(error),
        });
        Ok(())
    }

    pub(crate) fn ensure_active(&self) -> Result<()> {
        self.terminal_error.as_ref().map_or_else(
            || Ok(()),
            |source| {
                Err(FrontendError::AutotuneAborted {
                    source: Arc::clone(source),
                }
                .into())
            },
        )
    }

    fn finish_call<T>(&mut self, result: Result<T>) -> Result<T> {
        match result {
            Err(error @ Error::Cuda(_)) => {
                let source = Arc::new(error);
                self.terminal_error = Some(Arc::clone(&source));
                Err(FrontendError::AutotuneAborted { source }.into())
            }
            result => result,
        }
    }

    fn run_trial(&mut self, plan: &PlanId, timed: bool) -> Result<Option<f32>> {
        let execution_plan = self.execution_plan(plan)?;
        self.prepare_trial()?;
        let stage = if timed {
            PlanFailureStage::Measurement
        } else {
            PlanFailureStage::Warmup
        };
        let prepared = match execution_plan.prepare_execution(self.context, self.pack) {
            Ok(prepared) => prepared,
            Err(error) => {
                self.record_failure(plan, stage, error)?;
                return Ok(None);
            }
        };
        if timed {
            self.start
                .record_on(self.stream, EventRecordFlags::DEFAULT)?;
        }
        // SAFETY: the enclosing tuning scope retains plans, scratch, scalars and workspace through completion.
        // Preparation and workspace initialization ran before the event without changing the handle's stream.
        let result = unsafe { prepared.execute() };
        if let Err(error) = result {
            self.record_failure(plan, stage, error)?;
            return Ok(None);
        }
        if !timed {
            self.stream.synchronize()?;
            return Ok(Some(0.0));
        }
        self.stop
            .record_on(self.stream, EventRecordFlags::DEFAULT)?;
        self.stop.synchronize()?;
        let elapsed_ms = self.stop.elapsed_time_since(self.start)?;
        let elapsed_ms = match normalize_elapsed(elapsed_ms) {
            Ok(elapsed_ms) => elapsed_ms,
            Err(error) => {
                self.record_failure(plan, stage, error)?;
                return Ok(None);
            }
        };
        Ok(Some(elapsed_ms))
    }

    pub(crate) fn sample(
        &mut self,
        config: &AutotuneConfig,
        sample_limit: usize,
    ) -> Result<Vec<PlanTiming>> {
        let mut timings = Vec::new();
        for plan in self.plans.clone() {
            let mut ready = true;
            for _ in 0..config.warmup {
                if !self.warmup(&plan)? {
                    ready = false;
                    break;
                }
            }
            if !ready {
                continue;
            }
            let mut samples = TimingSamples::new(config.sampling);
            for _ in 0..sample_limit {
                if let Some(elapsed_ms) = self.measure(&plan)?
                    && !samples.record(elapsed_ms)
                {
                    break;
                }
            }
            if let Some(timing) = samples.finish(plan) {
                timings.push(timing);
                if config
                    .max_plans
                    .is_some_and(|limit| timings.len() >= limit.get())
                {
                    break;
                }
            }
        }
        Ok(timings)
    }
}
