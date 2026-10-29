use std::{
    num::NonZeroUsize,
    panic::resume_unwind,
    result::Result as StdResult,
    sync::{Arc, Mutex, MutexGuard},
    thread,
};

use singe_cuda::context::Context as CudaContext;

use crate::{
    backend::{
        device::DeviceProperties,
        execution::{
            EngineChoice, EngineConfig, EnginePlanningConfig, ExecutionPlan, KernelCache,
            KernelCacheJson, OperationGraph,
        },
    },
    context::Context,
    error::{Error, Result},
    frontend::{
        error::Error as FrontendError,
        graph::Graph,
        logging,
        plan::{
            BuildPlanPolicy, CompileConfig, EngineConfigMetadata, PlanExclusion, PlanFailure,
            PlanFailureStage,
            entry::{BuiltPlanEntry, PlanBuildOutcome, PlanCandidate},
        },
    },
};

struct BuildRequest {
    choice: EngineChoice,
    planning: EnginePlanningConfig,
}

struct BuiltNativePlan {
    // A filtered configuration has refreshed metadata but no compiled plan.
    build_result: Option<Result<ExecutionPlan>>,
    config: EngineConfig,
    metadata: EngineConfigMetadata,
}

type BuildResult<T> = StdResult<T, (PlanFailureStage, Error)>;

fn exclude_plans_over_limit(entries: &mut [BuiltPlanEntry], limit: Option<NonZeroUsize>) {
    if let Some(limit) = limit {
        for entry in entries
            .iter_mut()
            .filter(|entry| entry.is_available())
            .skip(limit.get())
        {
            entry.exclusion = Some(PlanExclusion::BuildLimit { limit });
        }
    }
}

impl BuildRequest {
    fn from_candidate(candidate: &PlanCandidate, config: &CompileConfig) -> Result<Self> {
        let inherited = candidate.engine_config.planning_config()?;
        Ok(Self {
            choice: candidate.choice(),
            planning: EnginePlanningConfig {
                sm_count_target: config.sm_count_target().or(inherited.sm_count_target),
                shared_memory_limit: config
                    .shared_memory_limit()
                    .or(inherited.shared_memory_limit),
            },
        })
    }

    fn create_engine_config(
        &self,
        context: Option<&Context>,
        graph: &OperationGraph,
        properties: Option<&DeviceProperties>,
    ) -> Result<EngineConfig> {
        EngineConfig::create_for_target(context, graph, &self.choice, properties, self.planning)
    }

    fn build(
        &self,
        cuda_context: Option<&Arc<CudaContext>>,
        operation_graph: &OperationGraph,
        device_properties: Option<&DeviceProperties>,
        kernel_cache: Option<&Arc<Mutex<KernelCache>>>,
        frontend_workspace_bytes: usize,
        config: &CompileConfig,
    ) -> BuildResult<BuiltNativePlan> {
        let context = cuda_context
            .map(Context::create)
            .transpose()
            .map_err(|error| (PlanFailureStage::Build, error))?;
        let engine_config = self
            .create_engine_config(context.as_ref(), operation_graph, device_properties)
            .map_err(|error| (PlanFailureStage::Build, error))?;
        let metadata = EngineConfigMetadata::create(&engine_config)
            .and_then(|metadata| metadata.with_frontend_workspace(frontend_workspace_bytes))
            .map_err(|error| (PlanFailureStage::Requirements, error))?;
        if metadata.exclusion(config).is_some() {
            return Ok(BuiltNativePlan {
                build_result: None,
                config: engine_config,
                metadata,
            });
        }
        // Native finalization with a shared cache remains serialized; engine setup runs in parallel.
        let build_result = (|| {
            let cache = lock_kernel_cache(kernel_cache)?;
            ExecutionPlan::create_for_target(
                context.as_ref(),
                &engine_config,
                cache.as_deref(),
                device_properties,
            )
        })();
        Ok(BuiltNativePlan {
            build_result: Some(build_result),
            config: engine_config,
            metadata,
        })
    }
}

fn record_build_result(
    entry: &mut BuiltPlanEntry,
    result: BuildResult<ExecutionPlan>,
    frontend_workspace_bytes: usize,
    config: &CompileConfig,
) {
    let result = result.and_then(|plan| {
        let bytes = plan
            .workspace_size()
            .and_then(|native| {
                native
                    .checked_add(frontend_workspace_bytes)
                    .ok_or_else(|| Error::out_of_range("built plan workspace bytes"))
            })
            .map_err(|error| (PlanFailureStage::Requirements, error));
        bytes.map(|bytes| (plan, bytes))
    });
    log_build_result(&entry.candidate, &result);
    entry.outcome = match result {
        Ok((plan, bytes)) => {
            entry.candidate.metadata.workspace_size = bytes;
            entry.exclusion = entry.candidate.metadata.exclusion(config);
            PlanBuildOutcome::Built(plan)
        }
        Err((stage, error)) => PlanBuildOutcome::Failed(PlanFailure {
            plan: entry.candidate.id.clone(),
            stage,
            source: Arc::new(error),
        }),
    };
}

#[expect(
    clippy::too_many_lines,
    reason = "plan construction and selection keep the complete native execution lifecycle together"
)]
pub fn build_execution_plan_entries(
    context: Option<&Context>,
    operation_graph: &OperationGraph,
    frontend_workspace_bytes: usize,
    device_properties: Option<&DeviceProperties>,
    entries: &mut [BuiltPlanEntry],
    kernel_cache: Option<&Arc<Mutex<KernelCache>>>,
    config: &CompileConfig,
) {
    let (width, first_only) = match config.build_policy() {
        BuildPlanPolicy::FirstSupported => (1, true),
        BuildPlanPolicy::FirstSupportedParallel { window_size } => (window_size.get(), true),
        BuildPlanPolicy::AllSupported => (1, false),
        BuildPlanPolicy::AllSupportedParallel { worker_count } => (worker_count.get(), false),
    };
    let limit = if first_only {
        Some(NonZeroUsize::MIN)
    } else {
        config.max_plans()
    };
    for entry in entries.iter_mut() {
        if matches!(entry.exclusion, Some(PlanExclusion::BuildLimit { .. })) {
            entry.exclusion = None;
        }
    }
    let mut admitted = 0;
    if width == 1 {
        for entry in entries.iter_mut() {
            if entry.exclusion.is_some() {
                continue;
            }
            if !matches!(entry.outcome, PlanBuildOutcome::Built(_)) {
                build_execution_plan_entry(
                    context,
                    operation_graph,
                    entry,
                    frontend_workspace_bytes,
                    device_properties,
                    kernel_cache,
                    config,
                );
            }
            admitted += usize::from(entry.is_available());
            if limit.is_some_and(|limit| admitted >= limit.get()) {
                break;
            }
        }
        exclude_plans_over_limit(entries, limit);
        return;
    }

    let cuda_context = context.map(|context| Arc::clone(context.cuda_context()));
    let device_properties = device_properties.cloned().map(Arc::new);

    for window in entries.chunks_mut(width) {
        let mut requests = Vec::new();
        for (position, entry) in window.iter_mut().enumerate() {
            if entry.exclusion.is_some() || matches!(entry.outcome, PlanBuildOutcome::Built(_)) {
                continue;
            }
            match BuildRequest::from_candidate(&entry.candidate, config) {
                Ok(request) => requests.push((position, request)),
                Err(error) => {
                    record_build_result(
                        entry,
                        Err((PlanFailureStage::Requirements, error)),
                        frontend_workspace_bytes,
                        config,
                    );
                }
            }
        }
        let results = thread::scope(|scope| {
            #[expect(
                clippy::needless_collect,
                reason = "eager collection starts every compile worker before any join can block"
            )]
            let tasks = requests
                .into_iter()
                .map(|(position, request)| {
                    let cuda_context = cuda_context.clone();
                    let device_properties = device_properties.clone();
                    scope.spawn(move || {
                        let result = request.build(
                            cuda_context.as_ref(),
                            operation_graph,
                            device_properties.as_deref(),
                            kernel_cache,
                            frontend_workspace_bytes,
                            config,
                        );
                        (position, result)
                    })
                })
                .collect::<Vec<_>>();
            tasks
                .into_iter()
                .map(|task| match task.join() {
                    Ok(result) => result,
                    Err(panic) => resume_unwind(panic),
                })
                .collect::<Vec<_>>()
        });
        for (position, result) in results {
            let entry = &mut window[position];
            match result {
                Ok(built) => {
                    entry.candidate.engine_config = Arc::new(built.config);
                    entry.candidate.metadata = built.metadata;
                    entry.exclusion = entry.candidate.metadata.exclusion(config);
                    if let Some(result) = built.build_result {
                        record_build_result(
                            entry,
                            result.map_err(|error| (PlanFailureStage::Build, error)),
                            frontend_workspace_bytes,
                            config,
                        );
                    }
                }
                Err(error) => {
                    record_build_result(entry, Err(error), frontend_workspace_bytes, config);
                }
            }
        }
        admitted += window.iter().filter(|entry| entry.is_available()).count();
        if limit.is_some_and(|limit| admitted >= limit.get()) {
            break;
        }
    }
    exclude_plans_over_limit(entries, limit);
}

pub fn build_execution_plan_entry(
    context: Option<&Context>,
    operation_graph: &OperationGraph,
    entry: &mut BuiltPlanEntry,
    frontend_workspace_bytes: usize,
    device_properties: Option<&DeviceProperties>,
    kernel_cache: Option<&Arc<Mutex<KernelCache>>>,
    config: &CompileConfig,
) {
    let reconfigured = (|| -> BuildResult<_> {
        let request = BuildRequest::from_candidate(&entry.candidate, config)
            .map_err(|error| (PlanFailureStage::Requirements, error))?;
        let inherited = entry
            .candidate
            .engine_config
            .planning_config()
            .map_err(|error| (PlanFailureStage::Requirements, error))?;
        if request.planning == inherited {
            return Ok(None);
        }
        let engine_config = request
            .create_engine_config(context, operation_graph, device_properties)
            .map_err(|error| (PlanFailureStage::Build, error))?;
        let metadata = EngineConfigMetadata::create(&engine_config)
            .and_then(|metadata| metadata.with_frontend_workspace(frontend_workspace_bytes))
            .map_err(|error| (PlanFailureStage::Requirements, error))?;
        Ok(Some((engine_config, metadata)))
    })();
    match reconfigured {
        Ok(Some((engine_config, metadata))) => {
            entry.candidate.engine_config = Arc::new(engine_config);
            entry.candidate.metadata = metadata;
        }
        Ok(None) => {}
        Err(error) => {
            record_build_result(entry, Err(error), frontend_workspace_bytes, config);
            return;
        }
    }
    entry.exclusion = entry.candidate.metadata.exclusion(config);
    if entry.exclusion.is_some() {
        return;
    }
    let result = (|| {
        let cache = lock_kernel_cache(kernel_cache)?;
        ExecutionPlan::create_for_target(
            context,
            &entry.candidate.engine_config,
            cache.as_deref(),
            device_properties,
        )
    })();
    record_build_result(
        entry,
        result.map_err(|error| (PlanFailureStage::Build, error)),
        frontend_workspace_bytes,
        config,
    );
}

fn log_build_result<T>(candidate: &PlanCandidate, result: &BuildResult<T>) {
    match result {
        Ok(_) => logging::diagnostic!(
            "built candidate {}: engine {}, knobs {:?}",
            candidate.id,
            candidate.metadata.engine_index,
            candidate.metadata.knob_choices,
        ),
        Err((stage, error)) => logging::diagnostic!(
            "candidate {} failed during {stage}: engine {}, knobs {:?}: {error}",
            candidate.id,
            candidate.metadata.engine_index,
            candidate.metadata.knob_choices,
        ),
    }
}

pub fn materialize_graph_kernel_cache(
    graph: &Graph,
    operation_graph: &OperationGraph,
) -> Result<Option<Arc<Mutex<KernelCache>>>> {
    materialize_kernel_cache_state(
        operation_graph,
        graph.kernel_cache_enabled(),
        graph.kernel_cache_json(),
        graph.runtime_kernel_cache(),
    )
}

pub fn materialize_kernel_cache_state(
    operation_graph: &OperationGraph,
    kernel_cache_enabled: bool,
    kernel_cache_json: Option<&KernelCacheJson>,
    kernel_cache_runtime: Option<&Arc<Mutex<KernelCache>>>,
) -> Result<Option<Arc<Mutex<KernelCache>>>> {
    if !kernel_cache_enabled {
        return Ok(None);
    }

    let kernel_cache = if let Some(kernel_cache) = kernel_cache_runtime {
        Arc::clone(kernel_cache)
    } else {
        Arc::new(Mutex::new(match kernel_cache_json {
            Some(json) => KernelCache::deserialize(json)?,
            None => KernelCache::create()?,
        }))
    };

    {
        let mut kernel_cache_guard = kernel_cache
            .lock()
            .map_err(|_| FrontendError::KernelCacheLockPoisoned)?;
        if !kernel_cache_guard.is_finalized() {
            kernel_cache_guard.build(Some(operation_graph))?;
        }
    }
    Ok(Some(kernel_cache))
}

pub fn lock_kernel_cache(
    kernel_cache: Option<&Arc<Mutex<KernelCache>>>,
) -> Result<Option<MutexGuard<'_, KernelCache>>> {
    kernel_cache
        .map(|kernel_cache| {
            kernel_cache
                .lock()
                .map_err(|_| FrontendError::KernelCacheLockPoisoned.into())
        })
        .transpose()
}
