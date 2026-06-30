use std::{
    mem,
    sync::{Arc, Mutex, MutexGuard},
    thread,
};

use crate::{
    context::Context,
    device::DeviceProperties,
    engine::EngineIndex,
    error::{Error, Result},
    execution::{EngineConfig, ExecutionPlan, KernelCache, OperationGraph},
    frontend::{
        graph::Graph,
        operation::CompileConfig,
        plan::{
            built::BuiltPlanEntry,
            metadata::{EngineConfigMetadata, workspace_limit_error},
        },
    },
    knob::BackendKnobType,
};

fn should_attempt_plan_build(
    config: &EngineConfig,
    compile_config: &CompileConfig,
) -> Result<bool> {
    Ok(workspace_limit_error(
        config.workspace_size()?,
        compile_config.max_workspace_size(),
    )
    .is_none())
}

pub(crate) fn build_execution_plan_entries(
    ctx: Option<&Context>,
    operation_graph: &OperationGraph,
    kernel_cache_enabled: bool,
    kernel_cache_json: Option<&str>,
    kernel_cache_runtime: Option<&Arc<Mutex<KernelCache>>>,
    device_properties: Option<&DeviceProperties>,
    candidates: Vec<(EngineConfig, EngineConfigMetadata)>,
    compile_config: &CompileConfig,
) -> Result<Vec<BuiltPlanEntry>> {
    let mut entries = Vec::new();
    let kernel_cache = materialize_kernel_cache_state(
        operation_graph,
        kernel_cache_enabled,
        kernel_cache_json,
        kernel_cache_runtime,
    )?;

    for (config, metadata) in candidates {
        if !should_attempt_plan_build(&config, compile_config)? {
            continue;
        }

        let result = match ctx {
            Some(ctx) => ExecutionPlan::create_with_config(
                ctx,
                &config,
                lock_kernel_cache(kernel_cache.as_ref())?.as_deref(),
                device_properties,
            ),
            None => ExecutionPlan::create_deviceless(
                &config,
                lock_kernel_cache(kernel_cache.as_ref())?.as_deref(),
                device_properties,
            ),
        };

        match result {
            Ok(plan) => entries.push(BuiltPlanEntry {
                engine_config: config,
                engine_metadata: metadata,
                execution_plan: Some(plan),
                build_error: None,
                support_error: None,
            }),
            Err(error) => {
                entries.push(BuiltPlanEntry {
                    engine_config: config,
                    engine_metadata: metadata,
                    execution_plan: None,
                    build_error: Some(error),
                    support_error: None,
                });
            }
        }
    }

    if entries.is_empty() {
        Err(Error::NoAvailableEngines)
    } else {
        Ok(entries)
    }
}

pub(crate) fn build_execution_plan_choice(
    ctx: Option<&Context>,
    operation_graph: &OperationGraph,
    kernel_cache_enabled: bool,
    kernel_cache_json: Option<&str>,
    kernel_cache_runtime: Option<&Arc<Mutex<KernelCache>>>,
    device_properties: Option<&DeviceProperties>,
    candidates: Vec<(EngineConfig, EngineConfigMetadata)>,
    compile_config: &CompileConfig,
) -> Result<Vec<BuiltPlanEntry>> {
    let mut last_error = None;
    let kernel_cache = materialize_kernel_cache_state(
        operation_graph,
        kernel_cache_enabled,
        kernel_cache_json,
        kernel_cache_runtime,
    )?;

    for (config, metadata) in candidates {
        if !should_attempt_plan_build(&config, compile_config)? {
            continue;
        }
        let result = match ctx {
            Some(ctx) => ExecutionPlan::create_with_config(
                ctx,
                &config,
                lock_kernel_cache(kernel_cache.as_ref())?.as_deref(),
                device_properties,
            ),
            None => ExecutionPlan::create_deviceless(
                &config,
                lock_kernel_cache(kernel_cache.as_ref())?.as_deref(),
                device_properties,
            ),
        };
        match result {
            Ok(plan) => {
                return Ok(vec![BuiltPlanEntry {
                    engine_config: config,
                    engine_metadata: metadata,
                    execution_plan: Some(plan),
                    build_error: None,
                    support_error: None,
                }]);
            }
            Err(error) => {
                last_error = Some(error);
            }
        }
    }

    Err(last_error.unwrap_or(Error::NoAvailableEngines))
}

fn build_execution_plan_entries_parallel(
    ctx: Option<&Context>,
    operation_graph: &OperationGraph,
    kernel_cache_enabled: bool,
    kernel_cache_json: Option<&str>,
    kernel_cache_runtime: Option<&Arc<Mutex<KernelCache>>>,
    device_properties_json: Option<&str>,
    device_properties: Option<&DeviceProperties>,
    candidates: Vec<(EngineConfig, EngineConfigMetadata)>,
    compile_config: &CompileConfig,
) -> Result<Vec<BuiltPlanEntry>> {
    struct BuildRequest {
        engine_index: EngineIndex,
        knob_choices: Vec<(BackendKnobType, i64)>,
    }

    let mut filtered = Vec::new();
    for (config, metadata) in candidates {
        if !should_attempt_plan_build(&config, compile_config)? {
            continue;
        }

        let engine_index = config.engine()?.index();
        let knob_choices = config
            .knob_choices()?
            .into_iter()
            .map(|choice| (choice.knob_type, choice.value))
            .collect::<Vec<_>>();
        filtered.push((
            config,
            metadata,
            BuildRequest {
                engine_index,
                knob_choices,
            },
        ));
    }

    if filtered.is_empty() {
        return Err(Error::NoAvailableEngines);
    }

    let kernel_cache_json = kernel_cache_json.map(ToOwned::to_owned);
    let kernel_cache_runtime = kernel_cache_runtime.cloned();
    let cuda_context = ctx.map(|ctx| Arc::clone(ctx.cuda_context()));
    let device_properties_json = device_properties_json.map(ToOwned::to_owned);
    let device_properties = device_properties
        .map(|device_properties| {
            DeviceProperties::from_json_representation(&device_properties.json_representation()?)
        })
        .transpose()?;
    let device_properties = device_properties.map(Arc::new);
    let results = thread::scope(|scope| {
        let mut tasks = Vec::with_capacity(filtered.len());
        for (_, _, request) in &filtered {
            let kernel_cache_json = kernel_cache_json.clone();
            let kernel_cache_runtime = kernel_cache_runtime.clone();
            let cuda_context = cuda_context.clone();
            let device_properties_json = device_properties_json.clone();
            let device_properties = device_properties.clone();
            tasks.push(scope.spawn(move || -> Result<ExecutionPlan> {
                let build_context = cuda_context.as_ref().map(Context::create).transpose()?;
                let owned_device_properties = if let Some(device_properties) = device_properties {
                    Some(device_properties)
                } else {
                    device_properties_json
                        .as_deref()
                        .map(DeviceProperties::from_json_representation)
                        .transpose()?
                        .map(Arc::new)
                };
                let engine_config = match build_context.as_ref() {
                    Some(build_context) => EngineConfig::create_with_config(
                        build_context,
                        operation_graph,
                        request.engine_index,
                        &request.knob_choices,
                        owned_device_properties.as_deref(),
                    )?,
                    None => EngineConfig::create_deviceless(
                        operation_graph,
                        request.engine_index,
                        &request.knob_choices,
                        owned_device_properties
                            .as_deref()
                            .ok_or(Error::FrontendDevicelessRequiresDeviceProperties)?,
                    )?,
                };
                let kernel_cache = materialize_kernel_cache_state(
                    operation_graph,
                    kernel_cache_enabled,
                    kernel_cache_json.as_deref(),
                    kernel_cache_runtime.as_ref(),
                )?;
                match build_context.as_ref() {
                    Some(build_context) => ExecutionPlan::create_with_config(
                        build_context,
                        &engine_config,
                        lock_kernel_cache(kernel_cache.as_ref())?.as_deref(),
                        owned_device_properties.as_deref(),
                    ),
                    None => ExecutionPlan::create_deviceless(
                        &engine_config,
                        lock_kernel_cache(kernel_cache.as_ref())?.as_deref(),
                        owned_device_properties.as_deref(),
                    ),
                }
            }));
        }

        tasks
            .into_iter()
            .map(|task| {
                task.join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
            })
            .collect::<Vec<_>>()
    });

    let mut entries = Vec::with_capacity(filtered.len());
    for ((config, metadata, _), result) in filtered.into_iter().zip(results) {
        match result {
            Ok(plan) => entries.push(BuiltPlanEntry {
                engine_config: config,
                engine_metadata: metadata,
                execution_plan: Some(plan),
                build_error: None,
                support_error: None,
            }),
            Err(error) => {
                entries.push(BuiltPlanEntry {
                    engine_config: config,
                    engine_metadata: metadata,
                    execution_plan: None,
                    build_error: Some(error),
                    support_error: None,
                });
            }
        }
    }

    if entries.is_empty() {
        Err(Error::NoAvailableEngines)
    } else {
        Ok(entries)
    }
}

pub(crate) fn build_execution_plan_choice_parallel_window(
    ctx: Option<&Context>,
    operation_graph: &OperationGraph,
    kernel_cache_enabled: bool,
    kernel_cache_json: Option<&str>,
    kernel_cache_runtime: Option<&Arc<Mutex<KernelCache>>>,
    device_properties_json: Option<&str>,
    device_properties: Option<&DeviceProperties>,
    candidates: Vec<(EngineConfig, EngineConfigMetadata)>,
    compile_config: &CompileConfig,
    width: usize,
) -> Result<Vec<BuiltPlanEntry>> {
    let width = width.max(1);
    let mut last_error = None;

    let mut chunk = Vec::with_capacity(width);
    for candidate in candidates {
        chunk.push(candidate);
        if chunk.len() < width {
            continue;
        }

        let entries = build_execution_plan_entries_parallel(
            ctx,
            operation_graph,
            kernel_cache_enabled,
            kernel_cache_json,
            kernel_cache_runtime,
            device_properties_json,
            device_properties,
            mem::take(&mut chunk),
            compile_config,
        )?;

        for entry in entries {
            if let Some(execution_plan) = entry.execution_plan {
                return Ok(vec![BuiltPlanEntry {
                    execution_plan: Some(execution_plan),
                    ..entry
                }]);
            }
            if let Some(error) = entry.build_error {
                last_error = Some(error);
            }
        }
    }

    if !chunk.is_empty() {
        let entries = build_execution_plan_entries_parallel(
            ctx,
            operation_graph,
            kernel_cache_enabled,
            kernel_cache_json,
            kernel_cache_runtime,
            device_properties_json,
            device_properties,
            chunk,
            compile_config,
        )?;

        for entry in entries {
            if let Some(execution_plan) = entry.execution_plan {
                return Ok(vec![BuiltPlanEntry {
                    execution_plan: Some(execution_plan),
                    ..entry
                }]);
            }
            if let Some(error) = entry.build_error {
                last_error = Some(error);
            }
        }
    }

    Err(last_error.unwrap_or(Error::NoAvailableEngines))
}

pub(crate) fn materialize_graph_kernel_cache(
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

pub(crate) fn materialize_kernel_cache_state(
    operation_graph: &OperationGraph,
    kernel_cache_enabled: bool,
    kernel_cache_json: Option<&str>,
    kernel_cache_runtime: Option<&Arc<Mutex<KernelCache>>>,
) -> Result<Option<Arc<Mutex<KernelCache>>>> {
    if !kernel_cache_enabled {
        return Ok(None);
    }

    let kernel_cache = if let Some(kernel_cache) = kernel_cache_runtime {
        Arc::clone(kernel_cache)
    } else {
        Arc::new(Mutex::new(match kernel_cache_json {
            Some(json) => KernelCache::from_json(json)?,
            None => KernelCache::create()?,
        }))
    };

    {
        let mut kernel_cache_guard = kernel_cache
            .lock()
            .map_err(|_| Error::FrontendKernelCacheLockPoisoned)?;
        if !kernel_cache_guard.is_finalized() {
            kernel_cache_guard.build(operation_graph)?;
        }
    }
    Ok(Some(kernel_cache))
}

pub(crate) fn lock_kernel_cache(
    kernel_cache: Option<&Arc<Mutex<KernelCache>>>,
) -> Result<Option<MutexGuard<'_, KernelCache>>> {
    kernel_cache
        .map(|kernel_cache| {
            kernel_cache
                .lock()
                .map_err(|_| Error::FrontendKernelCacheLockPoisoned)
        })
        .transpose()
}
