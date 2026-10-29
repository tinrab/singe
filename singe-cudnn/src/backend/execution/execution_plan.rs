use std::sync::{Arc, Mutex, OnceLock};

use singe_cuda::{
    context::Context as CudaContext,
    error::Error as CudaError,
    graph::{Graph, GraphRetainedResource, collect_graph_resources, has_deferred_graph_resources},
    stream::StreamBinding,
};
use singe_cudnn_sys as sys;

use crate::{
    backend::{
        attribute::BackendAttributeName,
        behavior::BackendBehaviorNote,
        descriptor::{BackendDescriptor, BackendDescriptorType, backend_execute},
        device::DeviceProperties,
        execution::{EngineConfig, IntermediateId, KernelCache, KernelCacheJson, VariantPack},
    },
    context::Context,
    error::{Error, Result, Status},
    try_ffi,
    utility::{bind_context, bind_for_drop, to_usize, validate_context_pair},
};

/// Shared ownership of a finalized native execution plan, cached requirements, and retained code dependencies.
///
/// Clones share native resources and successful metadata queries. Live targets retain their CUDA context; deviceless plans retain execution contexts when used.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    inner: Arc<GraphRetainedResource<ExecutionPlanOwner>>,
}

/// One native enqueue with workspace and capture resources prepared ahead of timing.
/// Borrows the exact plan, handle and variant pack that preparation validated.
#[must_use]
pub struct PreparedExecution<'a> {
    plan: &'a ExecutionPlan,
    context: &'a Context,
    variant_pack: &'a VariantPack,
}

impl PreparedExecution<'_> {
    fn initialize_workspace(&self) -> Result<()> {
        self.variant_pack.initialize_workspace(self.context)
    }

    /// Enqueues only the native operation after workspace initialization has been ordered.
    ///
    /// # Safety
    /// Uphold [`ExecutionPlan::execute`]'s allocation, initialization, aliasing and lifetime requirements.
    /// Keep the handle's stream and capture unchanged after preparation, and order workspace initializers before this call.
    pub(crate) unsafe fn execute(self) -> Result<()> {
        // SAFETY: preparation retained native resources and checked workspace; caller supplies the execution contract.
        unsafe {
            backend_execute(
                self.context,
                self.plan.descriptor(),
                self.variant_pack.descriptor(),
            )
        }
    }
}

/// Owns one immutable native plan/argument pair whose workspace was checked before submission.
/// Buffer ownership and asynchronous access remain with the frontend submission resources.
#[derive(Debug)]
pub struct RetainedExecution {
    plan: ExecutionPlan,
    variant_pack: VariantPack,
}

impl RetainedExecution {
    pub(crate) fn create(
        plan: &ExecutionPlan,
        context: &Context,
        variant_pack: VariantPack,
    ) -> Result<Self> {
        plan.validate_workspace(context, &variant_pack)?;
        plan.retain_execution_context(context.cuda_context())?;
        Ok(Self {
            plan: plan.clone(),
            variant_pack,
        })
    }

    pub(crate) fn variant_pack(&self) -> &VariantPack {
        &self.variant_pack
    }

    /// Enqueues the validated pair on the handle's current stream without another workspace query.
    ///
    /// # Safety
    /// Uphold [`ExecutionPlan::execute`]'s allocation, initialization, aliasing and lifetime requirements.
    pub(crate) unsafe fn execute(&self, context: &Context) -> Result<()> {
        validate_context_pair(
            Some(context.cuda_context()),
            Some(self.variant_pack.cuda_context()),
            "prepared execution",
        )?;
        if self.plan.needs_cuda_graph_retention() || has_deferred_graph_resources() {
            self.plan.retain_on_capturing_stream(&context.stream()?)?;
        }
        self.variant_pack.initialize_workspace(context)?;
        // SAFETY: construction validated this exact immutable plan/pack pair and retained its CUDA context.
        // The caller supplies live buffers and ordered accesses through completion or capture replay.
        unsafe {
            backend_execute(
                context,
                self.plan.descriptor(),
                self.variant_pack.descriptor(),
            )
        }
    }
}

#[derive(Debug)]
struct ExecutionPlanOwner {
    descriptor: BackendDescriptor,
    engine_config: OnceLock<Arc<EngineConfig>>,
    workspace_size: OnceLock<usize>,
    restored_kernel_cache: Option<Arc<KernelCache>>,
    _restored_device_properties: Option<Arc<DeviceProperties>>,
    cuda_context: Option<Arc<CudaContext>>,
    runtime_compiled: bool,
    execution_contexts: Mutex<Vec<Arc<CudaContext>>>,
}

/// Target used to finalize a saved native plan.
/// Device properties allow restoration without creating a live cuDNN handle.
#[derive(Debug, Clone, Copy)]
pub enum ExecutionPlanTarget<'a> {
    /// Restores against a live cuDNN context and its actual CUDA target.
    Context(&'a Context),
    /// Restores against explicit device properties without creating a live cuDNN handle.
    DeviceProperties(&'a DeviceProperties),
}

impl ExecutionPlan {
    /// Finalizes a native plan from an engine configuration using a live cuDNN context.
    ///
    /// # Errors
    ///
    /// Returns an error if live contexts disagree or cannot be bound, or native descriptor creation, attribute assignment, or plan finalization fails. cuDNN validates engine, cache, and target compatibility.
    pub fn create(context: &Context, engine_config: &EngineConfig) -> Result<Self> {
        Self::create_for_target(Some(context), engine_config, None, None)
    }

    /// Finalizes a native plan with a live context and optional kernel cache and device properties.
    ///
    /// # Errors
    ///
    /// Returns an error if live contexts disagree or cannot be bound, or native descriptor creation, attribute assignment, or plan finalization fails. cuDNN validates engine, cache, and target compatibility.
    pub fn create_with_config(
        context: &Context,
        engine_config: &EngineConfig,
        kernel_cache: Option<&KernelCache>,
        device_properties: Option<&DeviceProperties>,
    ) -> Result<Self> {
        Self::create_for_target(
            Some(context),
            engine_config,
            kernel_cache,
            device_properties,
        )
    }

    /// Finalizes a native plan without a cuDNN handle, using the engine target and optional explicit properties and cache.
    ///
    /// An engine configuration retaining a live CUDA context still supplies that context for native calls and ownership.
    ///
    /// # Errors
    ///
    /// Returns an error if live contexts disagree or cannot be bound, or native descriptor creation, attribute assignment, or plan finalization fails. cuDNN validates engine, cache, and target compatibility.
    pub fn create_deviceless(
        engine_config: &EngineConfig,
        kernel_cache: Option<&KernelCache>,
        device_properties: Option<&DeviceProperties>,
    ) -> Result<Self> {
        Self::create_for_target(None, engine_config, kernel_cache, device_properties)
    }

    pub(crate) fn create_for_target(
        context: Option<&Context>,
        engine_config: &EngineConfig,
        kernel_cache: Option<&KernelCache>,
        device_properties: Option<&DeviceProperties>,
    ) -> Result<Self> {
        validate_context_pair(
            context.map(Context::cuda_context),
            engine_config.cuda_context(),
            "engine config",
        )?;

        let cuda_context = context
            .map(Context::cuda_context)
            .or_else(|| engine_config.cuda_context());
        bind_context(cuda_context)?;
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::ExecutionPlan)?;
        if let Some(context) = context {
            // SAFETY: context is live through native plan configuration and finalization.
            unsafe {
                descriptor.set_attribute_handle(
                    BackendAttributeName::ExecutionPlanHandle,
                    context.as_raw(),
                )
            }?;
        }
        descriptor.set_attribute_descriptor(
            BackendAttributeName::ExecutionPlanEngineConfig,
            engine_config.descriptor(),
        )?;
        if let Some(kernel_cache) = kernel_cache {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::ExecutionPlanKernelCache,
                kernel_cache.descriptor(),
            )?;
        }
        if let Some(device_properties) = device_properties {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::ExecutionPlanDeviceProp,
                device_properties.descriptor(),
            )?;
        }
        descriptor.finalize()?;

        let retained_config = read_engine_config(&descriptor, cuda_context.map(Arc::clone))
            .ok()
            .map(Arc::new);
        let runtime_compiled = config_requires_graph_retention(retained_config.as_deref());
        Ok(Self {
            inner: Arc::new(GraphRetainedResource::new(ExecutionPlanOwner {
                descriptor,
                engine_config: retained_config.map(OnceLock::from).unwrap_or_default(),
                workspace_size: OnceLock::new(),
                restored_kernel_cache: None,
                _restored_device_properties: None,
                cuda_context: cuda_context.map(Arc::clone),
                runtime_compiled,
                execution_contexts: Mutex::new(Vec::new()),
            })),
        })
    }

    /// Restores the saved native plan without reconstructing an engine configuration.
    /// Native finalization checks serialization and target compatibility and may perform preparation work.
    /// Retains an optional restored kernel cache and a private copy of deviceless properties.
    ///
    /// # Errors
    ///
    /// Returns an error if a live target cannot be bound, the JSON string is empty or its byte count does not fit `i64`, or native descriptor creation, attribute assignment, or restoration finalization fails.
    pub fn from_json(
        target: ExecutionPlanTarget<'_>,
        json_plan: &str,
        kernel_cache: Option<KernelCache>,
    ) -> Result<Self> {
        let (cuda_context, device_properties) = match target {
            ExecutionPlanTarget::Context(context) => {
                context.bind()?;
                (Some(Arc::clone(context.cuda_context())), None)
            }
            ExecutionPlanTarget::DeviceProperties(properties) => (None, Some(properties.clone())),
        };

        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::ExecutionPlan)?;
        descriptor.set_attribute_char_string(
            BackendAttributeName::ExecutionPlanJsonRepresentation,
            json_plan,
        )?;
        if let ExecutionPlanTarget::Context(context) = target {
            // SAFETY: context is live through native plan configuration and finalization.
            unsafe {
                descriptor.set_attribute_handle(
                    BackendAttributeName::ExecutionPlanHandle,
                    context.as_raw(),
                )
            }?;
        }
        if let Some(properties) = &device_properties {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::ExecutionPlanDeviceProp,
                properties.descriptor(),
            )?;
        }
        if let Some(cache) = &kernel_cache {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::ExecutionPlanKernelCache,
                cache.descriptor(),
            )?;
        }
        descriptor.finalize()?;

        let retained_config = read_engine_config(&descriptor, cuda_context.clone())
            .ok()
            .map(Arc::new);
        let runtime_compiled = config_requires_graph_retention(retained_config.as_deref());
        Ok(Self {
            inner: Arc::new(GraphRetainedResource::new(ExecutionPlanOwner {
                descriptor,
                engine_config: retained_config.map(OnceLock::from).unwrap_or_default(),
                workspace_size: OnceLock::new(),
                restored_kernel_cache: kernel_cache.map(Arc::new),
                _restored_device_properties: device_properties.map(Arc::new),
                cuda_context,
                runtime_compiled,
                execution_contexts: Mutex::new(Vec::new()),
            })),
        })
    }

    /// Static native workspace requirement shared by this plan's clones.
    /// Retains the first successful query, including zero; failures remain retryable.
    /// Runtime layout overrides use [`Self::workspace_size_for`] instead.
    ///
    /// # Errors
    ///
    /// Returns an error if an uncached query cannot bind the retained context, the native attribute query fails, or the reported byte count is negative or does not fit `usize`.
    pub fn workspace_size(&self) -> Result<usize> {
        if let Some(&size) = self.inner.workspace_size.get() {
            return Ok(size);
        }
        bind_context(self.cuda_context())?;
        let size = to_usize(
            self.inner
                .descriptor
                .attribute_i64(BackendAttributeName::ExecutionPlanWorkspaceSize)?,
            "workspace_size",
        )?;
        Ok(*self.inner.workspace_size.get_or_init(|| size))
    }

    /// Required workspace for the concrete runtime shapes represented by a pack.
    ///
    /// # Errors
    ///
    /// Returns an error if context, plan, and variant-pack targets disagree, context retention or binding fails, or the static or runtime native workspace query fails or reports a byte count that does not fit `usize`.
    pub fn workspace_size_for(
        &self,
        context: &Context,
        variant_pack: &VariantPack,
    ) -> Result<usize> {
        validate_context_pair(
            Some(context.cuda_context()),
            self.cuda_context(),
            "execution plan",
        )?;
        validate_context_pair(
            Some(context.cuda_context()),
            Some(variant_pack.cuda_context()),
            "variant pack",
        )?;
        if !variant_pack.has_overrides() {
            return self.workspace_size();
        }
        self.retain_execution_context(context.cuda_context())?;
        context.bind()?;
        let mut required = 0;
        // SAFETY: both finalized descriptors and the context remain live throughout the query.
        unsafe {
            try_ffi!(sys::cudnnGetExecutionPlanWorkspaceSize(
                context.as_raw(),
                self.inner.descriptor.as_raw(),
                variant_pack.descriptor().as_raw(),
                &raw mut required
            ))?;
        }
        to_usize(required, "runtime workspace size")
    }

    pub(crate) fn validate_workspace(
        &self,
        context: &Context,
        variant_pack: &VariantPack,
    ) -> Result<()> {
        let required = self.workspace_size_for(context, variant_pack)?;
        let actual = variant_pack.workspace_capacity();
        if actual < required {
            let required = required
                .checked_add(variant_pack.workspace_prefix_bytes())
                .ok_or_else(|| Error::out_of_range("execution workspace bytes"))?;
            return Err(Error::InsufficientWorkspaceSize {
                required,
                actual: variant_pack.total_workspace_capacity(),
            });
        }
        Ok(())
    }

    /// Queries the native computed-intermediate identities for this plan.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding or the native attribute query fails, including invalid or persistently changing payload counts.
    pub fn computed_intermediate_ids(&self) -> Result<Vec<IntermediateId>> {
        bind_context(self.cuda_context())?;
        self.inner
            .descriptor
            .attribute_i64_slice(BackendAttributeName::ExecutionPlanComputedIntermediateIds)
            .map(|ids| ids.into_iter().map(IntermediateId::from).collect())
    }

    /// Queries the native run-only intermediate identities for this plan.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding or the native attribute query fails, including invalid or persistently changing payload counts.
    pub fn run_only_intermediate_ids(&self) -> Result<Vec<IntermediateId>> {
        bind_context(self.cuda_context())?;
        self.inner
            .descriptor
            .attribute_i64_slice(BackendAttributeName::ExecutionPlanRunOnlyIntermediateIds)
            .map(|ids| ids.into_iter().map(IntermediateId::from).collect())
    }

    /// Queries the plan’s native serialized JSON representation.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding or the native attribute query fails, including invalid or persistently changing payload counts.
    pub fn json_representation(&self) -> Result<String> {
        bind_context(self.cuda_context())?;
        self.inner
            .descriptor
            .attribute_char_string(BackendAttributeName::ExecutionPlanJsonRepresentation)
    }

    /// Shares this plan's retained engine configuration without copying native descriptors.
    /// Plan clones and returned owners reuse the same successful metadata snapshot.
    /// Failed extraction remains retryable, with native queries outside the cache initializer.
    /// The returned owner can outlive the plan and retains its CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if an uncached engine configuration cannot be extracted, its retained CUDA context cannot be bound, or its native engine index is invalid.
    pub fn engine_config(&self) -> Result<Arc<EngineConfig>> {
        self.retained_engine_config().map(Arc::clone)
    }

    fn retained_engine_config(&self) -> Result<&Arc<EngineConfig>> {
        if let Some(config) = self.inner.engine_config.get() {
            return Ok(config);
        }
        bind_context(self.cuda_context())?;
        let config = Arc::new(read_engine_config(
            &self.inner.descriptor,
            self.inner.cuda_context.as_ref().map(Arc::clone),
        )?);
        Ok(self.inner.engine_config.get_or_init(|| config))
    }

    /// Borrows the canonical native engine/knob tag shared by this plan's clones.
    /// Uses the same cached string as the retained engine configuration without copying it.
    /// Failed native queries remain retryable, and no native query runs under the cache initializer.
    ///
    /// # Errors
    ///
    /// Returns an error if engine-configuration extraction, context binding, or native engine or knob-choice querying or decoding fails.
    pub fn tag(&self) -> Result<&str> {
        self.retained_engine_config()?.tag()
    }

    /// Returns the native kernel-cache JSON, or `None` when this plan has no cache.
    ///
    /// # Errors
    ///
    /// Returns an error if context binding, native cache-descriptor extraction, or cache-payload querying fails. An absent cache attribute or the native no-cache status returns `None`.
    pub fn kernel_cache_json(&self) -> Result<Option<KernelCacheJson>> {
        bind_context(self.cuda_context())?;
        if let Some(cache) = &self.inner.restored_kernel_cache {
            return cache.json_representation().map(Some);
        }
        // SAFETY: the named attribute requires the explicitly selected native class.
        let descriptor = match unsafe {
            self.inner.descriptor.attribute_descriptor(
                BackendAttributeName::ExecutionPlanKernelCache,
                BackendDescriptorType::KernelCache,
            )
        } {
            Ok(descriptor) => descriptor,
            // cuDNN reports `BAD_PARAM` for plans built without a kernel cache.
            Err(
                Error::DescriptorAttributeNotFound(_)
                | Error::Cudnn {
                    code: Status::BadParam,
                    ..
                },
            ) => return Ok(None),
            Err(error) => return Err(error),
        };
        Ok(Some(KernelCacheJson::new(
            descriptor
                .attribute_char_string(BackendAttributeName::KernelCacheJsonRepresentation)?,
        )))
    }

    /// Enqueues this plan with the supplied variant pack.
    ///
    /// # Safety
    /// All native accesses require live plan code and valid, initialized inputs or writable outputs,
    /// with compatible layouts, permitted aliases and established producer dependencies.
    /// Keep device storage and code alive through completion and every captured replay.
    /// Host argument storage must remain valid while cuDNN consumes it.
    /// Retain previous device bindings and workspace through executable snapshots
    /// and pending launches that still reference them.
    ///
    /// # Errors
    ///
    /// Returns an error if contexts disagree, workspace is insufficient, context retention or capture-resource preparation fails, workspace initialization fails, or CUDA binding or native execution fails.
    pub unsafe fn execute(&self, context: &Context, variant_pack: &VariantPack) -> Result<()> {
        let prepared = self.prepare_execution(context, variant_pack)?;
        prepared.initialize_workspace()?;
        // SAFETY: the caller upholds this function's safety contract.
        // SAFETY: the caller guarantees valid bindings, ordered accesses and retained storage through completion or replay.
        unsafe { prepared.execute() }
    }

    /// Resolves workspace and native resource retention before enqueue or timing events.
    pub(crate) fn prepare_execution<'a>(
        &'a self,
        context: &'a Context,
        variant_pack: &'a VariantPack,
    ) -> Result<PreparedExecution<'a>> {
        self.validate_workspace(context, variant_pack)?;
        self.retain_execution_context(context.cuda_context())?;
        if self.needs_cuda_graph_retention() || has_deferred_graph_resources() {
            self.retain_on_capturing_stream(&context.stream()?)?;
        }
        Ok(PreparedExecution {
            plan: self,
            context,
            variant_pack,
        })
    }

    /// Populates a CUDA graph using the supplied bindings and workspace.
    ///
    /// # Safety
    /// All native accesses require live plan code and valid, initialized inputs or writable outputs,
    /// with compatible layouts, permitted aliases and established producer dependencies.
    /// Keep device storage and code alive through completion and every captured replay.
    /// Host argument storage must remain valid while cuDNN consumes it.
    /// Retain previous device bindings and workspace through executable snapshots
    /// and pending launches that still reference them.
    ///
    /// # Errors
    ///
    /// Returns an error if graph, plan, context, or variant-pack targets disagree, workspace is insufficient, resource retention or workspace-initializer preparation fails, or CUDA graph operations or the native graph population fails.
    pub unsafe fn populate_graph(
        &self,
        context: &Context,
        variant_pack: &VariantPack,
        graph: &mut Graph,
    ) -> Result<()> {
        validate_context_pair(Some(context.cuda_context()), graph.context(), "cuda graph")?;
        self.validate_workspace(context, variant_pack)?;
        if has_deferred_graph_resources() {
            collect_graph_resources(&context.stream()?)?;
        }
        if variant_pack.has_workspace_initializations() {
            variant_pack.populate_initialized_graph(context, self, graph)
        } else {
            self.populate_native_graph(context, variant_pack, graph)
        }
    }

    /// Updates a native CUDA graph with replacement bindings and workspace.
    /// Graphs with frontend workspace initializers publish a complete replacement child
    /// only after every initializer and native update succeeds. Preparation errors leave
    /// their previous definition unchanged; existing executables retain their own snapshots.
    ///
    /// # Safety
    /// All native accesses require live plan code and valid, initialized inputs or writable outputs,
    /// with compatible layouts, permitted aliases and established producer dependencies.
    /// Keep device storage and code alive through completion and every captured replay.
    /// Host argument storage must remain valid while cuDNN consumes it.
    /// Retain previous device bindings and workspace through executable snapshots
    /// and pending launches that still reference them.
    ///
    /// # Errors
    ///
    /// Returns an error if graph, plan, context, or variant-pack targets disagree, workspace is insufficient, resource retention or workspace-initializer preparation fails, or CUDA graph operations or the native graph update fails.
    pub unsafe fn update_graph(
        &self,
        context: &Context,
        variant_pack: &VariantPack,
        graph: &Graph,
    ) -> Result<()> {
        validate_context_pair(Some(context.cuda_context()), graph.context(), "cuda graph")?;
        self.validate_workspace(context, variant_pack)?;
        if has_deferred_graph_resources() {
            collect_graph_resources(&context.stream()?)?;
        }
        if variant_pack.has_workspace_initializations() {
            variant_pack.update_initialized_graph(context, self, graph)
        } else {
            self.update_native_graph(context, variant_pack, graph)
        }
    }

    /// Whether captured kernels require this plan's retained runtime-compiled code.
    /// Unknown behavior notes conservatively require retention, matching C++.
    #[must_use]
    pub fn needs_cuda_graph_retention(&self) -> bool {
        self.inner.runtime_compiled
    }

    /// Retains the native plan and its restored dependencies in a graph,
    /// including clones, executable snapshots and pending launches.
    /// Application buffers and workspace retain their separate replay contracts.
    ///
    /// # Errors
    ///
    /// Returns an error if graph and plan contexts disagree, context retention or binding fails, or CUDA graph resource retention fails.
    pub fn retain_on_cuda_graph(&self, graph: &Graph) -> Result<()> {
        validate_context_pair(graph.context(), self.cuda_context(), "execution plan graph")?;
        if self.needs_cuda_graph_retention() {
            if let Some(context) = graph.context() {
                self.retain_execution_context(context)?;
            }
            if let Some(context) = self.cuda_context() {
                context.bind()?;
            }
            self.inner.retain_on_graph(graph)?;
        }
        Ok(())
    }

    /// Retains runtime-compiled code in the stream's current capture.
    /// Ordinary execution collects resources returned by completed CUDA graphs.
    ///
    /// # Errors
    ///
    /// Returns an error if stream and plan contexts disagree, execution-context retention fails, or capture-resource retention or completed-resource collection fails.
    pub fn retain_on_capturing_stream(&self, stream: &StreamBinding) -> Result<()> {
        validate_context_pair(
            Some(stream.context()),
            self.cuda_context(),
            "execution plan stream",
        )?;
        if self.needs_cuda_graph_retention() {
            self.retain_execution_context(stream.context())?;
            self.inner.retain_on_capturing_stream(stream)?;
        } else {
            collect_graph_resources(stream)?;
        }
        Ok(())
    }

    pub(crate) fn populate_native_graph(
        &self,
        context: &Context,
        pack: &VariantPack,
        graph: &Graph,
    ) -> Result<()> {
        self.retain_native_graph_resources(context, graph)?;
        // SAFETY: the caller guarantees valid bindings, ordered accesses and retained storage through completion or replay.
        unsafe {
            self.descriptor()
                .populate_graph(context, pack.descriptor(), graph)
        }
    }

    pub(crate) fn update_native_graph(
        &self,
        context: &Context,
        pack: &VariantPack,
        graph: &Graph,
    ) -> Result<()> {
        self.retain_native_graph_resources(context, graph)?;
        // SAFETY: the caller guarantees valid bindings, ordered accesses and retained storage through completion or replay.
        unsafe {
            self.descriptor()
                .update_graph(context, pack.descriptor(), graph)
        }
    }

    /// Retains code and its live context before changing native graph nodes.
    pub(crate) fn retain_native_graph_resources(
        &self,
        context: &Context,
        graph: &Graph,
    ) -> Result<()> {
        validate_context_pair(Some(context.cuda_context()), graph.context(), "cuda graph")?;
        self.retain_execution_context(context.cuda_context())?;
        self.retain_on_cuda_graph(graph)
    }

    /// Borrows the finalized native execution-plan descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.inner.descriptor
    }

    /// Returns the actual retained CUDA context, or None for deviceless metadata.
    #[must_use]
    pub fn cuda_context(&self) -> Option<&Arc<CudaContext>> {
        self.inner.cuda_context.as_ref()
    }

    pub(crate) fn retain_execution_context(&self, context: &Arc<CudaContext>) -> Result<()> {
        if self.cuda_context().is_some() {
            return Ok(());
        }
        let mut contexts = self.inner.execution_contexts.lock().map_err(|_| {
            Error::Cuda(CudaError::ResourcePoisoned {
                name: "execution plan contexts",
            })
        })?;
        if !contexts
            .iter()
            .any(|existing| existing.as_ref() == context.as_ref())
        {
            contexts.push(Arc::clone(context));
        }
        drop(contexts);
        Ok(())
    }
}

impl Drop for ExecutionPlanOwner {
    fn drop(&mut self) {
        let contexts = self
            .execution_contexts
            .get_mut()
            .unwrap_or_else(|error| error.into_inner());
        bind_for_drop(
            self.cuda_context.as_ref().or_else(|| contexts.first()),
            "cudnn execution plan",
        );
    }
}

fn read_engine_config(
    plan: &BackendDescriptor,
    cuda_context: Option<Arc<CudaContext>>,
) -> Result<EngineConfig> {
    // SAFETY: this attribute requires an engine-configuration descriptor.
    let descriptor = unsafe {
        plan.attribute_descriptor(
            BackendAttributeName::ExecutionPlanEngineConfig,
            BackendDescriptorType::EngineCfg,
        )
    }?;
    Ok(EngineConfig::from_descriptor(descriptor, cuda_context))
}

fn config_requires_graph_retention(config: Option<&EngineConfig>) -> bool {
    let notes = config.and_then(|config| {
        config
            .engine()
            .and_then(|engine| engine.behavior_notes())
            .ok()
    });
    // C++ conservatively retains plans when behavior-note introspection fails.
    notes.is_none_or(|notes| notes.contains(&BackendBehaviorNote::RuntimeCompilation))
}
