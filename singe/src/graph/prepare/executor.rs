#[cfg(feature = "cuda")]
use std::fmt::Write as _;
use std::{
    collections::HashMap,
    mem,
    time::{Duration, Instant},
};

#[cfg(feature = "cuda")]
use crate::{
    backend::cuda::{
        self, CudaCompiledGraph, CudaStream, compiled::executable::GraphCompileProviderDecision,
    },
    error::Error,
    graph::{GraphExecutionMode, run::write_targets},
};
use crate::{
    device::Device,
    error::Result,
    graph::{
        CompiledGraph, Graph, GraphExecutionCacheStats, GraphFingerprintOptions, GraphProgram,
        GraphRunStats, GraphSpecializationKey, NumericalScope,
        generators::with_generator_states,
        prepare::{
            GraphExecutionPlan, GraphExecutor, GraphPrepareOptions, PreparedBackend, PreparedPlan,
            SharedBuffers,
            cache::{CachedPlan, RuntimeMetadataKey},
        },
    },
    state::StateSnapshot,
    tensor::Tensor,
    utility::emit,
};

impl GraphExecutor {
    pub(crate) fn create(
        compiled: CompiledGraph,
        options: GraphPrepareOptions,
        program: GraphProgram,
        state: StateSnapshot,
        template: Graph,
    ) -> Result<Self> {
        let started = Instant::now();
        #[cfg(feature = "cuda")]
        let options = {
            let mut options = options;
            options.cuda.arena = Some(options.cuda.shared_arena.clone().unwrap_or_default());
            options
        };
        let active_key = RuntimeMetadataKey::from_signature(compiled.signature());
        let fingerprint = program.fingerprint()?;
        let active = PreparedPlan::create(compiled, &options, fingerprint, &state)?;
        Ok(Self {
            program,
            state,
            template,
            options,
            active,
            active_key,
            inactive: Vec::new(),
            cache_stats: GraphExecutionCacheStats {
                misses: 1,
                analysis_builds: 1,
                ..GraphExecutionCacheStats::default()
            },
            cache_clock: 0,
            preparation_time: started.elapsed(),
            runs: 0,
            last_run: None,
        })
    }

    /// Creates an independent instance sharing the prepared immutable analysis.
    /// Its native plans, workspaces, and replay allocations are independent.
    ///
    /// # Errors
    ///
    /// Returns an error if the execution mode or operation support is unavailable, the numerical policy cannot be satisfied, fingerprinting fails, or creating CUDA streams or compiled replay resources fails.
    pub fn create_instance(&self) -> Result<Self> {
        self.execution_plan().create_instance()
    }

    #[cfg(feature = "cuda")]
    pub fn create_instance_with_inputs_in_place(
        &self,
        inputs: impl IntoIterator<Item = (String, Tensor)>,
    ) -> Result<Self> {
        self.execution_plan()
            .create_instance_with_inputs_in_place(inputs)
    }

    // A new instance whose buffers are `buffers`, which the instances sharing them must run one at a time.
    #[cfg(feature = "cuda")]
    pub(crate) fn create_instance_sharing(
        &self,
        inputs: impl IntoIterator<Item = (String, Tensor)>,
        buffers: &SharedBuffers,
    ) -> Result<Self> {
        self.execution_plan()
            .create_instance_sharing(inputs, buffers)
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn create_instance_sharing(
        &self,
        inputs: impl IntoIterator<Item = (String, Tensor)>,
        _buffers: &SharedBuffers,
    ) -> Result<Self> {
        self.create_instance_with_inputs_in_place(inputs)
    }

    // Without CUDA there is nothing to bind, since every backend reads its inputs where they are.
    #[cfg(not(feature = "cuda"))]
    pub fn create_instance_with_inputs_in_place(
        &self,
        _inputs: impl IntoIterator<Item = (String, Tensor)>,
    ) -> Result<Self> {
        self.create_instance()
    }

    /// Shares the active specialization and state without native graph handles.
    /// Move this plan to another worker and create an instance there.
    #[must_use]
    pub fn execution_plan(&self) -> GraphExecutionPlan {
        GraphExecutionPlan {
            compiled: self.active.compiled.clone(),
            active_key: self.active_key.clone(),
            options: self.options.clone(),
            program: self.program.clone(),
            state: self.state.clone(),
            template: self.template.clone(),
        }
    }

    /// Returns the immutable logical program shared by this executor.
    #[must_use]
    pub fn program(&self) -> &GraphProgram {
        &self.program
    }

    /// Returns this instance's immutable bound state publication.
    #[must_use]
    pub fn state(&self) -> &StateSnapshot {
        &self.state
    }

    /// Returns the active plan's program, state, numerical, placement, and backend specialization identity.
    #[must_use]
    pub fn specialization_key(&self) -> &GraphSpecializationKey {
        &self.active.specialization
    }

    /// Replaces this instance only after the complete new preparation succeeds.
    /// Other instances retain their previous state and replay storage.
    ///
    /// A failure leaves this executor unchanged.
    /// Successful replacement resets its run and cache bookkeeping to the newly prepared instance.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration has zero cache capacity, state or runtime shape requirements are invalid, graph placement or compilation fails, a required backend or numerical policy is unsupported, or preparing native resources fails.
    pub fn rebind_state(&mut self, state: &StateSnapshot) -> Result<()> {
        let replacement = self.program.prepare(state, self.options.clone())?;
        *self = replacement;
        Ok(())
    }

    /// Returns completed tensors whose values cannot be overwritten by a later run.
    ///
    /// Selects a cached specialization for the runtime input metadata or prepares a new one.
    /// CUDA eager execution waits for submitted work even when an operation returns an error.
    ///
    /// # Errors
    ///
    /// Returns an error if runtime inputs fail signature guards, specializing or preparing an uncached plan fails, backend execution or native graph replay fails, or synchronizing work or copying output snapshots fails.
    pub fn run(&mut self, inputs: &HashMap<String, Tensor>) -> Result<Vec<Tensor>> {
        let _numerics = NumericalScope::enter(self.options.numerics);
        let inputs = with_generator_states(&self.program.logical_graph().generators, inputs)?;
        let inputs = inputs.as_ref();
        self.select_plan(inputs)?;
        let outputs = match &mut self.active.backend {
            PreparedBackend::Cpu => {
                let (outputs, stats) = self.active.compiled.run_with_stats(inputs)?;
                self.last_run = Some(stats);
                outputs
            }
            #[cfg(feature = "cuda")]
            PreparedBackend::CudaEager { stream } => {
                let _guard = cuda::set_current_stream(stream)?;
                let result = self.active.compiled.run_with_stats(inputs);
                // Complete all submissions even when a later operation failed.
                stream.synchronize()?;
                let (outputs, stats) = result?;
                self.last_run = Some(stats);
                outputs
            }
            #[cfg(feature = "cuda")]
            PreparedBackend::Cuda { graph, stream } => {
                run_cuda(&self.active.compiled, graph, inputs, stream, |_| Ok(()))?.0
            }
        };
        self.runs = self.runs.saturating_add(1);
        Ok(outputs)
    }

    /// Runs native replay on an explicit stream and returns completed snapshots.
    /// The stream must belong to this instance's actual CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if the executor is not in native CUDA replay mode, the stream belongs to a different device or CUDA context, or input validation, specialization, replay, synchronization, or output copying fails.
    #[cfg(feature = "cuda")]
    pub fn run_on(
        &mut self,
        inputs: &HashMap<String, Tensor>,
        stream: &CudaStream,
    ) -> Result<Vec<Tensor>> {
        self.run_on_then(inputs, stream, |_| Ok(()))
            .map(|(outputs, _)| outputs)
    }

    // Runs native replay as `run_on` does, calling `enqueued` once the replay's work is on the stream and before waiting for any of it.
    // Also returns how long writing values back into inputs took afterwards, which that work does not include.
    //
    // # Errors
    //
    // Returns what `run_on` does, and what `enqueued` returns.
    #[cfg(feature = "cuda")]
    pub fn run_on_then(
        &mut self,
        inputs: &HashMap<String, Tensor>,
        stream: &CudaStream,
        enqueued: impl FnOnce(&CudaStream) -> Result<()>,
    ) -> Result<(Vec<Tensor>, Duration)> {
        let _numerics = NumericalScope::enter(self.options.numerics);
        if !self.device().is_cuda() {
            return Err(Error::unsupported_operation(
                "cuda stream for cpu graph executor",
            ));
        }

        if stream.device() != self.device() {
            return Err(Error::device_mismatch(
                "graph replay stream",
                self.device(),
                stream.device(),
            ));
        }

        if self.active.specialization.native_context_identity()
            != Some(stream.raw().context().as_raw() as usize)
        {
            return Err(Error::invalid_argument("graph replay stream context"));
        }

        let inputs = with_generator_states(&self.program.logical_graph().generators, inputs)?;
        let inputs = inputs.as_ref();
        self.select_plan(inputs)?;
        let PreparedBackend::Cuda { graph, .. } = &mut self.active.backend else {
            return Err(Error::unsupported_operation(
                "cuda stream for cpu graph executor",
            ));
        };
        let outputs = run_cuda(&self.active.compiled, graph, inputs, stream, enqueued)?;
        self.runs = self.runs.saturating_add(1);
        Ok(outputs)
    }

    fn select_plan(&mut self, inputs: &HashMap<String, Tensor>) -> Result<()> {
        let key = RuntimeMetadataKey::from_inputs(self.active.compiled.signature(), inputs)?;
        self.cache_clock = self.cache_clock.saturating_add(1);
        if key == self.active_key {
            self.cache_stats.hits = self.cache_stats.hits.saturating_add(1);
            return Ok(());
        }

        let replacement = if let Some(index) =
            self.inactive.iter().position(|entry| entry.key == key)
        {
            self.cache_stats.hits = self.cache_stats.hits.saturating_add(1);
            self.inactive.remove(index).plan
        } else {
            #[cfg(feature = "cuda")]
            if self.device().is_cuda() && self.options.mode == GraphExecutionMode::Compiled {
                self.options
                    .cuda
                    .validate_next_recording(self.cache_stats.misses)?;
            }

            let specialized = if self.program.logical_graph().shape_program.is_empty() {
                self.template.specialize_runtime_input_metadata(inputs)?
            } else {
                let resolved = self.program.resolve_runtime_shapes(inputs)?;
                self.program
                    .bind_graph_state(resolved, &self.state)?
                    .retain_executable_nodes()?
                    .placed_on(self.device())?
                    .specialize_runtime_input_metadata(inputs)?
            };
            let compiled =
                specialized.compile_with_freeze(self.options.compile, self.options.freeze)?;
            let plan = PreparedPlan::create(
                compiled,
                &self.options,
                self.active.specialization.program(),
                &self.state,
            )?;
            self.cache_stats.misses = self.cache_stats.misses.saturating_add(1);
            self.cache_stats.analysis_builds = self.cache_stats.analysis_builds.saturating_add(1);
            plan
        };
        let old = mem::replace(&mut self.active, replacement);
        let old_key = mem::replace(&mut self.active_key, key);
        self.inactive.push(CachedPlan {
            key: old_key,
            plan: old,
            last_used: self.cache_clock.saturating_sub(1),
        });
        while self.inactive.len() + 1 > self.options.cache_capacity {
            if let Some(index) = self
                .inactive
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(index, _)| index)
            {
                self.inactive.remove(index);
                self.cache_stats.evictions = self.cache_stats.evictions.saturating_add(1);
            }
        }
        Ok(())
    }

    /// Returns specialization cache counters with the current active and inactive entry count.
    #[must_use]
    pub fn cache_stats(&self) -> GraphExecutionCacheStats {
        GraphExecutionCacheStats {
            entries: self.inactive.len() + 1,
            ..self.cache_stats
        }
    }

    /// Returns the compiled analysis for the currently active specialization.
    #[must_use]
    pub fn analysis(&self) -> &CompiledGraph {
        &self.active.compiled
    }

    /// Returns the configured target execution device.
    #[must_use]
    pub fn device(&self) -> Device {
        self.options.device
    }

    /// Returns the saturated count of successfully completed executions on this instance.
    #[must_use]
    pub fn runs(&self) -> u64 {
        self.runs
    }

    /// Returns statistics from the most recent successful CPU or CUDA eager run, or `None` if none were recorded.
    ///
    /// Native CUDA graph replay does not refresh this measurement.
    #[must_use]
    pub fn last_run_stats(&self) -> Option<GraphRunStats> {
        self.last_run
    }

    /// Returns the elapsed preparation time recorded when this instance was created.
    #[must_use]
    pub fn preparation_time(&self) -> Duration {
        self.preparation_time
    }

    /// Summarizes preparation, retained allocations, and provider decisions.
    ///
    /// # Errors
    ///
    /// Returns an error if computing the graph fingerprint or lowering retained logical metadata fails, or retained allocation byte arithmetic overflows.
    pub fn report(&self) -> Result<String> {
        let key = self.active.compiled.fingerprint_with_options(
            &GraphFingerprintOptions::with_compile_options(self.options.compile),
        )?;
        let memory = self.memory_stats()?;
        let mut text = format!(
            "target: {}\ngraph fingerprint: {:016x}\npreparation: {:?}\nanalysis: shared immutable IR\ncompleted runs: {}\nbound allocation bytes: {}\n",
            self.device(),
            key.value(),
            self.preparation_time,
            self.runs,
            memory.bound_allocation_bytes()
        );
        let cache = self.cache_stats();
        emit!(
            text,
            "cache entries: {} / {}\ncache hits: {}\ncache misses: {}\ncache evictions: {}\nanalysis builds: {}\nshared state allocation bytes (included in bound bytes): {}\ninactive workspace bytes: {}\ninactive replay allocation bytes: {}\nretained allocation bytes across cached plans: {}\nexternal outputs: caller-owned, excluded from instance totals\nprovider caches and driver resources: excluded from instance totals",
            cache.entries(),
            self.options.cache_capacity,
            cache.hits(),
            cache.misses(),
            cache.evictions(),
            cache.analysis_builds(),
            memory.shared_state_allocation_bytes(),
            memory.inactive_workspace_bytes(),
            memory.inactive_replay_allocation_bytes(),
            memory.retained_allocation_bytes()?
        );
        emit!(
            text,
            "program fingerprint: {:016x}\nstate publication: {:?}\nfreeze policy: {:?}",
            self.active.specialization.program().value(),
            self.state.revision(),
            self.options.freeze
        );
        emit!(
            text,
            "compute precision: {:?}\ndeterminism: {:?}",
            self.options.compute_precision(),
            self.options.determinism()
        );
        emit!(
            text,
            "specialization key: {:016x}",
            self.active.specialization.digest()
        );
        if !self.program.specialized_state().is_empty() {
            emit!(
                text,
                "value-specialized state bindings: {:?}",
                self.program.specialized_state().keys().collect::<Vec<_>>()
            );
        }

        let shapes = &self.program.logical_graph().shape_program;
        if !shapes.is_empty() {
            emit!(
                text,
                "runtime shape program: {} reshape expressions, {} integer values, {} annotated values\nshape specialization: exact, metadata only",
                shapes.reshapes.len(),
                shapes.values.len(),
                shapes.constraints.len()
            );
        }

        if let Some(capture) = self.program.capture_info() {
            emit!(
                text,
                "capture: eager fixed path\nvalue specialization: {}",
                capture.allows_value_specialization()
            );
        }

        match &self.active.backend {
            PreparedBackend::Cpu => text.push_str(
                "provider: cpu interpreter\nworkspace bytes: 0\nreplay allocation bytes: 0\n",
            ),
            #[cfg(feature = "cuda")]
            PreparedBackend::CudaEager { .. } => text.push_str(
                "provider: cuda eager dispatcher\nworkspace bytes: 0\nreplay allocation bytes: 0\n",
            ),
            #[cfg(feature = "cuda")]
            PreparedBackend::Cuda { graph, .. } => {
                let allocator = cuda::allocator_stats(self.device())?;
                if let Some(records) = &self.options.cuda.tuning_records {
                    let coverage = records.coverage();
                    emit!(
                        text,
                        "tuning records: {} entries, {} reads used",
                        records.len(),
                        coverage.used
                    );
                }

                emit!(
                    text,
                    "workspace bytes: {}\nreplay allocation bytes: {}\nallocator cached free bytes (context-wide): {}\nallocator active bytes (context-wide): {}\ncuda shape policy: {:?}\ncuda provider policy: {:?}",
                    graph.library_workspace_bytes(),
                    graph.graph_allocation_bytes(),
                    allocator.cached_bytes(),
                    allocator.active_bytes(),
                    self.options.cuda.shapes,
                    self.options.cuda.lowering
                );
                for operation in graph.report().operations() {
                    emit!(
                        text,
                        "operation {} {}: provider {:?}, materialization {:?}",
                        operation.operation_id().index(),
                        operation.operation(),
                        operation
                            .realized_decision()
                            .and_then(GraphCompileProviderDecision::provider),
                        operation.output_materialization()
                    );
                    for failure in operation.provider_failures() {
                        emit!(text, "  provider failure: {failure:?}");
                    }
                }

                write!(text, "fusion groups:\n{}", graph.report().fusion_plan())
                    .expect("formatting into a string cannot fail");
            }
        }
        Ok(text)
    }

    // Identifies what the active plan runs: its providers, its fusion groups, and the tuning choices it read, which decide its kernels.
    #[cfg(feature = "cuda")]
    pub(crate) fn plan_identity(&self) -> Result<String> {
        let PreparedBackend::Cuda { graph, .. } = &self.active.backend else {
            return Err(Error::unsupported_operation("plan identity off cuda"));
        };
        let mut text = String::new();
        for operation in graph.report().operations() {
            emit!(
                text,
                "{} {:?} {:?}",
                operation.operation_id().index(),
                operation
                    .realized_decision()
                    .and_then(GraphCompileProviderDecision::provider),
                operation.output_materialization()
            );
        }

        write!(text, "{}", graph.report().fusion_plan())
            .expect("formatting into a string cannot fail");
        if let Some(records) = &self.options.cuda.tuning_records {
            emit!(text, "{:?}", records.used_choices());
        }
        Ok(text)
    }

    /// Returns a readable graph dump without backend compiler imports.
    ///
    /// # Errors
    ///
    /// Returns an error if generating the execution report fails.
    pub fn dump(&self) -> Result<String> {
        let mut text = self.report()?;
        for input in self.active.compiled.signature().inputs() {
            emit!(
                text,
                "input {}: {:?} {:?}",
                input.name(),
                input.data_type(),
                input.layout().dimensions()
            );
        }

        for operation in self.active.compiled.ir().operations() {
            emit!(
                text,
                "op {} {:?}: {:?} -> {:?}",
                operation.id().index(),
                operation.opcode(),
                operation.inputs(),
                operation.result_values().collect::<Vec<_>>()
            );
        }

        for (index, output) in self
            .active
            .compiled
            .signature()
            .outputs()
            .iter()
            .enumerate()
        {
            emit!(
                text,
                "output {} {:?}: {:?}",
                index,
                output.name(),
                output.spec()
            );
        }
        Ok(text)
    }
}

#[cfg(feature = "cuda")]
fn run_cuda(
    compiled: &CompiledGraph,
    graph: &mut CudaCompiledGraph,
    inputs: &HashMap<String, Tensor>,
    stream: &CudaStream,
    enqueued: impl FnOnce(&CudaStream) -> Result<()>,
) -> Result<(Vec<Tensor>, Duration)> {
    let resolved = compiled.signature().resolve(inputs)?;
    let targets = write_targets(compiled.ir().input_writes(), &resolved)?;
    let live = graph
        .ir()
        .input_values()
        .filter_map(|(_, name)| inputs.get_key_value(name))
        .map(|(name, tensor)| (name.clone(), tensor.clone()))
        .collect();
    // Trailing outputs hold the values written into inputs; the graph copies them into inputs it reads in place, and device copies write the others back.
    let mut outputs = graph.run_on_then(&live, stream, enqueued)?;
    let copied_back = compiled
        .ir()
        .input_writes()
        .iter()
        .filter(|name| !graph.writes_input_in_graph(name))
        .count();
    let visible = outputs
        .len()
        .checked_sub(copied_back)
        .ok_or_else(|| Error::missing_output("values written into graph inputs"))?;
    let mut written = outputs.split_off(visible).into_iter();
    let started = Instant::now();
    for (name, target) in compiled.ir().input_writes().iter().zip(&targets) {
        if !graph.writes_input_in_graph(name) {
            let value = written
                .next()
                .ok_or_else(|| Error::missing_output("value written into a graph input"))?;
            target.assign(&value)?;
        }
    }
    Ok((outputs, started.elapsed()))
}
