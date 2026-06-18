use std::{collections::BTreeMap, mem};

use serde::{Deserialize, Serialize};
use singe_cuda::{
    event::{EventFlags, EventRecordFlags},
    memory::DeviceMemory,
};

use crate::{
    behavior::{BackendBehaviorNote, BackendNumericalNote},
    context::Context,
    engine::EngineIndex,
    error::{Error, Result},
    execution::{
        EngineConfig, EngineKnobInfo, ExecutionPlan, IntermediateInfo, KnobChoiceInfo,
        OperationGraph,
    },
    frontend::{
        bindings::Bindings,
        graph::{Graph, TensorRecord},
        operation::CompileConfig,
        plan::{
            AliasTensorBinding, AutotuneConfig, AutotuneResult, BindingReplacement,
            BuildPlanPolicy, CachedPlanChoice, PlanCandidates, PlanTiming, RequiredTensor,
            bindings::{alias_tensor_bindings, required_tensor_bindings},
        },
    },
    scalar::ScalarValue,
    tensor::TensorId,
};

/// Executable frontend graph with one or more cuDNN execution plans.
///
/// A compiled graph owns the lowered operation graph, selected engine configurations, execution plans, and tensor binding templates needed to execute the graph.
/// It corresponds to the frontend workflow of choosing an engine configuration and creating an execution plan.
#[derive(Debug)]
pub struct CompiledGraph {
    pub(crate) graph: Graph,
    pub(crate) graph_name: Option<String>,
    pub(crate) tensors: BTreeMap<TensorId, TensorRecord>,
    pub(crate) binding_template_ids: Vec<TensorId>,
    pub(crate) scalar_bindings: Vec<(TensorId, ScalarValue)>,
    pub(crate) binding_replacements: Vec<BindingReplacement>,
    pub(crate) operation_graph: OperationGraph,
    pub(crate) engine_configs: Vec<EngineConfig>,
    pub(crate) execution_plans: Vec<ExecutionPlan>,
    pub(crate) default_plan_index: usize,
}

const COMPILED_GRAPH_SCHEMA_VERSION: &str = "v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompiledGraphSnapshot {
    schema: String,
    graph: Graph,
    plan_choices: Vec<CachedPlanChoice>,
    default_plan_index: usize,
}

impl CompiledGraph {
    pub fn cached_plan_choices(&self) -> Result<Vec<CachedPlanChoice>> {
        self.engine_configs
            .iter()
            .map(CachedPlanChoice::from_engine_config)
            .collect()
    }

    pub fn autotune(
        mut self,
        ctx: &Context,
        bindings: &Bindings<'_>,
        workspace: Option<&mut DeviceMemory<u8>>,
        config: &AutotuneConfig,
    ) -> Result<(Self, AutotuneResult)> {
        let result = self.autotune_in_place(ctx, bindings, workspace, config)?;
        Ok((self, result))
    }

    pub fn autotune_rank(
        &self,
        ctx: &Context,
        bindings: &Bindings<'_>,
        workspace: Option<&mut DeviceMemory<u8>>,
        config: &AutotuneConfig,
    ) -> Result<AutotuneResult> {
        self.measure_autotune(ctx, bindings, workspace, config)
    }

    pub fn autotune_in_place(
        &mut self,
        ctx: &Context,
        bindings: &Bindings<'_>,
        workspace: Option<&mut DeviceMemory<u8>>,
        config: &AutotuneConfig,
    ) -> Result<AutotuneResult> {
        let result = self.measure_autotune(ctx, bindings, workspace, config)?;
        self.apply_autotune_ranking(&result.timings);
        Ok(result)
    }

    fn measure_autotune(
        &self,
        ctx: &Context,
        bindings: &Bindings<'_>,
        mut workspace: Option<&mut DeviceMemory<u8>>,
        config: &AutotuneConfig,
    ) -> Result<AutotuneResult> {
        let plan_count = self.execution_plans.len();
        if plan_count == 0 {
            return Err(Error::NoAvailableEngines);
        }

        if config.iterations == 0 {
            return Err(Error::OutOfRange {
                name: "iterations".into(),
            });
        }

        let stream = ctx.stream()?;

        let start = ctx
            .cuda_context()
            .create_event_with_flags(EventFlags::DEFAULT)?;
        let stop = ctx
            .cuda_context()
            .create_event_with_flags(EventFlags::DEFAULT)?;

        let mut timings = Vec::new();
        for plan_index in 0..plan_count {
            let mut warmup_ok = false;
            for _ in 0..config.warmup.max(1) {
                if self
                    .execute_at(ctx, bindings, workspace.as_deref_mut(), plan_index)
                    .is_ok()
                {
                    warmup_ok = true;
                } else {
                    warmup_ok = false;
                    break;
                }
            }
            if !warmup_ok {
                continue;
            }

            ctx.cuda_context().synchronize()?;

            let mut best_ms = f32::INFINITY;
            for _ in 0..config.iterations {
                start.record_on(&stream, EventRecordFlags::DEFAULT)?;
                if self
                    .execute_at(ctx, bindings, workspace.as_deref_mut(), plan_index)
                    .is_err()
                {
                    best_ms = f32::INFINITY;
                    break;
                }
                stop.record_on(&stream, EventRecordFlags::DEFAULT)?;
                stop.synchronize()?;
                best_ms = best_ms.min(stop.elapsed_time_since(&start)?);
            }

            if best_ms.is_finite() {
                timings.push(PlanTiming {
                    original_plan_index: plan_index,
                    ranked_plan_index: usize::MAX,
                    elapsed_ms: best_ms,
                });
            }
        }

        if timings.is_empty() {
            return Err(Error::NoAvailableEngines);
        }

        timings.sort_by(|lhs, rhs| lhs.elapsed_ms.total_cmp(&rhs.elapsed_ms));
        for (new_index, timing) in timings.iter_mut().enumerate() {
            timing.ranked_plan_index = new_index;
        }

        Ok(AutotuneResult {
            winner_original_plan_index: timings[0].original_plan_index,
            winner_ranked_plan_index: timings[0].ranked_plan_index,
            timings,
        })
    }

    fn apply_autotune_ranking(&mut self, timings: &[PlanTiming]) {
        let mut ranked_plan_indices = timings
            .iter()
            .map(|timing| timing.original_plan_index)
            .collect::<Vec<_>>();
        let mut ranked_configs = Vec::with_capacity(ranked_plan_indices.len());
        let mut ranked_plans = Vec::with_capacity(ranked_plan_indices.len());
        let mut remaining_configs = mem::take(&mut self.engine_configs);
        let mut remaining_plans = mem::take(&mut self.execution_plans);
        for original_index in ranked_plan_indices.drain(..).rev() {
            ranked_configs.push(remaining_configs.swap_remove(original_index));
            ranked_plans.push(remaining_plans.swap_remove(original_index));
        }
        ranked_configs.reverse();
        ranked_plans.reverse();
        self.engine_configs = ranked_configs;
        self.execution_plans = ranked_plans;
        self.default_plan_index = 0;
    }

    pub fn workspace_size(&self) -> Result<usize> {
        self.execution_plans[self.default_plan_index].workspace_size()
    }

    pub fn max_workspace_size(&self) -> Result<usize> {
        let mut max_workspace_size = 0;
        for plan_index in 0..self.execution_plans.len() {
            max_workspace_size = max_workspace_size.max(self.workspace_size_at(plan_index)?);
        }
        Ok(max_workspace_size)
    }

    pub fn required_tensor_ids(&self) -> Vec<TensorId> {
        self.required_tensor_bindings()
            .into_iter()
            .map(|binding| binding.id)
            .collect()
    }

    pub fn required_tensors(&self) -> Vec<(TensorId, TensorId)> {
        self.required_tensor_bindings()
            .into_iter()
            .map(|binding| (binding.id, binding.backend_uid.as_tensor_id()))
            .collect()
    }

    pub fn required_tensor_bindings(&self) -> Vec<RequiredTensor> {
        required_tensor_bindings(&self.tensors, &self.binding_replacements)
    }

    pub fn alias_tensor_bindings(&self) -> Vec<AliasTensorBinding> {
        alias_tensor_bindings(&self.tensors, &self.binding_replacements)
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    pub fn operation_graph(&self) -> &OperationGraph {
        &self.operation_graph
    }

    pub fn engine_config(&self) -> &EngineConfig {
        &self.engine_configs[self.default_plan_index]
    }

    pub fn engine_configs(&self) -> &[EngineConfig] {
        &self.engine_configs
    }

    pub fn execution_plan(&self) -> &ExecutionPlan {
        &self.execution_plans[self.default_plan_index]
    }

    pub fn execution_plans(&self) -> &[ExecutionPlan] {
        &self.execution_plans
    }

    pub fn default_plan_index(&self) -> usize {
        self.default_plan_index
    }

    pub fn set_default_plan(&mut self, plan_index: usize) -> Result<()> {
        if plan_index != 0 {
            self.execution_plan_at(plan_index)?;
        }
        self.execution_plan_at(plan_index)?;
        self.default_plan_index = plan_index;
        Ok(())
    }

    pub fn with_default_plan(mut self, plan_index: usize) -> Result<Self> {
        self.set_default_plan(plan_index)?;
        Ok(self)
    }

    pub fn engine_config_at(&self, plan_index: usize) -> Result<&EngineConfig> {
        self.engine_configs
            .get(plan_index)
            .ok_or(Error::OutOfRange {
                name: "plan_index".into(),
            })
    }

    pub fn name(&self) -> Result<String> {
        self.name_at(self.default_plan_index())
    }

    pub fn engine_index(&self) -> Result<EngineIndex> {
        self.engine_index_at(self.default_plan_index())
    }

    pub fn engine_numerical_notes(&self) -> Result<Vec<BackendNumericalNote>> {
        self.engine_numerical_notes_at(self.default_plan_index())
    }

    pub fn engine_behavior_notes(&self) -> Result<Vec<BackendBehaviorNote>> {
        self.engine_behavior_notes_at(self.default_plan_index())
    }

    pub fn engine_workspace_size(&self) -> Result<usize> {
        self.engine_workspace_size_at(self.default_plan_index())
    }

    pub fn engine_knob_infos(&self) -> Result<Vec<EngineKnobInfo>> {
        self.engine_knob_infos_at(self.default_plan_index())
    }

    pub fn engine_knob_choices(&self) -> Result<Vec<KnobChoiceInfo>> {
        self.engine_knob_choices_at(self.default_plan_index())
    }

    pub fn engine_intermediate_infos(&self) -> Result<Vec<IntermediateInfo>> {
        self.engine_intermediate_infos_at(self.default_plan_index())
    }

    pub fn name_at(&self, plan_index: usize) -> Result<String> {
        PlanCandidates::format_name(
            self.graph_name.as_deref(),
            self.engine_config_at(plan_index)?,
        )
    }

    pub fn names(&self) -> Result<Vec<String>> {
        (0..self.engine_configs.len())
            .map(|plan_index| self.name_at(plan_index))
            .collect()
    }

    pub fn engine_index_at(&self, plan_index: usize) -> Result<EngineIndex> {
        Ok(self.engine_config_at(plan_index)?.engine()?.index())
    }

    pub fn engine_numerical_notes_at(
        &self,
        plan_index: usize,
    ) -> Result<Vec<BackendNumericalNote>> {
        self.engine_config_at(plan_index)?
            .engine()?
            .numerical_notes(self.operation_graph().descriptor())
    }

    pub fn engine_behavior_notes_at(&self, plan_index: usize) -> Result<Vec<BackendBehaviorNote>> {
        self.engine_config_at(plan_index)?
            .engine()?
            .behavior_notes(self.operation_graph().descriptor())
    }

    pub fn engine_workspace_size_at(&self, plan_index: usize) -> Result<usize> {
        self.engine_config_at(plan_index)?.workspace_size()
    }

    pub fn engine_knob_infos_at(&self, plan_index: usize) -> Result<Vec<EngineKnobInfo>> {
        self.engine_config_at(plan_index)?.knob_infos()
    }

    pub fn engine_knob_choices_at(&self, plan_index: usize) -> Result<Vec<KnobChoiceInfo>> {
        self.engine_config_at(plan_index)?.knob_choices()
    }

    pub fn engine_intermediate_infos_at(&self, plan_index: usize) -> Result<Vec<IntermediateInfo>> {
        self.engine_config_at(plan_index)?.intermediate_infos()
    }

    pub fn workspace_size_at(&self, plan_index: usize) -> Result<usize> {
        self.execution_plan_at(plan_index)?.workspace_size()
    }

    pub fn computed_intermediate_ids(&self) -> Result<Vec<i64>> {
        self.computed_intermediate_ids_at(self.default_plan_index())
    }

    pub fn computed_intermediate_ids_at(&self, plan_index: usize) -> Result<Vec<i64>> {
        self.execution_plan_at(plan_index)?
            .computed_intermediate_ids()
    }

    pub fn run_only_intermediate_ids(&self) -> Result<Vec<i64>> {
        self.run_only_intermediate_ids_at(self.default_plan_index())
    }

    pub fn run_only_intermediate_ids_at(&self, plan_index: usize) -> Result<Vec<i64>> {
        self.execution_plan_at(plan_index)?
            .run_only_intermediate_ids()
    }

    pub fn execution_plan_at(&self, plan_index: usize) -> Result<&ExecutionPlan> {
        self.execution_plans
            .get(plan_index)
            .ok_or(Error::OutOfRange {
                name: "plan_index".into(),
            })
    }

    // pub fn plan_json(&self) -> Result<String> {
    //     self.plan_json_at(self.default_plan_index())
    // }

    // pub fn plan_json_at(&self, plan_index: usize) -> Result<String> {
    //     self.execution_plan_at(plan_index)?.json_representation()
    // }

    // pub fn kernel_cache_json(&self) -> Result<Option<String>> {
    //     self.kernel_cache_json_at(self.default_plan_index())
    // }

    // pub fn kernel_cache_json_at(&self, plan_index: usize) -> Result<Option<String>> {
    //     self.execution_plan_at(plan_index)?.kernel_cache_json()
    // }

    pub fn serialize(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(&self.get_snapshot()?).map_err(Error::from)
    }

    pub fn deserialize(ctx: &Context, data: &[u8]) -> Result<Self> {
        let snapshot: CompiledGraphSnapshot = serde_json::from_slice(data)?;
        Self::create_from_snapshot(ctx, snapshot)
    }

    fn create_from_snapshot(ctx: &Context, snapshot: CompiledGraphSnapshot) -> Result<Self> {
        if snapshot.schema != COMPILED_GRAPH_SCHEMA_VERSION {
            return Err(Error::FrontendGraphSchemaVersionMismatch {
                expected: COMPILED_GRAPH_SCHEMA_VERSION.into(),
                actual: snapshot.schema,
            });
        }
        if snapshot.plan_choices.is_empty() {
            return Err(Error::FrontendGraphMissingPlanChoices);
        }

        let compile_config = CompileConfig::new().with_build_policy(BuildPlanPolicy::AllSupported);
        let mut compiled = snapshot.graph.compile_with_cached_plan_choices(
            ctx,
            &snapshot.plan_choices,
            &compile_config,
        )?;
        compiled.set_default_plan(snapshot.default_plan_index)?;
        Ok(compiled)
    }

    fn get_snapshot(&self) -> Result<CompiledGraphSnapshot> {
        Ok(CompiledGraphSnapshot {
            schema: COMPILED_GRAPH_SCHEMA_VERSION.to_owned(),
            graph: self.graph.clone(),
            plan_choices: self.cached_plan_choices()?,
            default_plan_index: self.default_plan_index,
        })
    }

    pub(crate) fn binding_replacements(&self) -> &[BindingReplacement] {
        &self.binding_replacements
    }

    pub(crate) fn cuda_graph_binding_ids(&self) -> Vec<TensorId> {
        self.binding_template_ids.clone()
    }

    #[cfg(all(test, feature = "testing"))]
    pub(crate) fn tensor_record(&self, tensor: TensorId) -> Result<&TensorRecord> {
        self.tensors
            .get(&tensor)
            .ok_or(Error::FrontendTensorNotFound(tensor))
    }
}
