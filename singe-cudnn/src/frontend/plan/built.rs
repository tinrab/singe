use std::collections::BTreeMap;

use singe_cuda::memory::DeviceMemory;

use crate::{
    behavior::{BackendBehaviorNote, BackendNumericalNote},
    context::Context,
    error::{Error, Result},
    execution::{
        EngineConfig, EngineKnobInfo, ExecutionPlan, IntermediateInfo, KnobChoiceInfo,
        OperationGraph,
    },
    frontend::{
        bindings::{
            Bindings, RuntimeOverrides,
            variant_pack_with_replacements_and_overrides_for_ids_and_tensors,
        },
        graph::{Graph, TensorRecord},
        plan::{
            AliasTensorBinding, BindingReplacement, CompiledGraph, RequiredTensor,
            bindings::{alias_tensor_bindings, required_tensor_bindings},
            metadata::{EngineConfigFilter, EngineConfigMetadata},
        },
    },
    scalar::ScalarValue,
    tensor::TensorId,
};

/// Candidate plans after attempting support checks and plan construction.
///
/// Entries may contain either a built [`ExecutionPlan`] or the error observed while checking support/building that candidate.
#[derive(Debug)]
pub struct BuiltPlanCandidates {
    pub(crate) graph: Graph,
    pub(crate) graph_name: Option<String>,
    pub(crate) tensors: BTreeMap<TensorId, TensorRecord>,
    pub(crate) binding_template_ids: Vec<TensorId>,
    pub(crate) scalar_bindings: Vec<(TensorId, ScalarValue)>,
    pub(crate) binding_replacements: Vec<BindingReplacement>,
    pub(crate) operation_graph: OperationGraph,
    pub(crate) entries: Vec<BuiltPlanEntry>,
}

#[derive(Debug)]
pub(crate) struct BuiltPlanEntry {
    pub(crate) engine_config: EngineConfig,
    pub(crate) engine_metadata: EngineConfigMetadata,
    pub(crate) execution_plan: Option<ExecutionPlan>,
    pub(crate) build_error: Option<Error>,
    pub(crate) support_error: Option<Error>,
}

impl BuiltPlanCandidates {
    pub fn graph_name(&self) -> Option<&str> {
        self.graph_name.as_deref()
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    pub fn candidate_count(&self) -> usize {
        self.entries.len()
    }

    pub fn names(&self) -> Result<Vec<String>> {
        (0..self.entries.len())
            .map(|plan_index| self.name_at(plan_index))
            .collect()
    }

    pub fn engine_index_at(&self, plan_index: usize) -> Result<i64> {
        Ok(self.engine_metadata_at(plan_index)?.engine_index)
    }

    pub fn engine_numerical_notes_at(
        &self,
        plan_index: usize,
    ) -> Result<Vec<BackendNumericalNote>> {
        Ok(self.engine_metadata_at(plan_index)?.numerical_notes.clone())
    }

    pub fn engine_behavior_notes_at(&self, plan_index: usize) -> Result<Vec<BackendBehaviorNote>> {
        Ok(self.engine_metadata_at(plan_index)?.behavior_notes.clone())
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

    pub fn name_at(&self, plan_index: usize) -> Result<String> {
        let engine_name = &self.engine_metadata_at(plan_index)?.name;
        Ok(match self.graph_name.as_deref() {
            Some(graph_name) => format!("{graph_name}/{engine_name}"),
            None => engine_name.clone(),
        })
    }

    pub fn execution_plan_at(&self, plan_index: usize) -> Result<&ExecutionPlan> {
        self.entries
            .get(plan_index)
            .ok_or(Error::OutOfRange {
                name: "plan_index".into(),
            })?
            .execution_plan
            .as_ref()
            .ok_or(Error::OutOfRange {
                name: "plan_index".into(),
            })
    }

    pub fn engine_config_at(&self, plan_index: usize) -> Result<&EngineConfig> {
        self.entries
            .get(plan_index)
            .map(|entry| &entry.engine_config)
            .ok_or(Error::OutOfRange {
                name: "plan_index".into(),
            })
    }

    fn engine_metadata_at(&self, plan_index: usize) -> Result<&EngineConfigMetadata> {
        self.entries
            .get(plan_index)
            .map(|entry| &entry.engine_metadata)
            .ok_or(Error::OutOfRange {
                name: "plan_index".into(),
            })
    }

    pub fn build_error_at(&self, plan_index: usize) -> Result<Option<&Error>> {
        self.entries
            .get(plan_index)
            .map(|entry| entry.build_error.as_ref())
            .ok_or(Error::OutOfRange {
                name: "plan_index".into(),
            })
    }

    pub fn support_error_at(&self, plan_index: usize) -> Result<Option<&Error>> {
        self.entries
            .get(plan_index)
            .map(|entry| entry.support_error.as_ref())
            .ok_or(Error::OutOfRange {
                name: "plan_index".into(),
            })
    }

    pub fn check_support_at(&self, plan_index: usize) -> Result<()> {
        let entry = self.entries.get(plan_index).ok_or(Error::OutOfRange {
            name: "plan_index".into(),
        })?;
        if let Some(error) = entry.support_error.as_ref() {
            return Err(Error::FrontendCompile(error.to_string()));
        }
        if let Some(error) = entry.build_error.as_ref() {
            return Err(Error::FrontendCompile(error.to_string()));
        }
        if entry.execution_plan.is_none() {
            return Err(Error::NoAvailableEngines);
        }
        Ok(())
    }

    pub fn compiled_plan_index_at(&self, plan_index: usize) -> Result<Option<usize>> {
        if plan_index >= self.entries.len() {
            return Err(Error::OutOfRange {
                name: "plan_index".into(),
            });
        }

        Ok(self.entries[plan_index]
            .execution_plan
            .as_ref()
            .map(|_| plan_index))
    }

    pub fn workspace_size_at(&self, plan_index: usize) -> Result<usize> {
        self.execution_plan_at(plan_index)?.workspace_size()
    }

    pub fn max_workspace_size(&self) -> Result<usize> {
        let mut max_workspace_size = 0;
        for entry in &self.entries {
            if let Some(execution_plan) = entry.execution_plan.as_ref()
                && let Ok(workspace_size) = execution_plan.workspace_size()
            {
                max_workspace_size = max_workspace_size.max(workspace_size);
            }
        }
        Ok(max_workspace_size)
    }

    pub fn execute_at(
        &self,
        ctx: &Context,
        bindings: &Bindings<'_>,
        workspace: Option<&mut DeviceMemory<u8>>,
        plan_index: usize,
    ) -> Result<()> {
        let entry = self.entries.get(plan_index).ok_or(Error::OutOfRange {
            name: "plan_index".into(),
        })?;
        if let Some(error) = entry.support_error.as_ref() {
            return Err(Error::FrontendCompile(error.to_string()));
        }
        if let Some(error) = entry.build_error.as_ref() {
            return Err(Error::FrontendCompile(error.to_string()));
        }
        let execution_plan = entry
            .execution_plan
            .as_ref()
            .ok_or(Error::NoAvailableEngines)?;
        let variant_pack = variant_pack_with_replacements_and_overrides_for_ids_and_tensors(
            &self.binding_replacements,
            bindings,
            workspace,
            &RuntimeOverrides::new(),
            Some(&self.binding_template_ids),
            Some(&self.tensors),
            Some(&self.graph.dynamic_shape_constraints),
        )?;
        execution_plan.execute(ctx, &variant_pack)
    }

    pub fn bindings(&self) -> Bindings<'_> {
        let mut bindings = Bindings::with_graph_template(&self.tensors, &self.binding_template_ids);
        for record in self.tensors.values() {
            if let Some(scalar_value) = record.tensor.scalar_value {
                bindings.set_scalar_id(record.id, scalar_value);
            }
        }
        for &(id, scalar_value) in &self.scalar_bindings {
            bindings.set_scalar_id(id, scalar_value);
        }
        bindings
    }

    pub fn required_tensor_bindings(&self) -> Vec<RequiredTensor> {
        required_tensor_bindings(&self.tensors, &self.binding_replacements)
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

    pub fn alias_tensor_bindings(&self) -> Vec<AliasTensorBinding> {
        alias_tensor_bindings(&self.tensors, &self.binding_replacements)
    }

    impl_engine_config_filter_methods!();

    pub(crate) fn apply_filter(&mut self, filter: EngineConfigFilter<'_>) {
        for entry in &mut self.entries {
            if !filter.accepts(&entry.engine_metadata) {
                entry.support_error = Some(filter.rejection_error(&entry.engine_metadata));
                entry.execution_plan = None;
            }
        }
    }

    pub fn compile(self) -> Result<CompiledGraph> {
        self.into_compiled_graph()
    }

    pub fn compile_at(mut self, plan_index: usize) -> Result<CompiledGraph> {
        let entry = self.entries.swap_remove(plan_index);
        let execution_plan = entry.execution_plan.ok_or_else(|| {
            entry
                .build_error
                .or(entry.support_error)
                .unwrap_or(Error::NoAvailableEngines)
        })?;
        Ok(CompiledGraph {
            graph: self.graph,
            graph_name: self.graph_name,
            tensors: self.tensors,
            binding_template_ids: self.binding_template_ids,
            scalar_bindings: self.scalar_bindings,
            binding_replacements: self.binding_replacements,
            operation_graph: self.operation_graph,
            engine_configs: vec![entry.engine_config],
            execution_plans: vec![execution_plan],
            default_plan_index: 0,
        })
    }

    pub fn into_compiled_graph(self) -> Result<CompiledGraph> {
        let mut engine_configs = Vec::new();
        let mut execution_plans = Vec::new();
        for entry in self.entries {
            if let Some(execution_plan) = entry.execution_plan {
                engine_configs.push(entry.engine_config);
                execution_plans.push(execution_plan);
            }
        }

        if execution_plans.is_empty() {
            return Err(Error::NoAvailableEngines);
        }

        Ok(CompiledGraph {
            graph: self.graph,
            graph_name: self.graph_name,
            tensors: self.tensors,
            binding_template_ids: self.binding_template_ids,
            scalar_bindings: self.scalar_bindings,
            binding_replacements: self.binding_replacements,
            operation_graph: self.operation_graph,
            engine_configs,
            execution_plans,
            default_plan_index: 0,
        })
    }
}
