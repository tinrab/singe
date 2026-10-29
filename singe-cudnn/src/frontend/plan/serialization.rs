use std::{collections::BTreeMap, sync::Arc};

use serde::{Deserialize, Serialize};

use crate::{
    backend::{
        device::DeviceProperties,
        execution::{ExecutionPlan, ExecutionPlanTarget, KernelCache, KernelCacheJson},
        tensor::TensorId,
    },
    context::Context,
    error::{Error, Result},
    frontend::{
        bindings::BindingTemplate,
        error::Error as FrontendError,
        graph::TensorRecord,
        logging,
        plan::{
            BindingReplacement, CompiledGraph, EngineConfigMetadata, PlanArtifacts,
            PlanBuildReport, PlanBuildStatus, PlanId,
            bindings::required_tensor_bindings,
            compiled::{CompiledGraphCore, CompiledGraphStructure},
            config::RuntimeShapePolicy,
            entry::{CompiledPlanEntry, PlanCandidate},
        },
        require_baseline,
        workspace::WorkspaceLayout,
    },
    scalar::ScalarValue,
    version,
};

const NATIVE_ARTIFACT_SCHEMA: &str = "singe-cudnn-native-plans-v40";

/// Whether a native executable artifact retains its declarative and resolved graphs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum NativePlanStructure {
    /// Include both graph descriptions, matching the C++ serialization default.
    #[default]
    Include,
    /// Save executable plans and execution metadata without structural graph descriptions.
    Omit,
}

/// Built plans included in a native executable artifact.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum NativePlanSet {
    /// Save only the selected plan, matching C++ and avoiding serialization of unused plans.
    #[default]
    Selected,
    /// Save every retained plan and preserve the selected plan's position.
    All,
}

/// Native executable serialization policy.
/// Defaults to the selected plan with both graph descriptions, matching C++.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativePlanSerializationConfig {
    /// Whether the executable artifact retains declared and resolved graph descriptions.
    pub structure: NativePlanStructure,
    /// Whether to save only the selected plan or all retained built plans.
    pub plans: NativePlanSet,
}

/// Target and capture policy for restoring compiled frontend artifacts.
/// Raw native JSON restoration uses [`ExecutionPlanTarget`] instead.
#[derive(Debug)]
pub enum NativePlanRestoreTarget<'a> {
    /// Restore against a live handle and warm the selected plan through discarded capture, matching C++ defaults.
    Context(&'a mut Context),
    /// Restore against a live handle without capture or temporary stream changes.
    ContextWithoutWarmup(&'a Context),
    /// Restore against device properties without a live handle or capture.
    DeviceProperties(&'a DeviceProperties),
}

#[derive(Serialize, Deserialize)]
struct NativePlanArtifact {
    native_plan: String,
    kernel_cache: Option<KernelCacheJson>,
    metadata: EngineConfigMetadata,
}

#[derive(Serialize, Deserialize)]
struct CompiledGraphArtifact {
    schema: String,
    cudnn_version: u64,
    graph_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    structure: Option<CompiledGraphStructure>,
    shape_policy: RuntimeShapePolicy,
    tensors: BTreeMap<TensorId, TensorRecord>,
    binding_template_ids: Vec<TensorId>,
    scalar_bindings: Vec<(TensorId, ScalarValue)>,
    binding_replacements: Vec<BindingReplacement>,
    plans: Vec<NativePlanArtifact>,
    selected_plan: usize,
}

impl CompiledGraph {
    /// Saves the requested built native plans, kernel caches and selected plan's position.
    /// Retains the declared recipe separately from the resolved graph and its actual binding metadata.
    /// The artifact can be restored without lowering operations, running heuristics or choosing another engine.
    /// [`NativePlanStructure::Omit`] excludes both graphs while preserving execution-time policy and metadata.
    /// A plan-only restored artifact can be serialized again with `Omit`.
    ///
    /// # Errors
    /// Returns native serialization errors or a missing-structure error when `Include` is requested for a plan-only artifact.
    pub fn serialize(&self, config: NativePlanSerializationConfig) -> Result<Vec<u8>> {
        let structure = match config.structure {
            NativePlanStructure::Include => Some(
                self.core
                    .structure
                    .as_ref()
                    .ok_or(FrontendError::NativeArtifactMissingStructure)?
                    .clone(),
            ),
            NativePlanStructure::Omit => None,
        };
        let (entries, selected_plan) = match config.plans {
            NativePlanSet::Selected => (vec![self.default_entry()], 0),
            NativePlanSet::All => (
                self.ordered_entries().collect::<Vec<_>>(),
                self.default_plan_index,
            ),
        };
        let plans = entries
            .iter()
            .map(|entry| {
                Ok(NativePlanArtifact {
                    native_plan: entry.execution_plan.json_representation()?,
                    kernel_cache: entry.execution_plan.kernel_cache_json()?,
                    metadata: entry.candidate.metadata.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        serde_json::to_vec(&CompiledGraphArtifact {
            schema: NATIVE_ARTIFACT_SCHEMA.into(),
            cudnn_version: version()?.raw(),
            graph_name: self.core.artifacts.graph_name.clone(),
            structure,
            shape_policy: self.core.shape_policy.clone(),
            tensors: self.core.artifacts.tensors.clone(),
            binding_template_ids: self.core.binding_template.ids().to_vec(),
            scalar_bindings: self.core.artifacts.scalar_bindings.clone(),
            binding_replacements: self.core.artifacts.binding_replacements.clone(),
            plans,
            selected_plan,
        })
        .map_err(Error::from)
    }

    /// Restores saved native plans against a live context or deviceless device properties.
    /// All saved plans must restore successfully; failures never trigger recompilation from the recipe.
    /// Native finalization checks cuDNN serialization and target compatibility and may perform preparation work.
    /// [`NativePlanRestoreTarget::Context`] warms the selected plan without application buffers, matching C++ defaults.
    /// Choose [`NativePlanRestoreTarget::ContextWithoutWarmup`] to skip capture.
    /// Deviceless restoration never captures or requires a live handle.
    /// Native artifacts always require saved plans, equivalent to C++ `enforce_precompiled`.
    /// Plan-only artifacts retain execution and runtime-layout support; [`Self::graph`] and [`Self::recipe`] return `None`.
    ///
    /// # Safety
    /// The native payloads and Rust binding metadata must originate from the same untampered artifact produced by [`Self::serialize`].
    /// Native introspection cannot establish that arbitrary plan JSON accesses exactly the described tensors and storage.
    /// A deviceless artifact must only execute on live devices compatible with the supplied properties.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed or incompatible artifact metadata, missing plans or an invalid selection, unsupported cuDNN runtime version, invalid bindings or workspace geometry, target binding failure, native plan or kernel-cache restoration failure, or requested capture warmup failure.
    pub unsafe fn deserialize(target: NativePlanRestoreTarget<'_>, data: &[u8]) -> Result<Self> {
        let artifact: CompiledGraphArtifact = serde_json::from_slice(data)?;
        if artifact.schema != NATIVE_ARTIFACT_SCHEMA {
            return Err(FrontendError::GraphSchemaVersionMismatch {
                expected: NATIVE_ARTIFACT_SCHEMA.to_owned(),
                actual: artifact.schema,
            }
            .into());
        }
        if artifact.plans.is_empty() {
            return Err(FrontendError::NativeArtifactMissingPlans.into());
        }
        if artifact.selected_plan >= artifact.plans.len() {
            return Err(FrontendError::NativeArtifactSelectionOutOfRange {
                selected: artifact.selected_plan,
                plans: artifact.plans.len(),
            }
            .into());
        }
        let required_tensors = required_tensor_bindings(
            &artifact.tensors,
            &artifact.binding_replacements,
            &artifact.binding_template_ids,
        )?;
        let runtime_version = version()?.raw();
        require_baseline(runtime_version)?;
        let native_target = match &target {
            NativePlanRestoreTarget::Context(context) => ExecutionPlanTarget::Context(context),
            NativePlanRestoreTarget::ContextWithoutWarmup(context) => {
                ExecutionPlanTarget::Context(context)
            }
            NativePlanRestoreTarget::DeviceProperties(properties) => {
                ExecutionPlanTarget::DeviceProperties(properties)
            }
        };
        if let ExecutionPlanTarget::Context(context) = native_target {
            context.bind()?;
        }
        let layout =
            WorkspaceLayout::create(&artifact.tensors, &artifact.binding_template_ids, &[])?;
        let binding_template = BindingTemplate::create(
            artifact.binding_template_ids,
            &artifact.binding_replacements,
        )?;
        let mut entries = Vec::with_capacity(artifact.plans.len());
        for (position, (id, saved)) in PlanId::collection(artifact.plans.len())
            .zip(artifact.plans)
            .enumerate()
        {
            let entry = restore_plan(native_target, id, saved, &layout).map_err(|source| {
                FrontendError::NativePlanRestoreFailed {
                    position,
                    serialized_cudnn_version: artifact.cudnn_version,
                    runtime_cudnn_version: runtime_version,
                    source: Box::new(source),
                }
            })?;
            entries.push(entry);
        }
        let reports = entries
            .iter()
            .map(|entry| PlanBuildReport {
                plan: entry.candidate.id.clone(),
                name: entry.candidate.name(artifact.graph_name.as_deref()),
                choice: entry.candidate.choice(),
                status: PlanBuildStatus::Built,
                exclusion: None,
            })
            .collect();
        let compiled = Self {
            default_plan_index: artifact.selected_plan,
            plan_order: (0..entries.len()).collect(),
            core: Arc::new(CompiledGraphCore {
                structure: artifact.structure,
                shape_policy: artifact.shape_policy,
                entries,
                reports,
                operation_graph: None,
                artifacts: PlanArtifacts {
                    graph_name: artifact.graph_name,
                    tensors: artifact.tensors,
                    scalar_bindings: artifact.scalar_bindings,
                    binding_replacements: artifact.binding_replacements,
                    required_tensors,
                },
                binding_template,
                workspace_layout: layout,
                _lowered: None,
            }),
        };
        if let NativePlanRestoreTarget::Context(context) = target {
            let outcome = compiled.warmup(context)?;
            logging::diagnostic!("restored-plan capture warmup: {outcome:?}");
        }
        Ok(compiled)
    }
}

fn restore_plan(
    target: ExecutionPlanTarget<'_>,
    id: PlanId,
    saved: NativePlanArtifact,
    layout: &WorkspaceLayout,
) -> Result<CompiledPlanEntry> {
    let kernel_cache = saved
        .kernel_cache
        .as_ref()
        .map(|json| -> Result<KernelCache> {
            let mut cache = KernelCache::deserialize(json)?;
            cache.build(None)?;
            Ok(cache)
        })
        .transpose()?;
    let execution_plan = ExecutionPlan::from_json(target, &saved.native_plan, kernel_cache)?;
    let engine_config = execution_plan.engine_config()?;
    let mut actual = engine_config.choice()?.clone();
    actual
        .knob_choices
        .sort_unstable_by_key(|choice| choice.knob_type);
    let mut metadata = saved.metadata;
    metadata
        .knob_choices
        .sort_unstable_by_key(|choice| choice.knob_type);
    let mut candidate = PlanCandidate {
        id,
        engine_config,
        metadata,
    };
    let expected = candidate.choice();
    if actual != expected {
        return Err(FrontendError::NativePlanChoiceMismatch { expected, actual }.into());
    }
    candidate
        .engine_config
        .tag()?
        .clone_into(&mut candidate.metadata.name);
    candidate.metadata.workspace_size = layout.total_byte_len(execution_plan.workspace_size()?)?;
    Ok(CompiledPlanEntry {
        execution_plan,
        candidate,
    })
}
