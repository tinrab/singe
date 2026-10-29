use std::collections::{BTreeMap, BTreeSet};

use crate::{
    backend::tensor::TensorId,
    context::Context,
    error::Result,
    frontend::{
        error::Error as FrontendError,
        graph::{Graph, GraphLayout, subgraphs::ScoreSubgraph},
        logging,
        operation::{AttentionBackwardHeuristicOverride, Operation, ReductionOperation},
        plan::{RequiredTensor, TensorInfo, TensorInfoIter},
    },
    version,
};

/// Target-specific graph declarations validated without creating native descriptors.
/// Owns expanded operations, reconciled score graphs, reducer bodies and binding requirements.
/// Native engine support remains a later planning decision.
#[derive(Debug, Clone)]
pub struct ValidatedGraph {
    pub(crate) recipe: Graph,
    pub(crate) expanded: Graph,
    pub(crate) layout: GraphLayout,
    pub(crate) subgraphs: BTreeMap<usize, ScoreSubgraph<Self>>,
    pub(crate) reducers: BTreeMap<usize, Self>,
}

impl ValidatedGraph {
    /// Original recipe, before target-specific expansion and layout reconciliation.
    #[must_use]
    pub fn recipe(&self) -> &Graph {
        &self.recipe
    }

    /// Expanded declarations and operation geometry used by native preparation.
    #[must_use]
    pub fn graph(&self) -> &Graph {
        &self.expanded
    }

    /// All retained declarations and access metadata in frontend-ID order.
    /// Includes expanded intermediates, score-subgraph tensors and storage-only sources.
    /// This stage has no native descriptors, so every native UID is absent.
    #[must_use]
    pub fn tensors(&self) -> TensorInfoIter<'_> {
        TensorInfoIter::declarations(&self.layout.tensors)
    }

    /// Retained declaration and access metadata before native preparation.
    ///
    /// # Errors
    /// Returns an error when the validated graph does not contain the frontend identity.
    pub fn tensor_info(&self, id: TensorId) -> Result<TensorInfo<'_>> {
        self.layout
            .tensors
            .get(&id)
            .map(|record| TensorInfo::from_record(id, record))
            .ok_or(FrontendError::TensorNotFound(id).into())
    }

    /// Caller storage and assigned native identities, including nested captures and storage views.
    /// Does not query native descriptors or engine support.
    #[must_use]
    pub fn required_tensor_bindings(&self) -> &[RequiredTensor] {
        &self.layout.required_tensors
    }
}

impl Graph {
    /// Validates and expands a recipe without creating native tensor or operation descriptors.
    /// Resolves scalar/index promotion, shared score layouts, reducer bodies, views and bindings.
    /// Supply a live target, or None with an explicit architecture when attention requires one.
    /// The original recipe remains unchanged, and the result can be prepared without repeating validation.
    ///
    /// # Errors
    /// Returns an error for invalid declarations, operation geometry, references, bounds or incompatible shared layouts.
    /// Native engine availability is queried after preparation.
    pub fn validate(&self, context: Option<&Context>) -> Result<ValidatedGraph> {
        logging::diagnostic!(
            "validating graph {:?}: {} operations, {} declared tensors",
            self.name(),
            self.operations.len(),
            self.tensors.len(),
        );
        self.validate_runtime_feature_support_for_version(version()?.raw())?;
        self.validate_tensor_declarations()?;
        self.validate_attention_statistics_bases()?;
        for (id, bounds) in &self.runtime_shape_constraints {
            bounds.validate_for(self.tensor_spec(*id)?)?;
        }
        let mut expanded = self.clone();
        expanded.config.sm_version = expanded.compilation_sm_version(context)?;
        expanded.retain_attention_alignment()?;
        expanded.promote_attention_ragged_offsets()?;
        expanded.expand_attention_requests()?;
        expanded.expand_attention_backward_requests()?;
        expanded.promote_recorded_index_tensors()?;
        let subgraphs = loop {
            expanded.expand_fused_scalar_shapes()?;
            let subgraphs = expanded.validate_sdpa_subgraphs(context)?;
            if !expanded.merge_shared_sdpa_layouts(&subgraphs)? {
                break subgraphs;
            }
        };
        for subgraph in subgraphs.values() {
            AttentionBackwardHeuristicOverride::retain_into(
                &mut expanded.attention_backward_heuristic_overrides,
                subgraph
                    .graph
                    .expanded
                    .attention_backward_heuristic_overrides
                    .clone(),
            );
        }
        let mut reducers = BTreeMap::new();
        for (index, operation) in expanded.operations.iter().enumerate() {
            if let Operation::Reduction(ReductionOperation::Custom { body, .. }) = operation {
                reducers.insert(index, body.validate(context)?);
            }
        }
        let layout = GraphLayout::create(&mut expanded, &subgraphs)?;
        Ok(ValidatedGraph {
            recipe: self.clone(),
            expanded,
            layout,
            subgraphs,
            reducers,
        })
    }

    fn validate_tensor_declarations(&self) -> Result<()> {
        let mut pending = Vec::new();
        for operation in &self.operations {
            let uses = match operation {
                Operation::AttentionForward(request) => request.direct_tensor_uses(),
                Operation::AttentionBackward(request) => request.direct_tensor_uses(),
                Operation::Sdpa(operation) => {
                    operation.validate_graph(self)?;
                    operation.native_tensor_uses()
                }
                operation => operation.tensor_uses(self.config.data_type_policy),
            };
            pending.extend(uses.reads.into_iter().chain(uses.writes));
        }
        let mut visited = BTreeSet::new();
        while let Some(id) = pending.pop() {
            if !visited.insert(id) {
                continue;
            }
            let tensor = self.tensor_spec(id)?;
            tensor.validate()?;
            pending.extend(tensor.ragged_offset);
        }
        Ok(())
    }
}
