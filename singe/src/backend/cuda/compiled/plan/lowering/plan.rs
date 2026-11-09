use std::collections::HashMap;

use crate::{
    backend::cuda::{
        compiled::plan::{
            kernel::KernelFeasibility,
            lowering::{
                CudaGraphProvider, LoweringCandidate, LoweringDecision, LoweringDecisionRecord,
                LoweringPolicy, LoweringRejection, ModuleKernel,
            },
        },
        tuning::{TunedChoice, TunedWay, way_key},
    },
    error::{Error, Result},
    graph::ir::{Graph, OperationId, ValueId},
};

/// Cost-ordered CUDA provider candidates and planned decisions by graph node and IR operation.
#[derive(Debug, Clone)]
pub struct LoweringPlan {
    pub(crate) candidates: HashMap<ValueId, Vec<LoweringCandidate>>,
    pub(crate) decisions: HashMap<ValueId, LoweringDecisionRecord>,
    pub(crate) operation_decisions: HashMap<OperationId, LoweringDecisionRecord>,
    // The ways tuning records chose, by node.
    pub(crate) ways: HashMap<ValueId, TunedWay>,
    // The policy this plan was made under.
    pub(crate) policy: LoweringPolicy,
}

impl LoweringPlan {
    /// Returns lowering candidates for a node, or an empty slice if it has none.
    pub fn candidates(&self, id: ValueId) -> &[LoweringCandidate] {
        self.candidates.get(&id).map_or(&[], Vec::as_slice)
    }

    /// Returns whether a listed candidate for this node uses the supplied provider.
    #[must_use]
    pub fn node_supports(&self, id: ValueId, provider: CudaGraphProvider) -> bool {
        self.candidates(id)
            .iter()
            .any(|candidate| candidate.provider() == provider)
    }

    // The way a tuning record chose for this node.
    #[must_use]
    pub fn way(&self, id: ValueId) -> Option<TunedWay> {
        self.ways.get(&id).copied()
    }

    // Whether the policy this plan was made under lets the recognizer choose `module`.
    #[must_use]
    pub fn allows_module(&self, module: ModuleKernel) -> bool {
        self.policy.allows_module(module)
    }

    /// Returns the retained lowering decision for a node, if present.
    #[must_use]
    pub fn decision(&self, id: ValueId) -> Option<&LoweringDecisionRecord> {
        self.decisions.get(&id)
    }

    /// Returns the retained lowering decision for an IR operation, if present.
    #[must_use]
    pub fn decision_for_operation(&self, id: OperationId) -> Option<&LoweringDecisionRecord> {
        self.operation_decisions.get(&id)
    }

    /// Returns whether the node decision selects this provider or lists it among rejected candidates.
    /// Returns false when the node has no decision record.
    #[must_use]
    pub fn should_try_provider(&self, id: ValueId, provider: CudaGraphProvider) -> bool {
        self.decision(id).is_some_and(|record| {
            record.decision().provider() == Some(provider)
                || record
                    .rejected_candidates()
                    .iter()
                    .any(|candidate| candidate.provider() == provider)
        })
    }

    /// Clones nonempty node candidate lists in the supplied execution order for cache identity construction.
    #[must_use]
    pub fn candidate_key_material(
        &self,
        execution_order: &[ValueId],
    ) -> Vec<(ValueId, Vec<LoweringCandidate>)> {
        execution_order
            .iter()
            .filter_map(|id| {
                let candidates = self.candidates(*id);
                (!candidates.is_empty()).then(|| (*id, candidates.to_vec()))
            })
            .collect()
    }
}

impl LoweringPlan {
    /// Builds lowering candidates in execution order and chooses the lowest-cost enabled provider for each operation.
    /// Operations without candidates retain an unsupported decision.
    ///
    /// # Errors
    ///
    /// Returns an error if a node has no valid primary operation or a selected candidate has no operation code.
    pub fn create(
        ir: &Graph,
        execution_order: &[ValueId],
        policy: &LoweringPolicy,
        kernels: &KernelFeasibility,
    ) -> Result<Self> {
        let mut candidates = HashMap::new();
        let mut decisions = HashMap::new();
        let mut operation_decisions = HashMap::new();
        let mut ways = HashMap::new();
        let records = kernels
            .planner()
            .context()
            .records()
            .filter(|records| !records.is_empty());
        for &id in execution_order {
            let ir_operation = ir.required_operation_for_value(id)?;
            let operation_id = ir_operation.id();
            let operation = ir_operation.opcode();
            let mut node_candidates =
                policy.candidates_for_operation(ir, id, ir_operation, kernels);
            // A record keeps its winner, and the generated kernels that can run beside a library's call.
            let key = records.map(|_| way_key(ir, id)).transpose()?;
            if let (Some(records), Some(key)) = (records, &key)
                && let Some(TunedChoice::Way(way)) = records.get(key)
            {
                let kept = |provider: CudaGraphProvider| match way {
                    TunedWay::Library(library) => !provider.is_library() || provider == library,
                    TunedWay::Generated => !provider.is_library(),
                    TunedWay::Separate => true,
                };
                if node_candidates.iter().any(|candidate| match way {
                    TunedWay::Library(library) => candidate.provider() == library,
                    _ => true,
                }) {
                    node_candidates.retain(|candidate| kept(candidate.provider()));
                    ways.insert(id, way);
                    records.note_used(key);
                } else {
                    return Err(Error::invalid_argument(format!(
                        "tuning records chose {way:?} for `{key}`, which the policy does not allow there; tune again, or plan without records"
                    )));
                }
            }

            node_candidates.sort_by_key(|candidate| candidate.cost());
            if let Some((&planned, rejected)) = node_candidates.split_first() {
                let operation = operation.ok_or_else(|| Error::missing_node("plan create"))?;
                let record = LoweringDecisionRecord::new(
                    LoweringDecision::of(planned.provider(), operation),
                    rejected.to_vec(),
                    rejected
                        .iter()
                        .map(|candidate| {
                            LoweringRejection::new(
                                *candidate,
                                format!(
                                    "provider {:?} was rejected by planned provider {:?}",
                                    candidate.provider(),
                                    planned.provider()
                                ),
                            )
                        })
                        .collect(),
                );
                decisions.insert(id, record.clone());
                operation_decisions.insert(operation_id, record);
            } else if operation.is_some() {
                let record = LoweringDecisionRecord::new(
                    LoweringDecision::Unsupported {
                        reason: "no CUDA library provider".to_string(),
                    },
                    Vec::new(),
                    Vec::new(),
                );
                decisions.insert(id, record.clone());
                operation_decisions.insert(operation_id, record);
            }
            if !node_candidates.is_empty() {
                candidates.insert(id, node_candidates);
            }
        }
        Ok(Self {
            policy: policy.clone(),
            candidates,
            decisions,
            operation_decisions,
            ways,
        })
    }
}
