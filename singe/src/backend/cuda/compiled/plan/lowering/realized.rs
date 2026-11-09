use std::collections::HashMap;

use crate::{
    backend::{
        cuda::compiled::plan::lowering::{CudaGraphProvider, LoweringDecision},
        registry::Operation,
    },
    graph::ir::{OperationId, ValueId},
};

/// Actual provider decisions recorded by graph node and IR operation during CUDA compilation.
#[derive(Debug, Clone, Default)]
pub struct RealizedLoweringPlan {
    decisions: HashMap<ValueId, LoweringDecision>,
    operation_decisions: HashMap<OperationId, LoweringDecision>,
}

impl RealizedLoweringPlan {
    /// Records the provider's decision for both node and operation IDs, replacing previous entries for those IDs.
    pub fn record(
        &mut self,
        id: ValueId,
        operation_id: OperationId,
        provider: CudaGraphProvider,
        operation: Operation,
    ) {
        let decision = LoweringDecision::of(provider, operation);
        self.decisions.insert(id, decision.clone());
        self.operation_decisions.insert(operation_id, decision);
    }

    /// Returns the realized provider decision for a node, if recorded.
    #[must_use]
    pub fn decision(&self, id: ValueId) -> Option<&LoweringDecision> {
        self.decisions.get(&id)
    }

    /// Returns the realized provider decision for an IR operation, if recorded.
    #[must_use]
    pub fn decision_for_operation(&self, id: OperationId) -> Option<&LoweringDecision> {
        self.operation_decisions.get(&id)
    }
}
