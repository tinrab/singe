use crate::{
    backend::registry::{Operation, OperationAttributes},
    graph::{ir::ValueId, node::kind::NodeKind},
};

impl NodeKind {
    pub(crate) fn embedding(weight: ValueId, indices: ValueId) -> Self {
        Self::operation_invocation(
            Operation::Embedding,
            OperationAttributes::None,
            vec![weight, indices],
        )
    }

    pub(crate) fn repeat_kv(input: ValueId, repeats: i64) -> Self {
        Self::operation_invocation(
            Operation::RepeatKv,
            OperationAttributes::RepeatKv(repeats),
            vec![input],
        )
    }
}
