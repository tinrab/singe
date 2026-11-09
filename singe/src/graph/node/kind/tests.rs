use crate::{
    backend::registry::{Operation, OperationAttributeKind, registered_operations},
    data_type::DataType,
    graph::{ir::ValueId, node::kind::NodeKind},
};

fn operation_of(kind: &NodeKind) -> Option<Operation> {
    match kind {
        NodeKind::Operation(invocation) => Some(invocation.operation),
        NodeKind::Input { .. } | NodeKind::Constant { .. } => None,
    }
}

#[test]
fn test_generated_operation_family_builders_round_trip_operation_metadata() {
    let input = ValueId::new(1);
    for &operation in registered_operations() {
        if operation.has_attribute_kind(OperationAttributeKind::Quantization) {
            let kind =
                NodeKind::quantization_operation(operation, input, input, input, DataType::U8)
                    .unwrap();
            assert_eq!(operation_of(&kind), Some(operation));
        }
    }
}
