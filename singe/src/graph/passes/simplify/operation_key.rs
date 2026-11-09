use crate::{
    backend::registry::{Operation, OperationAttributeKind, OperationAttributes},
    data_type::DataType,
    device::Device,
    graph::{ir::ValueId, node::NodeRef},
    layout::Layout,
    scalar::Scalar,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OperationKey {
    kind: OperationKeyKind,
    data_type: DataType,
    device: Device,
    layout: Layout,
}

// Two operations share a key only when they compute the same values bit for bit, so arithmetic keys keep their operand order and exact scalars.
pub fn operation_key(node: &NodeRef<'_>) -> Option<OperationKey> {
    let operation = node.operation()?;
    if operation == Operation::Copy {
        return None;
    }

    let kind = if let Some(inputs) = node.binary_elementwise_inputs() {
        OperationKeyKind::Binary {
            op: operation,
            inputs,
        }
    } else if let Some(input) = node.scalar_elementwise_input() {
        OperationKeyKind::TypedScalar {
            op: operation,
            input,
            scalar: node.scalar_operation(operation)?.1,
        }
    } else if let Some(input) = node.unary_elementwise_input() {
        OperationKeyKind::Unary {
            op: operation,
            input,
        }
    } else {
        match operation.attribute_kind() {
            // Custom operations may have effects, and random operations must stay distinct draws.
            OperationAttributeKind::Custom
            | OperationAttributeKind::Strided
            | OperationAttributeKind::Random => return None,
            _ => OperationKeyKind::Generic {
                op: operation,
                attributes: node.attributes(),
                inputs: node.inputs().to_vec(),
            },
        }
    };
    Some(OperationKey {
        kind,
        data_type: node.output_data_type(),
        device: node.output_device(),
        layout: node.output_layout().clone(),
    })
}

/// What makes two operation nodes interchangeable for deduplication.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OperationKeyKind {
    /// A scalar operation compared with its exact typed scalar.
    TypedScalar {
        op: Operation,
        input: ValueId,
        scalar: Scalar,
    },
    /// A unary element-wise operation.
    Unary { op: Operation, input: ValueId },
    /// A binary element-wise operation.
    Binary { op: Operation, inputs: [ValueId; 2] },
    /// Any other operation, compared by its operation, attributes, and inputs.
    Generic {
        op: Operation,
        attributes: OperationAttributes,
        inputs: Vec<ValueId>,
    },
}
