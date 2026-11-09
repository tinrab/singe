use crate::{
    backend::registry::{Operation, OperationAttributes},
    graph::{
        ir::ValueId,
        operation::{GraphOperationInputs, GraphOperationOutput, GraphOperationRequest},
    },
};

impl GraphOperationRequest {
    /// Creates an attribute-free single-input operation request.
    /// The request stores operands and attributes without validating them; graph insertion performs logical inference.
    #[must_use]
    pub fn unary(
        name: Option<String>,
        operation: Operation,
        input: ValueId,
        output: GraphOperationOutput,
    ) -> Self {
        Self {
            name,
            inputs: GraphOperationInputs::new(operation, OperationAttributes::None, vec![input]),
            output,
        }
    }

    /// Creates an attribute-free binary request in left and right operand order.
    /// The request stores operands and attributes without validating them; graph insertion performs logical inference.
    #[must_use]
    pub fn binary(
        name: Option<String>,
        operation: Operation,
        inputs: [ValueId; 2],
        output: GraphOperationOutput,
    ) -> Self {
        Self {
            name,
            inputs: GraphOperationInputs::new(
                operation,
                OperationAttributes::None,
                inputs.to_vec(),
            ),
            output,
        }
    }
}
