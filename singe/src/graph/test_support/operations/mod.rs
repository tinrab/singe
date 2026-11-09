mod basic;
mod neural;

pub use self::{basic::*, neural::*};
use super::*;

pub fn add_operation_fixture(
    graph: &mut Graph,
    operation: Operation,
    attributes: OperationAttributes,
    inputs: Vec<ValueId>,
    output: &Tensor,
) -> ValueId {
    graph.add_operation_fixture(GraphOperationRequest::new(
        None,
        GraphOperationInputs::new(operation, attributes, inputs),
        graph_operation_output(output),
    ))
}
