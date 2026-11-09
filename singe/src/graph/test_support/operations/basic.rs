use super::*;
use crate::backend::registry::{OutputRole, operation_schema};

pub fn add_unary_operation(
    graph: &mut Graph,
    operation: Operation,
    input: ValueId,
    output: &Tensor,
) -> ValueId {
    graph.add_operation_fixture(GraphOperationRequest::unary(
        None,
        operation,
        input,
        graph_operation_output(output),
    ))
}

pub fn add_binary_operation(
    graph: &mut Graph,
    operation: Operation,
    inputs: [ValueId; 2],
    output: &Tensor,
) -> ValueId {
    graph.add_operation_fixture(GraphOperationRequest::binary(
        None,
        operation,
        inputs,
        GraphOperationOutput::new(
            if operation_schema(operation).single_output_role() == Some(OutputRole::Mask) {
                DataType::Bool
            } else {
                output.data_type()
            },
            output.shape().clone(),
            output.device(),
        ),
    ))
}

pub fn add_quantized_matmul_operation(
    graph: &mut Graph,
    inputs: [ValueId; 2],
    lhs_scale: f32,
    lhs_zero_point: i64,
    rhs_scale: f32,
    rhs_zero_point: i64,
    output: &Tensor,
) -> ValueId {
    let device = output.device();
    let mut constant = |tensor: Tensor| graph.add_constant(tensor);
    let parameters = [
        constant(Tensor::scalar(device, lhs_scale).unwrap()),
        constant(Tensor::scalar(device, lhs_zero_point).unwrap()),
        constant(Tensor::scalar(device, rhs_scale).unwrap()),
        constant(Tensor::scalar(device, rhs_zero_point).unwrap()),
    ];
    graph.add_operation_fixture(GraphOperationRequest::new(
        None,
        GraphOperationInputs::new(
            Operation::QuantizedMatMul,
            OperationAttributes::None,
            [inputs.as_slice(), parameters.as_slice()].concat(),
        ),
        graph_operation_output(output),
    ))
}

pub fn add_dimension_operation(
    graph: &mut Graph,
    operation: Operation,
    input: ValueId,
    dim: i64,
    output: &Tensor,
) -> ValueId {
    graph.add_operation_fixture(GraphOperationRequest::dimension(
        None,
        operation,
        input,
        dim,
        graph_operation_output(output),
    ))
}

pub fn add_reduction_operation(
    graph: &mut Graph,
    operation: Operation,
    input: ValueId,
    dimensions: Vec<i64>,
    keepdim: bool,
    output: &Tensor,
) -> ValueId {
    graph.add_operation_fixture(GraphOperationRequest::reduction(
        None,
        operation,
        input,
        dimensions,
        keepdim,
        graph_operation_output(output),
    ))
}

pub fn add_scalar_operation(
    graph: &mut Graph,
    operation: Operation,
    input: ValueId,
    scalar: f32,
    output: &Tensor,
) -> ValueId {
    graph.add_operation_fixture(GraphOperationRequest::scalar(
        None,
        operation,
        input,
        scalar,
        graph_operation_output(output),
    ))
}

pub fn add_topk_operation(
    graph: &mut Graph,
    operation: Operation,
    input: ValueId,
    k: i64,
    dim: i64,
    largest: bool,
    sorted: bool,
    output: &Tensor,
) -> ValueId {
    graph.add_operation_fixture(GraphOperationRequest::topk(
        None,
        operation,
        input,
        k,
        dim,
        largest,
        sorted,
        graph_operation_output(output),
    ))
}
