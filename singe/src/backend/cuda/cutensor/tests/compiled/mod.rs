mod empty;
mod replay;
mod report;

use super::*;
use crate::graph::ir::ValueId;

fn cuda_device() -> Device {
    Device::Cuda(0)
}

fn f32_tensor(shape: &Shape, data: Vec<f32>) -> Tensor {
    Tensor::from_vec(cuda_device(), shape, data).unwrap()
}

fn maybe_cuda_f32_tensor(shape: &Shape, data: Vec<f32>) -> Option<Tensor> {
    skip_if_stub(Tensor::from_vec(cuda_device(), shape, data))
}

fn cuda_f32_output(shape: &Shape) -> Tensor {
    Tensor::zeroed_on(cuda_device(), DataType::F32, shape).unwrap()
}

fn contraction_inputs(a: Tensor, b: Tensor, c: Tensor) -> HashMap<String, Tensor> {
    HashMap::from([
        ("a".to_string(), a),
        ("b".to_string(), b),
        ("c".to_string(), c),
    ])
}

fn contraction_graph(
    a_shape: Shape,
    b_shape: Shape,
    c_shape: Shape,
    output: &Tensor,
    alpha: f32,
    beta: f32,
) -> (Graph, ValueId) {
    let mut graph = Graph::new();
    let a_id = graph.add_input("a", DataType::F32, a_shape, cuda_device());
    let b_id = graph.add_input("b", DataType::F32, b_shape, cuda_device());
    let c_id = graph.add_input("c", DataType::F32, c_shape, cuda_device());
    let contraction = graph
        .add_operation_request(GraphOperationRequest::contraction(
            None,
            a_id,
            vec![0, 2],
            b_id,
            vec![2, 1],
            c_id,
            vec![0, 1],
            vec![0, 1],
            alpha,
            beta,
            operation_output(output),
        ))
        .unwrap();
    graph.set_outputs(vec![contraction]).unwrap();
    (graph, contraction)
}
