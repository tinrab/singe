mod accepted;
mod validation;

use super::*;

use crate::graph::node::NodeKind;

fn build_permute_graph(input_shape: Shape, output: &Tensor, order: Vec<i64>) -> Graph {
    let output_shape = output.shape().clone();
    let mut graph = Graph::new();
    let x = graph.add_input("x", output.data_type(), input_shape, output.device());
    let permuted = graph
        .add_named_operation(
            None,
            NodeKind::permute(x, order),
            output.data_type(),
            output_shape,
            output.device(),
        )
        .unwrap();
    graph.set_outputs(vec![permuted]).unwrap();

    graph
}

fn build_invalid_permute_graph(
    output_shape: &Shape,
    order: Vec<i64>,
) -> Option<Result<Graph, Error>> {
    let input_shape = Shape::create([2, 3]).unwrap();
    let output = skip_if_stub(Tensor::zeroed_on(
        Device::Cuda(0),
        DataType::F64,
        output_shape,
    ))?;
    Some((|| {
        let mut graph = Graph::new();
        let input = graph.add_input("x", output.data_type(), input_shape, output.device());
        let permuted = graph.add_named_operation(
            None,
            NodeKind::permute(input, order),
            output.data_type(),
            output.shape().clone(),
            output.device(),
        )?;
        graph.set_outputs(vec![permuted])?;
        Ok(graph)
    })())
}
