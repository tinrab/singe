use super::*;

use crate::graph::node::NodeKind;

#[test]
fn test_infer_value_metadata_pass_infers_transpose_view_layout() {
    let input_shape = Shape::create(vec![2, 3]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &input_shape)
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, input_shape, Device::Cpu);
    let transpose = graph
        .add_named_operation(
            None,
            NodeKind::transpose(input, 0, 1),
            output.data_type(),
            output.shape().clone(),
            output.device(),
        )
        .unwrap();
    replace_with_stale_shape_metadata(&mut graph, transpose);
    graph.set_outputs(vec![transpose]).unwrap();

    let pipeline = GraphPassPipeline::new(vec![
        GraphPassKind::InferValueMetadata,
        GraphPassKind::ValidateValueMetadata,
        GraphPassKind::ValidateNodeContracts,
        GraphPassKind::ValidateOutputs,
    ]);
    let (normalized, _) = pipeline.normalize(&graph).unwrap();

    assert_value_shape(&normalized, transpose, &[3, 2]);
    assert_value_layout(&normalized, transpose, &[3, 2], &[1, 3], 0);
}

#[test]
fn test_infer_value_metadata_pass_infers_slice_view_offset() {
    let input_shape = Shape::create(vec![3, 4]).unwrap();
    let output_shape = Shape::create(vec![1, 4]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &output_shape).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, input_shape, Device::Cpu);
    let slice = graph
        .add_named_slice_operation(
            None,
            input,
            0,
            1,
            1,
            output.data_type(),
            output.shape(),
            output.device(),
        )
        .unwrap();
    replace_with_stale_shape_metadata(&mut graph, slice);
    graph.set_outputs(vec![slice]).unwrap();

    let pipeline = GraphPassPipeline::new(vec![
        GraphPassKind::InferValueMetadata,
        GraphPassKind::ValidateValueMetadata,
        GraphPassKind::ValidateNodeContracts,
        GraphPassKind::ValidateOutputs,
    ]);
    let (normalized, _) = pipeline.normalize(&graph).unwrap();

    assert_value_shape(&normalized, slice, output_shape.dimensions());
    assert_value_layout(&normalized, slice, output_shape.dimensions(), &[4, 1], 4);
}
