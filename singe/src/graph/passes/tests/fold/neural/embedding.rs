use super::*;

use crate::graph::node::NodeKind;

#[test]
fn test_fold_small_constants_pass_folds_cpu_embedding() {
    let weight_shape = Shape::create(vec![3, 2]).unwrap();
    let indices_shape = Shape::create(vec![2]).unwrap();
    let output_shape = Shape::create(vec![2, 2]).unwrap();
    let weight = Tensor::from_vec(
        Device::Cpu,
        &weight_shape,
        vec![1.0_f32, 2.0, 10.0, 20.0, 100.0, 200.0],
    )
    .unwrap();
    let indices = Tensor::from_vec(Device::Cpu, &indices_shape, vec![2_i64, 0]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &output_shape).unwrap();
    let mut graph = Graph::new();
    let weight = graph.add_constant(weight);
    let indices = graph.add_constant(indices);
    let embedding = graph
        .add_named_operation(
            None,
            NodeKind::embedding(weight, indices),
            output.data_type(),
            output.shape().clone(),
            output.device(),
        )
        .unwrap();
    graph.set_outputs(vec![embedding]).unwrap();

    let pipeline = GraphPassPipeline::new(vec![
        GraphPassKind::FoldSmallConstants,
        GraphPassKind::ValidateValueMetadata,
        GraphPassKind::ValidateNodeContracts,
        GraphPassKind::ValidateOutputs,
    ]);
    let (folded, results) = pipeline.normalize(&graph).unwrap();

    assert_eq!(results[0].pass_name, "fold_small_constants");
    assert_eq!(results[0].checked_nodes, 3);
    let tensor = folded_constant_tensor(&folded, embedding);
    assert_close!(&tensor.to_vec::<f32>().unwrap(), &[100.0, 200.0, 1.0, 2.0],);
}

#[test]
fn test_fold_small_constants_pass_folds_cpu_rotary_embedding() {
    let input_shape = Shape::create(vec![1, 1, 2, 4]).unwrap();
    let trig_shape = Shape::create(vec![2, 1]).unwrap();
    let input = Tensor::from_vec(
        Device::Cpu,
        &input_shape,
        vec![1.0_f32, 2.0, 9.0, 10.0, 3.0, 4.0, 11.0, 12.0],
    )
    .unwrap();
    let cos = Tensor::from_vec(Device::Cpu, &trig_shape, vec![1.0_f32, 0.0]).unwrap();
    let sin = Tensor::from_vec(Device::Cpu, &trig_shape, vec![0.0_f32, 1.0]).unwrap();
    let output = input.rotary_embedding(&cos, &sin).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_constant(input);
    let cos = graph.add_constant(cos);
    let sin = graph.add_constant(sin);
    let rotary = add_rotary_embedding_operation(&mut graph, input, cos, sin, &output);
    graph.set_outputs(vec![rotary]).unwrap();

    let pipeline = GraphPassPipeline::new(vec![
        GraphPassKind::FoldSmallConstants,
        GraphPassKind::ValidateValueMetadata,
        GraphPassKind::ValidateNodeContracts,
        GraphPassKind::ValidateOutputs,
    ]);
    let (folded, results) = pipeline.normalize(&graph).unwrap();

    assert_eq!(results[0].pass_name, "fold_small_constants");
    assert_eq!(results[0].checked_nodes, 4);
    let tensor = folded_constant_tensor(&folded, rotary);
    assert_close!(
        &tensor.to_vec::<f32>().unwrap(),
        &[1.0, 2.0, 9.0, 10.0, -4.0, 3.0, 11.0, 12.0],
    );
}
