use super::*;

#[test]
fn test_fold_small_constants_pass_folds_cpu_layer_norm() {
    let input_shape = Shape::create(vec![1, 2]).unwrap();
    let parameter_shape = Shape::create(vec![2]).unwrap();
    let input = Tensor::from_vec(Device::Cpu, &input_shape, vec![3.0_f32, 5.0]).unwrap();
    let weight = Tensor::from_vec(Device::Cpu, &parameter_shape, vec![2.0_f32, 3.0]).unwrap();
    let bias = Tensor::from_vec(Device::Cpu, &parameter_shape, vec![0.5_f32, -1.0]).unwrap();
    let output = input
        .layer_norm(&[2], Some(&weight), Some(&bias), 0.0)
        .unwrap();
    let mut graph = Graph::new();
    let input = graph.add_constant(input);
    let weight = graph.add_constant(weight);
    let bias = graph.add_constant(bias);
    let layer_norm = add_layer_normalization_operation(
        &mut graph,
        RegistryOperation::LayerNorm,
        input,
        vec![2],
        Some(weight),
        Some(bias),
        0.0,
        &output,
    );
    graph.set_outputs(vec![layer_norm]).unwrap();

    let folded = fold_and_assert_checked_nodes(&graph, 4);
    let tensor = folded_constant_tensor(&folded, layer_norm);
    assert_close!(&tensor.to_vec::<f32>().unwrap(), &[-1.5, 2.0]);
}
