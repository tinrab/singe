use super::*;

#[test]
fn test_fold_small_constants_pass_folds_cpu_pool2d() {
    let input_shape = Shape::create(vec![1, 1, 4, 4]).unwrap();
    let output_shape = Shape::create(vec![1, 1, 2, 2]).unwrap();
    let input = Tensor::from_vec(
        Device::Cpu,
        &input_shape,
        vec![
            1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ],
    )
    .unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &output_shape).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_constant(input);
    let max_pool = add_pool2d_operation(
        &mut graph,
        RegistryOperation::MaxPool2d,
        input,
        (2, 2),
        (2, 2),
        (0, 0, 0, 0),
        (1, 1),
        false,
        false,
        &output,
    );
    let avg_pool = add_pool2d_operation(
        &mut graph,
        RegistryOperation::AvgPool2d,
        input,
        (2, 2),
        (2, 2),
        (0, 0, 0, 0),
        (1, 1),
        false,
        false,
        &output,
    );
    graph.set_outputs(vec![max_pool, avg_pool]).unwrap();

    let pipeline = GraphPassPipeline::new(vec![
        GraphPassKind::FoldSmallConstants,
        GraphPassKind::ValidateValueMetadata,
        GraphPassKind::ValidateNodeContracts,
        GraphPassKind::ValidateOutputs,
    ]);
    let (folded, results) = pipeline.normalize(&graph).unwrap();

    assert_eq!(results[0].pass_name, "fold_small_constants");
    assert_eq!(results[0].checked_nodes, 3);
    let tensor = folded_constant_tensor(&folded, max_pool);
    assert_eq!(tensor.shape().dimensions(), output_shape.dimensions());
    assert_close!(&tensor.to_vec::<f32>().unwrap(), &[6.0, 8.0, 14.0, 16.0]);
    let tensor = folded_constant_tensor(&folded, avg_pool);
    assert_eq!(tensor.shape().dimensions(), output_shape.dimensions());
    assert_close!(&tensor.to_vec::<f32>().unwrap(), &[3.5, 5.5, 11.5, 13.5],);
}

#[test]
fn test_fold_small_constants_pass_folds_cpu_conv2d() {
    let input_shape = Shape::create(vec![1, 1, 3, 3]).unwrap();
    let weight_shape = Shape::create(vec![1, 1, 2, 2]).unwrap();
    let bias_shape = Shape::create(vec![1]).unwrap();
    let output_shape = Shape::create(vec![1, 1, 2, 2]).unwrap();
    let input = Tensor::from_vec(
        Device::Cpu,
        &input_shape,
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    .unwrap();
    let weight =
        Tensor::from_vec(Device::Cpu, &weight_shape, vec![1.0_f32, 0.0, 0.0, 1.0]).unwrap();
    let bias = Tensor::from_vec(Device::Cpu, &bias_shape, vec![0.5_f32]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &output_shape).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_constant(input);
    let weight = graph.add_constant(weight);
    let bias = graph.add_constant(bias);
    let conv2d = add_conv2d_operation(&mut graph, input, weight, Some(bias), &output);
    graph.set_outputs(vec![conv2d]).unwrap();

    let pipeline = GraphPassPipeline::new(vec![
        GraphPassKind::FoldSmallConstants,
        GraphPassKind::ValidateValueMetadata,
        GraphPassKind::ValidateNodeContracts,
        GraphPassKind::ValidateOutputs,
    ]);
    let (folded, results) = pipeline.normalize(&graph).unwrap();

    assert_eq!(results[0].pass_name, "fold_small_constants");
    assert_eq!(results[0].checked_nodes, 4);
    let tensor = folded_constant_tensor(&folded, conv2d);
    assert_eq!(tensor.shape().dimensions(), output_shape.dimensions());
    assert_close!(&tensor.to_vec::<f32>().unwrap(), &[6.5, 8.5, 12.5, 14.5]);
}
