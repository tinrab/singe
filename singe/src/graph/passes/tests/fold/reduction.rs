use super::*;

#[test]
fn test_fold_small_constants_pass_folds_cpu_reductions() {
    let input_shape = Shape::create(vec![2, 4]).unwrap();
    let reduced_shape = Shape::create(vec![2, 1]).unwrap();
    let topk_shape = Shape::create(vec![2, 2]).unwrap();
    let input = Tensor::from_vec(
        Device::Cpu,
        &input_shape,
        vec![1.0_f32, 4.0, 2.0, 3.0, 9.0, 7.0, 8.0, 6.0],
    )
    .unwrap();
    let sum_output = input.sum(&[1], true).unwrap();
    let (topk_values_output, topk_indices_output) = input.topk(2, 1, true, true).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_constant(input);
    let sum_output_tensor = Tensor::zeroed_on(Device::Cpu, DataType::F32, &reduced_shape).unwrap();
    let sum = add_reduction_operation(
        &mut graph,
        RegistryOperation::Sum,
        input,
        vec![1],
        true,
        &sum_output_tensor,
    );
    let topk_values = add_topk_operation(
        &mut graph,
        RegistryOperation::TopKValues,
        input,
        2,
        1,
        true,
        true,
        &topk_values_output,
    );
    let topk_indices = add_topk_operation(
        &mut graph,
        RegistryOperation::TopKIndices,
        input,
        2,
        1,
        true,
        true,
        &topk_indices_output,
    );
    graph
        .set_outputs(vec![sum, topk_values, topk_indices])
        .unwrap();

    let pipeline = GraphPassPipeline::new(vec![
        GraphPassKind::FoldSmallConstants,
        GraphPassKind::ValidateValueMetadata,
        GraphPassKind::ValidateNodeContracts,
        GraphPassKind::ValidateOutputs,
    ]);
    let (folded, results) = pipeline.normalize(&graph).unwrap();

    assert_eq!(results[0].pass_name, "fold_small_constants");
    assert_eq!(results[0].checked_nodes, 4);
    let tensor = folded_constant_tensor(&folded, sum);
    assert_eq!(tensor.shape().dimensions(), sum_output.shape().dimensions());
    assert_close!(&tensor.to_vec::<f32>().unwrap(), &[10.0, 30.0]);
    let tensor = folded_constant_tensor(&folded, topk_values);
    assert_eq!(tensor.shape().dimensions(), topk_shape.dimensions());
    assert_close!(&tensor.to_vec::<f32>().unwrap(), &[4.0, 3.0, 9.0, 8.0]);
    let tensor = folded_constant_tensor(&folded, topk_indices);
    assert_eq!(tensor.shape().dimensions(), topk_shape.dimensions());
    assert_eq!(tensor.to_vec::<i64>().unwrap(), vec![1, 3, 0, 2]);
}
