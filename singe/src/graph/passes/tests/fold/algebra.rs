use super::*;

#[test]
fn test_fold_small_constants_pass_folds_cpu_matmul() {
    let lhs_shape = Shape::create(vec![2, 3]).unwrap();
    let rhs_shape = Shape::create(vec![3, 2]).unwrap();
    let output_shape = Shape::create(vec![2, 2]).unwrap();
    let lhs = Tensor::from_vec(
        Device::Cpu,
        &lhs_shape,
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();
    let rhs = Tensor::from_vec(
        Device::Cpu,
        &rhs_shape,
        vec![7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0],
    )
    .unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &output_shape).unwrap();
    let mut graph = Graph::new();
    let lhs = graph.add_constant(lhs);
    let rhs = graph.add_constant(rhs);
    let matmul = add_binary_operation(&mut graph, RegistryOperation::MatMul, [lhs, rhs], &output);
    graph.set_outputs(vec![matmul]).unwrap();

    let pipeline = GraphPassPipeline::new(vec![
        GraphPassKind::FoldSmallConstants,
        GraphPassKind::ValidateValueMetadata,
        GraphPassKind::ValidateNodeContracts,
        GraphPassKind::ValidateOutputs,
    ]);
    let (folded, results) = pipeline.normalize(&graph).unwrap();

    assert_eq!(results[0].pass_name, "fold_small_constants");
    assert_eq!(results[0].checked_nodes, 3);
    let tensor = folded_constant_tensor(&folded, matmul);
    assert_eq!(tensor.shape().dimensions(), output_shape.dimensions());
    assert_close!(
        &tensor.to_vec::<f32>().unwrap(),
        &[58.0, 64.0, 139.0, 154.0]
    );
}

#[test]
fn test_fold_small_constants_pass_folds_cpu_contraction() {
    let a_shape = Shape::create(vec![2, 3]).unwrap();
    let b_shape = Shape::create(vec![3, 2]).unwrap();
    let c_shape = Shape::create(vec![2, 2]).unwrap();
    let a = Tensor::from_vec(
        Device::Cpu,
        &a_shape,
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();
    let b = Tensor::from_vec(
        Device::Cpu,
        &b_shape,
        vec![7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0],
    )
    .unwrap();
    let c = Tensor::zeroed_on(Device::Cpu, DataType::F32, &c_shape).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &c_shape).unwrap();
    let mut graph = Graph::new();
    let a = graph.add_constant(a);
    let b = graph.add_constant(b);
    let c = graph.add_constant(c);
    let contraction = add_contraction_operation(
        &mut graph,
        a,
        vec![0, 2],
        b,
        vec![2, 1],
        c,
        vec![0, 1],
        vec![0, 1],
        1.0,
        0.0,
        &output,
    );
    graph.set_outputs(vec![contraction]).unwrap();

    let pipeline = GraphPassPipeline::new(vec![
        GraphPassKind::FoldSmallConstants,
        GraphPassKind::ValidateValueMetadata,
        GraphPassKind::ValidateNodeContracts,
        GraphPassKind::ValidateOutputs,
    ]);
    let (folded, results) = pipeline.normalize(&graph).unwrap();

    assert_eq!(results[0].pass_name, "fold_small_constants");
    assert_eq!(results[0].checked_nodes, 4);
    let tensor = folded_constant_tensor(&folded, contraction);
    assert_eq!(tensor.shape().dimensions(), c_shape.dimensions());
    assert_close!(
        &tensor.to_vec::<f32>().unwrap(),
        &[58.0, 64.0, 139.0, 154.0],
        1e-5,
    );
}
