use super::*;

#[test]
fn test_fold_small_constants_pass_folds_cpu_scaled_dot_product_attention() {
    let query_shape = Shape::create(vec![1, 1, 2, 2]).unwrap();
    let key_shape = Shape::create(vec![1, 1, 3, 2]).unwrap();
    let value_shape = Shape::create(vec![1, 1, 3, 2]).unwrap();
    let output_shape = Shape::create(vec![1, 1, 2, 2]).unwrap();
    let query = Tensor::from_vec(Device::Cpu, &query_shape, vec![1.0_f32, 0.0, 0.0, 1.0]).unwrap();
    let key = Tensor::from_vec(
        Device::Cpu,
        &key_shape,
        vec![1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let value = Tensor::from_vec(
        Device::Cpu,
        &value_shape,
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &output_shape).unwrap();
    let mut graph = Graph::new();
    let query = graph.add_constant(query);
    let key = graph.add_constant(key);
    let value = graph.add_constant(value);
    let attention = add_scaled_dot_product_attention_operation(
        &mut graph,
        query,
        key,
        value,
        Some(0.5),
        false,
        &output,
    );
    graph.set_outputs(vec![attention]).unwrap();

    let pipeline = GraphPassPipeline::new(vec![
        GraphPassKind::FoldSmallConstants,
        GraphPassKind::ValidateValueMetadata,
        GraphPassKind::ValidateNodeContracts,
        GraphPassKind::ValidateOutputs,
    ]);
    let (folded, results) = pipeline.normalize(&graph).unwrap();

    assert_eq!(results[0].pass_name, "fold_small_constants");
    assert_eq!(results[0].checked_nodes, 4);
    let tensor = folded_constant_tensor(&folded, attention);
    assert_close!(
        &tensor.to_vec::<f32>().unwrap(),
        &[3.0, 4.0, 3.301_910_4, 4.301_910_4],
        1e-5,
    );
}
