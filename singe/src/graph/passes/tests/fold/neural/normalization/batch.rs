use super::*;

#[test]
fn test_fold_small_constants_pass_folds_cpu_batch_norm() {
    let shape = Shape::create(vec![2, 2, 2]).unwrap();
    let parameter_shape = Shape::create(vec![2]).unwrap();
    let input = Tensor::from_vec(
        Device::Cpu,
        &shape,
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    )
    .unwrap();
    let weight = Tensor::from_vec(Device::Cpu, &parameter_shape, vec![1.5_f32, 0.5]).unwrap();
    let bias = Tensor::from_vec(Device::Cpu, &parameter_shape, vec![0.25_f32, -0.75]).unwrap();
    let running_mean = Tensor::from_vec(Device::Cpu, &parameter_shape, vec![1.0_f32, 2.0]).unwrap();
    let running_var = Tensor::from_vec(Device::Cpu, &parameter_shape, vec![4.0_f32, 9.0]).unwrap();
    let output = input
        .batch_norm(
            Some(&weight),
            Some(&bias),
            Some(&running_mean),
            Some(&running_var),
            false,
            0.1,
            1e-5,
        )
        .unwrap();
    let mut graph = Graph::new();
    let input = graph.add_constant(input);
    let weight = graph.add_constant(weight);
    let bias = graph.add_constant(bias);
    let running_mean = graph.add_constant(running_mean);
    let running_var = graph.add_constant(running_var);
    let batch_norm = add_batch_normalization_operation(
        &mut graph,
        input,
        Some(weight),
        Some(bias),
        Some(running_mean),
        Some(running_var),
        false,
        0.1,
        1e-5,
        &output,
    );
    graph.set_outputs(vec![batch_norm]).unwrap();

    let folded = fold_and_assert_checked_nodes(&graph, 6);
    let tensor = folded_constant_tensor(&folded, batch_norm);
    assert_close!(
        &tensor.to_vec::<f32>().unwrap(),
        &[
            0.25,
            0.999_999_05,
            -0.583_333_8,
            -0.416_667_22,
            3.249_996_4,
            3.999_995_2,
            0.083_331_11,
            0.249_997_74,
        ],
    );
}
