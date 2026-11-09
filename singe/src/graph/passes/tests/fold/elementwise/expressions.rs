use super::*;

#[test]
fn test_fold_small_constants_pass_folds_cpu_unary_and_binary_expressions() {
    let shape = Shape::create(vec![2]).unwrap();
    let input = Tensor::from_vec(Device::Cpu, &shape, vec![-4.0_f32, 9.0]).unwrap();
    let rhs = Tensor::from_vec(Device::Cpu, &shape, vec![3.0_f32, 2.0]).unwrap();
    let abs_output = input.abs().unwrap();
    let sqrt_output = abs_output.sqrt().unwrap();
    let maximum_output = sqrt_output.maximum(&rhs).unwrap();
    let less_output = sqrt_output.less(&rhs).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_constant(input);
    let rhs = graph.add_constant(rhs);
    let abs = add_unary_operation(&mut graph, RegistryOperation::Abs, input, &abs_output);
    let sqrt = add_unary_operation(&mut graph, RegistryOperation::Sqrt, abs, &sqrt_output);
    let maximum = add_binary_operation(
        &mut graph,
        RegistryOperation::Maximum,
        [sqrt, rhs],
        &maximum_output,
    );
    let less = add_binary_operation(
        &mut graph,
        RegistryOperation::Less,
        [sqrt, rhs],
        &less_output,
    );
    graph.set_outputs(vec![sqrt, maximum, less]).unwrap();

    let folded = fold_small_constants(&graph, 6);
    assert_folded_constants(&folded, [(sqrt, &[2.0, 3.0][..]), (maximum, &[3.0, 3.0])]);
    assert_eq!(
        folded_constant_tensor(&folded, less)
            .to_vec::<bool>()
            .unwrap(),
        [true, false]
    );
}
