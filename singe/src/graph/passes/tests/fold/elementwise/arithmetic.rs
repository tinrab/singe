use super::*;

#[test]
fn test_fold_small_constants_pass_folds_cpu_arithmetic() {
    let shape = Shape::create(vec![2]).unwrap();
    let lhs = Tensor::from_vec(Device::Cpu, &shape, vec![1.0_f32, 2.0]).unwrap();
    let rhs = Tensor::from_vec(Device::Cpu, &shape, vec![3.0_f32, 4.0]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let mut graph = Graph::new();
    let lhs = graph.add_constant(lhs);
    let rhs = graph.add_constant(rhs);
    let add = add_binary_operation(&mut graph, RegistryOperation::Add, [lhs, rhs], &output);
    let scaled = add_scalar_operation(&mut graph, RegistryOperation::Scale, add, 0.5, &output);
    graph.set_outputs(vec![scaled]).unwrap();

    let folded = fold_small_constants(&graph, 4);
    assert_folded_constants(&folded, [(scaled, &[2.0, 3.0][..])]);
}
