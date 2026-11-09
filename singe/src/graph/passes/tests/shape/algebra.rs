use super::*;

fn stale_matmul_graph(
    lhs_shape: Shape,
    rhs_shape: Shape,
    output_shape: &Shape,
) -> (Graph, ValueId) {
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, output_shape).unwrap();
    let mut graph = Graph::new();
    let lhs = graph.add_input("lhs", DataType::F32, lhs_shape, Device::Cpu);
    let rhs = graph.add_input("rhs", DataType::F32, rhs_shape, Device::Cpu);
    let matmul = add_binary_operation(&mut graph, RegistryOperation::MatMul, [lhs, rhs], &output);
    replace_with_stale_shape_metadata(&mut graph, matmul);
    graph.set_outputs(vec![matmul]).unwrap();

    (graph, matmul)
}

fn assert_matmul_shape(lhs: Vec<i64>, rhs: Vec<i64>, output: Vec<i64>) {
    let output_shape = Shape::create(output).unwrap();
    let (graph, matmul) = stale_matmul_graph(
        Shape::create(lhs).unwrap(),
        Shape::create(rhs).unwrap(),
        &output_shape,
    );

    let normalized = infer_and_validate_value_metadata(&graph);
    assert_value_metadata(
        &normalized,
        matmul,
        DataType::F32,
        output_shape.dimensions(),
        output_shape.dimensions(),
        Device::Cpu,
    );
}

fn assert_matmul_error(
    lhs: Vec<i64>,
    rhs: Vec<i64>,
    stale_output: Vec<i64>,
    expected: usize,
    actual: usize,
    expected_name: &str,
) {
    let stale_output_shape = Shape::create(stale_output).unwrap();
    let (graph, _) = stale_matmul_graph(
        Shape::create(lhs).unwrap(),
        Shape::create(rhs).unwrap(),
        &stale_output_shape,
    );

    assert!(matches!(
        infer_value_metadata_error(&graph),
        Error::LengthMismatch { name, expected: error_expected, actual: error_actual } if name == expected_name && error_expected == expected && error_actual == actual
    ));
}

#[test]
fn test_infer_value_metadata_pass_infers_matmul_shape() {
    assert_matmul_shape(vec![2, 3], vec![3, 4], vec![2, 4]);
}

#[test]
fn test_infer_value_metadata_pass_infers_batched_matmul_shape() {
    assert_matmul_shape(vec![5, 2, 3], vec![5, 3, 4], vec![5, 2, 4]);
}

#[test]
fn test_infer_value_metadata_pass_infers_multi_batched_matmul_shape() {
    assert_matmul_shape(vec![2, 5, 3, 4], vec![2, 5, 4, 6], vec![2, 5, 3, 6]);
}

#[test]
fn test_infer_value_metadata_pass_rejects_mismatched_multi_batched_matmul_shape() {
    let (graph, _) = stale_matmul_graph(
        Shape::create([2, 5, 3, 4]).unwrap(),
        Shape::create([2, 6, 4, 7]).unwrap(),
        &Shape::create([2, 5, 3, 7]).unwrap(),
    );

    assert!(matches!(
        infer_value_metadata_error(&graph),
        Error::BroadcastMismatch { name, left, right }
            if name == "operation matmul batch dimensions"
                && left == [2, 5]
                && right == [2, 6]
    ));
}

#[test]
fn test_infer_value_metadata_pass_broadcasts_matmul_batch_ranks() {
    assert_matmul_shape(vec![2, 5, 3, 4], vec![5, 4, 7], vec![2, 5, 3, 7]);
}

#[test]
fn test_infer_value_metadata_pass_infers_vector_dot_scalar() {
    assert_matmul_shape(vec![4], vec![4], vec![]);
}

#[test]
fn test_infer_value_metadata_rejects_scalar_matmul_operand() {
    assert_matmul_error(
        vec![],
        vec![1],
        vec![1],
        2,
        0,
        "operation matmul lhs minimum rank",
    );
}
