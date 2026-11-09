use super::*;
use crate::graph::ir::OperationNode;

#[test]
fn test_graph_compile_prunes_dead_nodes_and_decomposes_quantized_matmul() {
    let mut graph = Graph::new();
    add_dead_scale(&mut graph);

    let lhs_shape = Shape::create([2, 3]).unwrap();
    let rhs_shape = Shape::create([3, 2]).unwrap();
    let output_shape = Shape::create([2, 2]).unwrap();
    let lhs = graph.add_input("lhs", DataType::I8, lhs_shape, Device::Cpu);
    let rhs = graph.add_input("rhs", DataType::U8, rhs_shape, Device::Cpu);
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &output_shape).unwrap();
    let live = add_quantized_matmul_operation(&mut graph, [lhs, rhs], 0.5, -1, 0.25, 2, &output);
    graph.set_outputs(vec![live]).unwrap();

    // The six operands, a dequantization of each side, and their product.
    let compiled = compile_and_assert_pruned_order(&graph, 9);
    let opcodes = compiled
        .to_ir()
        .operations()
        .iter()
        .filter_map(OperationNode::opcode)
        .collect::<Vec<_>>();
    assert_eq!(
        opcodes,
        [
            RegistryOperation::Dequantize,
            RegistryOperation::Dequantize,
            RegistryOperation::MatMul,
        ]
    );
}
