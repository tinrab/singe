use super::*;
use crate::layout::Layout;

#[test]
fn test_graph_guards_reject_under_sized_strided_storage() {
    let shape = Layout::with_strides(vec![2, 2], vec![100, 1]).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, shape.clone(), Device::Cpu);
    graph.replace_node_output_layout_for_test(input, shape.clone());
    graph.set_outputs(vec![input]).unwrap();

    // SAFETY: the storage was built for exactly this shape and data type.
    let tensor = unsafe {
        Tensor::from_parts_unchecked(
            CpuStorage::from_bytes(DataType::F32, vec![0_u8; 16]).unwrap(),
            DataType::F32,
            shape,
        )
    };
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), tensor);

    assert!(matches!(
        graph.run(&inputs),
        Err(Error::LengthMismatch { name, expected, actual })
            if name == "x storage bytes" && expected == 408 && actual == 16
    ));
}
