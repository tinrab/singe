use super::*;
use crate::layout::Layout;

#[test]
fn test_graph_fingerprint_changes_for_structural_metadata() {
    let shape = Shape::create(vec![2]).unwrap();
    let f32_graph = input_graph("x", DataType::F32, shape.clone(), Device::Cpu);
    let i32_graph = input_graph("x", DataType::I32, shape.clone(), Device::Cpu);
    let shape_graph = input_graph(
        "x",
        DataType::F32,
        Shape::create(vec![3]).unwrap(),
        Device::Cpu,
    );

    let mut layout_graph = Graph::new();
    let strided_shape = Layout::with_strides(vec![2], vec![2]).unwrap();
    let input = layout_graph.add_input("x", DataType::F32, strided_shape.clone(), Device::Cpu);
    layout_graph.replace_node_output_layout_for_test(input, strided_shape);
    layout_graph.set_outputs(vec![input]).unwrap();

    let device_graph = input_graph("x", DataType::F32, shape, Device::Cuda(0));

    let base = f32_graph.fingerprint().unwrap();
    assert_ne!(base, i32_graph.fingerprint().unwrap());
    assert_ne!(base, shape_graph.fingerprint().unwrap());
    assert_ne!(base, layout_graph.fingerprint().unwrap());
    assert_ne!(base, device_graph.fingerprint().unwrap());
}

#[test]
fn test_graph_fingerprint_config_exposes_compiler_version_salt() {
    let shape = Shape::create(vec![2]).unwrap();
    let graph = input_graph("x", DataType::F32, shape, Device::Cpu);

    let base_config = GraphFingerprintOptions::default().with_compiler_version(1);
    let same_config = GraphFingerprintOptions::default().with_compiler_version(1);
    let changed_config = base_config.with_compiler_version(2);

    assert_eq!(base_config.compiler_version(), 1);
    assert_eq!(
        graph.fingerprint_with_options(&base_config).unwrap(),
        graph.fingerprint_with_options(&same_config).unwrap()
    );
    assert_ne!(
        graph.fingerprint_with_options(&base_config).unwrap(),
        graph.fingerprint_with_options(&changed_config).unwrap()
    );
}
