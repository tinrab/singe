use super::*;
use crate::layout::Layout;

#[test]
fn test_input_guard_exposes_metadata_through_accessors() {
    let shape = Shape::create(vec![2]).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, shape.clone(), Device::Cpu);
    add_f32_initializer(&mut graph, "x", &shape, vec![1.0_f32, 2.0]);
    graph
        .add_input_dimension_guard("x", 0, "batch", Some(1), Some(4))
        .unwrap();
    graph.set_outputs(vec![input]).unwrap();

    let compiled = graph.compile().unwrap();
    let guards = create_guards_from_ir(&compiled);
    let guard = &guards.inputs()[0];

    assert_eq!(guard.name(), "x");
    assert_eq!(guard.data_type(), DataType::F32);
    assert_eq!(guard.device(), Device::Cpu);
    assert_eq!(guard.layout(), &Layout::from_shape(&shape));
    assert_eq!(guard.storage_span_bytes(), 8);
    assert!(guard.has_initializer());
    assert!(!guard.is_static_initializer());
    assert_eq!(guard.dimension_guards().len(), 1);
}
