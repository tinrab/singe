use super::*;
use crate::{graph::node::NodeKind, layout::Layout};

#[test]
fn test_graph_validates_input_metadata() {
    let shape = Shape::create(vec![2]).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, shape.clone(), Device::Cuda(0));
    graph.set_outputs(vec![input]).unwrap();

    let mut inputs = HashMap::new();
    inputs.insert(
        "x".to_string(),
        Tensor::from_vec(Device::Cpu, &shape, vec![1.0_f32, 2.0]).unwrap(),
    );
    assert!(matches!(
        graph.run(&inputs),
        Err(Error::DeviceMismatch { name, .. }) if name == "x"
    ));

    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, shape.clone(), Device::Cpu);
    graph.set_outputs(vec![input]).unwrap();

    let wrong_shape = Shape::create(vec![3]).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert(
        "x".to_string(),
        Tensor::from_vec(Device::Cpu, &wrong_shape, vec![1.0_f32, 2.0, 3.0]).unwrap(),
    );
    assert!(matches!(
        graph.run(&inputs),
        Err(Error::ShapeMismatch { name, .. }) if name == "x"
    ));

    let mut inputs = HashMap::new();
    inputs.insert(
        "x".to_string(),
        Tensor::from_vec(Device::Cpu, &shape, vec![1_i32, 2]).unwrap(),
    );
    assert!(matches!(
        graph.run(&inputs),
        Err(Error::DataTypeMismatch { name, .. }) if name == "x"
    ));

    let layout_shape = Shape::create(vec![2, 2]).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, layout_shape.clone(), Device::Cpu);
    graph.set_outputs(vec![input]).unwrap();

    let non_contiguous = Tensor::from_vec(Device::Cpu, &layout_shape, vec![1.0_f32, 2.0, 3.0, 4.0])
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), non_contiguous);
    assert!(matches!(
        graph.run(&inputs),
        Err(Error::LayoutMismatch { name, .. }) if name == "x"
    ));
}

#[test]
fn test_graph_compile_uses_the_authoritative_input_layout() {
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, Shape::create([2]).unwrap(), Device::Cpu);
    graph.replace_node_output_layout_for_test(input, Layout::contiguous([3]).unwrap());
    graph.set_outputs(vec![input]).unwrap();
    let compiled = graph.compile().unwrap();
    assert_eq!(compiled.signature().inputs()[0].layout().dimensions(), [3]);
    let wrong =
        Tensor::from_vec(Device::Cpu, Shape::create([2]).unwrap(), vec![1_f32, 2.]).unwrap();
    assert!(
        matches!(compiled.run(&HashMap::from([("x".into(), wrong)])), Err(Error::ShapeMismatch { name, expected, actual }) if name == "x" && expected == [3] && actual == [2])
    );
}

#[test]
fn test_graph_compile_rejects_alias_layout_metadata_mismatch() {
    let input_shape = Shape::create(vec![2, 2]).unwrap();
    let output_shape = Shape::create(vec![4]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &output_shape).unwrap();

    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, input_shape, Device::Cpu);
    let view = graph
        .add_named_operation(
            None,
            NodeKind::reshape(input, output_shape.clone()),
            output.data_type(),
            output_shape,
            output.device(),
        )
        .unwrap();
    graph.replace_node_output_layout_for_test(
        view,
        Layout::with_strides(vec![4], vec![2])
            .unwrap()
            .with_offset(1),
    );
    graph.set_outputs(vec![view]).unwrap();

    let error = graph.compile().unwrap_err();
    assert!(matches!(
        &error,
        Error::LayoutMismatch {
            name,
            expected_dimensions,
            expected_strides,
            expected_offset,
            actual_dimensions,
            actual_strides,
            actual_offset,
        } if name == "graph pass validate_value_metadata node 1 alias layout"
            && expected_dimensions.as_ref() == [4]
            && expected_strides.as_ref() == [1]
            && *expected_offset == 0
            && actual_dimensions.as_ref() == [4]
            && actual_strides.as_ref() == [2]
            && *actual_offset == 1
    ));
    assert_eq!(
        error.to_string(),
        "layout mismatch for `graph pass validate_value_metadata node 1 alias layout`: expected dimensions [4] strides [1] offset 0, got dimensions [4] strides [2] offset 1"
    );
}

#[test]
fn test_graph_compile_rejects_schema_alias_layout_metadata_mismatch() {
    let input_shape = Shape::create(vec![2, 2]).unwrap();
    let output_shape = input_shape.clone();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &output_shape).unwrap();

    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, input_shape, Device::Cpu);
    let copy = graph
        .add_named_operation(
            None,
            NodeKind::copy_to(input, Device::Cpu),
            output.data_type(),
            output.shape().clone(),
            output.device(),
        )
        .unwrap();
    graph.replace_node_output_layout_for_test(
        copy,
        Layout::with_strides(vec![2, 2], vec![1, 3]).unwrap(),
    );
    graph.set_outputs(vec![copy]).unwrap();

    assert!(matches!(
        graph.compile(),
        Err(Error::LayoutMismatch { name, .. })
            if name == "graph pass validate_value_metadata node 1 alias layout"
    ));
}

#[test]
fn test_graph_exposes_view_metadata() {
    let input_shape = Shape::create(vec![2, 2]).unwrap();
    let output_shape = Shape::create(vec![4]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &output_shape).unwrap();

    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, input_shape, Device::Cpu);
    let view = graph
        .add_named_operation(
            None,
            NodeKind::reshape(input, output_shape.clone()),
            output.data_type(),
            output_shape,
            output.device(),
        )
        .unwrap();

    graph.set_outputs(vec![view]).unwrap();
    let ir = graph.to_ir().unwrap();
    let input_value = input;
    let view_value = view;
    assert_eq!(
        ir.required_value_alias(view_value)
            .unwrap()
            .unwrap()
            .source()
            .index(),
        input.index()
    );
    assert_eq!(
        ir.required_value_layout(view_value).unwrap().dimensions(),
        &[4]
    );
    assert!(ir.value(input_value).unwrap().alias().is_none());
}

#[test]
fn test_graph_run_accepts_metadata_only_strided_view_layout() {
    let input_shape = Shape::create(vec![2, 3]).unwrap();
    let output_shape = input_shape.transpose_2d().unwrap();

    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, input_shape.clone(), Device::Cpu);
    let transpose = graph
        .add_named_operation(
            None,
            NodeKind::transpose(input, 0, 1),
            DataType::F32,
            output_shape,
            Device::Cpu,
        )
        .unwrap();
    graph.set_outputs(vec![transpose]).unwrap();

    let mut inputs = HashMap::new();
    inputs.insert(
        "x".to_string(),
        Tensor::from_vec(
            Device::Cpu,
            &input_shape,
            vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap(),
    );

    let outputs = graph.run(&inputs).unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].shape().dimensions(), &[3, 2]);
    assert_eq!(outputs[0].layout().strides(), &[1, 3]);
    assert_eq!(
        outputs[0].to_vec::<f32>().unwrap(),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
}
