#![cfg(feature = "onnx")]

use std::collections::HashMap;

use prost::Message;
use singe::{
    DataType, Device, OnnxModel, OnnxOptions, Shape, Tensor,
    error::Error as SingeError,
    graph::{GraphPrepareOptions, ProgramInput},
    onnx::Error,
};
use singe_onnx::proto::onnx::{
    AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
    TensorShapeProto, TypeProto, ValueInfoProto, attribute_proto, tensor_proto, tensor_shape_proto,
    type_proto,
};

fn model(operator: &str, version: i64) -> ModelProto {
    ModelProto {
        ir_version: 10,
        graph: Some(GraphProto {
            name: "compatibility".into(),
            node: vec![NodeProto {
                op_type: operator.into(),
                input: vec!["x".into()],
                output: vec!["y".into()],
                ..Default::default()
            }],
            input: vec![value_info("x", &[1, 2, 2])],
            output: vec![value_info("y", &[1, 2, 2])],
            ..Default::default()
        }),
        opset_import: vec![OperatorSetIdProto {
            version,
            ..Default::default()
        }],
        ..Default::default()
    }
}

fn value_info(name: &str, dimensions: &[i64]) -> ValueInfoProto {
    typed_value_info(name, dimensions, tensor_proto::DataType::Float)
}

fn typed_value_info(
    name: &str,
    dimensions: &[i64],
    data_type: tensor_proto::DataType,
) -> ValueInfoProto {
    ValueInfoProto {
        name: name.into(),
        r#type: Some(TypeProto {
            value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                elem_type: data_type as i32,
                shape: Some(TensorShapeProto {
                    dim: dimensions
                        .iter()
                        .copied()
                        .map(|dimension| tensor_shape_proto::Dimension {
                            value: Some(tensor_shape_proto::dimension::Value::DimValue(dimension)),
                            ..Default::default()
                        })
                        .collect(),
                }),
            })),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn import(model: &ModelProto) -> Result<OnnxModel, Error> {
    OnnxModel::from_reader(model.encode_to_vec().as_slice(), OnnxOptions::new())
}

#[test]
fn test_scalar_vector_and_empty_input_shapes_survive_onnx_import() {
    for dimensions in [vec![], vec![1], vec![0]] {
        let mut source = model("Identity", 13);
        let graph = source.graph.as_mut().unwrap();
        graph.input = vec![value_info("x", &dimensions)];
        graph.output = vec![value_info("y", &dimensions)];
        let imported = import(&source).unwrap();
        let shape = Shape::create(dimensions).unwrap();
        let input = Tensor::full(Device::Cpu, &shape, 3.5_f32).unwrap();
        let output = imported
            .run(&HashMap::from([("x".into(), input.clone())]))
            .unwrap();
        assert_eq!(output["y"].shape(), &shape);
        assert_eq!(output["y"].to_bytes().unwrap(), input.to_bytes().unwrap());
    }
}

#[test]
fn test_imported_graphs_use_prepared_analysis_and_preserve_output_names() {
    let imported = import(&model("Relu", 13)).unwrap();
    let prepared = imported.compile().unwrap();
    assert_eq!(prepared.signature().outputs()[0].name(), Some("y"));
    for values in [[-1.0_f32, 2., -3., 4.], [5., -6., 7., -8.]] {
        let input = Tensor::from_vec(
            Device::Cpu,
            Shape::create(vec![1, 2, 2]).unwrap(),
            values.to_vec(),
        )
        .unwrap();
        let bindings = HashMap::from([("x".into(), input)]);
        let prepared_output = prepared.run(&bindings).unwrap();
        assert_eq!(
            prepared_output[0].to_vec::<f32>().unwrap(),
            imported.run(&bindings).unwrap()["y"]
                .to_vec::<f32>()
                .unwrap()
        );
    }
}

#[test]
fn test_imported_transpose_reshape_materializes_in_logical_order() {
    let mut source = model("Transpose", 13);
    let graph = source.graph.as_mut().unwrap();
    graph.input = vec![value_info("x", &[2, 3])];
    graph.output = vec![value_info("y", &[6])];
    graph.node[0].output = vec!["transposed".into()];
    graph.node[0].attribute = vec![AttributeProto {
        name: "perm".into(),
        ints: vec![1, 0],
        r#type: attribute_proto::AttributeType::Ints as i32,
        ..Default::default()
    }];
    graph.initializer = vec![TensorProto {
        name: "shape".into(),
        dims: vec![1],
        data_type: tensor_proto::DataType::Int64 as i32,
        int64_data: vec![6],
        ..Default::default()
    }];
    graph.node.push(NodeProto {
        op_type: "Reshape".into(),
        input: vec!["transposed".into(), "shape".into()],
        output: vec!["y".into()],
        ..Default::default()
    });
    let imported = import(&source).unwrap();
    let check = |device| {
        let mut executor = imported.prepare(GraphPrepareOptions::new(device)).unwrap();
        assert_eq!(
            executor.analysis().signature().outputs()[0].name(),
            Some("y")
        );
        let input = Tensor::from_vec(
            device,
            Shape::create([2, 3]).unwrap(),
            vec![0_f32, 1., 2., 3., 4., 5.],
        )
        .unwrap();
        let output = executor.run(&HashMap::from([("x".into(), input)])).unwrap();
        assert_eq!(output[0].to_vec::<f32>().unwrap(), [0., 3., 1., 4., 2., 5.]);
    };
    check(Device::Cpu);
    #[cfg(feature = "cuda")]
    check(Device::Cuda(0));
}

#[test]
fn test_scalar_comparison_and_vector_dot_import_have_rank_zero_results() {
    for (operator, dimensions, output_type) in [
        ("Equal", vec![], tensor_proto::DataType::Bool),
        ("MatMul", vec![3], tensor_proto::DataType::Float),
    ] {
        let mut source = model(operator, 13);
        let graph = source.graph.as_mut().unwrap();
        graph.node[0].input = vec!["x".into(), "z".into()];
        graph.input = vec![value_info("x", &dimensions), value_info("z", &dimensions)];
        graph.output = vec![typed_value_info("y", &[], output_type)];
        let imported = import(&source).unwrap();
        let input = Tensor::full(Device::Cpu, Shape::create(dimensions).unwrap(), 2.0_f32).unwrap();
        let output = imported
            .run(&HashMap::from([
                ("x".into(), input.clone()),
                ("z".into(), input),
            ]))
            .unwrap();
        assert_eq!(output["y"].shape(), &Shape::scalar());
        if operator == "Equal" {
            assert_eq!(output["y"].data_type(), DataType::Bool);
            assert_eq!(output["y"].to_vec::<bool>().unwrap(), [true]);
        } else {
            assert_eq!(output["y"].to_vec::<f32>().unwrap(), [12.0]);
        }
    }
}

#[test]
fn test_old_softmax_and_unknown_future_schemas_fail_before_execution() {
    for (operator, version) in [
        ("Softmax", 11),
        ("LogSoftmax", 1),
        ("Softmax", 26),
        ("Attention", 24),
        ("AveragePool", 21),
        ("QuantizeLinear", 21),
    ] {
        assert!(
            matches!(import(&model(operator, version)), Err(Error::UnsupportedOperator { version: actual, .. }) if actual == version)
        );
    }
}

#[test]
fn test_modern_softmax_import_runs_without_cuda() {
    let model = import(&model("Softmax", 13)).unwrap();
    let input = Tensor::from_vec(
        Device::Cpu,
        Shape::create([1, 2, 2]).unwrap(),
        vec![0.0_f32, 1.0, 2.0, 3.0],
    )
    .unwrap();
    let outputs = model
        .run(&HashMap::from([("x".to_string(), input)]))
        .unwrap();
    let expected = [0.268_941_43_f32, 0.731_058_6, 0.268_941_43, 0.731_058_6];
    for (actual, expected) in outputs["y"]
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .zip(expected)
    {
        assert!((actual - expected).abs() < 1e-6);
    }
}

#[test]
fn test_missing_duplicate_and_non_positive_opsets_are_rejected() {
    let mut missing = model("Identity", 13);
    missing.opset_import.clear();
    assert!(matches!(import(&missing), Err(Error::InvalidModel { .. })));
    let mut duplicate = model("Identity", 13);
    duplicate.opset_import.push(OperatorSetIdProto {
        domain: String::new(),
        version: 13,
    });
    assert!(matches!(
        import(&duplicate),
        Err(Error::InvalidModel { .. })
    ));
    assert!(matches!(
        import(&model("Identity", 0)),
        Err(Error::InvalidModel { .. })
    ));
}

#[test]
fn test_exact_gelu_is_not_silently_lowered_to_the_tanh_approximation() {
    let imported = import(&model("Gelu", 20)).unwrap();
    let input = Tensor::from_vec(
        Device::Cpu,
        Shape::create([1, 2, 2]).unwrap(),
        vec![-2_f32, -1., 1., 2.],
    )
    .unwrap();
    let expected = input.gelu_exact().unwrap().to_vec::<f32>().unwrap();
    let approximate = input.gelu().unwrap().to_vec::<f32>().unwrap();
    assert_ne!(expected, approximate);
    let output = imported.run(&HashMap::from([("x".into(), input)])).unwrap();
    assert_eq!(output["y"].to_vec::<f32>().unwrap(), expected);
}

#[test]
fn test_swish_with_non_unit_alpha_is_rejected() {
    let mut proto = model("Swish", 24);
    proto.graph.as_mut().unwrap().node[0]
        .attribute
        .push(AttributeProto {
            name: "alpha".into(),
            f: 2.0,
            r#type: attribute_proto::AttributeType::Float as i32,
            ..Default::default()
        });
    assert!(matches!(
        import(&proto),
        Err(Error::UnsupportedOperator { .. })
    ));
}

#[test]
fn test_pooling_uses_onnx_unit_strides_when_omitted() {
    for (operator, expected) in [
        ("MaxPool", [4.0_f32, 5.0, 7.0, 8.0]),
        ("AveragePool", [2.0, 3.0, 5.0, 6.0]),
    ] {
        let mut proto = model(operator, 13);
        let graph = proto.graph.as_mut().unwrap();
        graph.input = vec![value_info("x", &[1, 1, 3, 3])];
        graph.output = vec![value_info("y", &[1, 1, 2, 2])];
        graph.node[0].attribute.push(AttributeProto {
            name: "kernel_shape".into(),
            ints: vec![2, 2],
            r#type: attribute_proto::AttributeType::Ints as i32,
            ..Default::default()
        });
        let model = import(&proto).unwrap();
        let input = Tensor::from_vec(
            Device::Cpu,
            Shape::create([1, 1, 3, 3]).unwrap(),
            (0..9).map(|x| x as f32).collect(),
        )
        .unwrap();
        let outputs = model
            .run(&HashMap::from([("x".to_string(), input)]))
            .unwrap();
        assert_eq!(outputs["y"].to_vec::<f32>().unwrap(), expected);
    }
}

#[test]
fn test_import_defers_cuda_placement_and_separates_initializer_state() {
    let mut source = model("Add", 13);
    let graph = source.graph.as_mut().unwrap();
    graph.node[0].input.push("weight".into());
    graph.initializer.push(TensorProto {
        name: "weight".into(),
        dims: vec![1, 2, 2],
        data_type: tensor_proto::DataType::Float as i32,
        float_data: vec![1., 2., 3., 4.],
        ..Default::default()
    });
    // This device is deliberately unavailable.
    // Import must not inspect it.
    let imported = OnnxModel::from_reader(
        source.encode_to_vec().as_slice(),
        OnnxOptions::new().with_device(Device::Cuda(i32::MAX)),
    )
    .unwrap();
    assert_eq!(
        imported.initializers().tensor("weight").unwrap().device(),
        Device::Cpu
    );
    assert_eq!(imported.program().state_schema().count(), 1);
    let mut executor = imported
        .prepare(GraphPrepareOptions::new(Device::Cpu))
        .unwrap();
    let input = Tensor::full(Device::Cpu, Shape::create([1, 2, 2]).unwrap(), 2_f32).unwrap();
    let output = executor.run(&HashMap::from([("x".into(), input)])).unwrap();
    assert_eq!(output[0].to_vec::<f32>().unwrap(), vec![3., 4., 5., 6.]);
}

#[test]
fn test_initializer_defaults_remain_overridable_in_the_common_executor() {
    let mut source = model("Add", 13);
    let graph = source.graph.as_mut().unwrap();
    graph.node[0].input.push("weight".into());
    graph.input.push(value_info("weight", &[1, 2, 2]));
    graph.initializer.push(TensorProto {
        name: "weight".into(),
        dims: vec![1, 2, 2],
        data_type: tensor_proto::DataType::Float as i32,
        float_data: vec![1., 2., 3., 4.],
        ..Default::default()
    });
    let imported = import(&source).unwrap();
    let mut executor = imported
        .prepare(GraphPrepareOptions::new(Device::Cpu))
        .unwrap();
    let tensor = |v| Tensor::full(Device::Cpu, Shape::create([1, 2, 2]).unwrap(), v).unwrap();
    let mut bindings = HashMap::from([("x".into(), tensor(2_f32))]);
    assert_eq!(
        executor.run(&bindings).unwrap()[0].to_vec::<f32>().unwrap(),
        vec![3., 4., 5., 6.]
    );
    bindings.insert("weight".into(), tensor(7_f32));
    assert_eq!(
        executor.run(&bindings).unwrap()[0].to_vec::<f32>().unwrap(),
        vec![9.; 4]
    );
    assert_eq!(
        imported.run_ordered(&bindings).unwrap()[0]
            .to_vec::<f32>()
            .unwrap(),
        vec![9.; 4]
    );
}

#[test]
fn test_imported_shape_specializations_reject_overrides_and_incompatible_state_rebinding() {
    let mut source = model("Reshape", 14);
    let graph = source.graph.as_mut().unwrap();
    graph.node[0].input.push("shape".into());
    graph.input.push(typed_value_info(
        "shape",
        &[2],
        tensor_proto::DataType::Int64,
    ));
    graph.output = vec![value_info("y", &[2, 2])];
    graph.initializer.push(TensorProto {
        name: "shape".into(),
        dims: vec![2],
        data_type: tensor_proto::DataType::Int64 as i32,
        int64_data: vec![2, 2],
        ..Default::default()
    });
    let imported = import(&source).unwrap();
    let mut executor = imported
        .prepare(GraphPrepareOptions::new(Device::Cpu))
        .unwrap();
    let input = Tensor::full(Device::Cpu, Shape::create([1, 2, 2]).unwrap(), 2_f32).unwrap();
    let changed_shape =
        Tensor::from_vec(Device::Cpu, Shape::create([2]).unwrap(), vec![1_i64, 4]).unwrap();
    let mut bindings = HashMap::from([("x".into(), input)]);
    assert_eq!(
        executor.run(&bindings).unwrap()[0].shape().dimensions(),
        &[2, 2]
    );
    bindings.insert("shape".into(), changed_shape.clone());
    assert!(executor.run(&bindings).is_err());
    let changed_state = imported
        .initializers()
        .replace(&HashMap::from([("shape".into(), changed_shape)]))
        .unwrap();
    assert!(
        imported
            .program()
            .prepare(&changed_state, GraphPrepareOptions::new(Device::Cpu))
            .is_err()
    );
    bindings.remove("shape");
    assert_eq!(
        executor.run(&bindings).unwrap()[0].shape().dimensions(),
        &[2, 2]
    );
}

fn integer_attribute(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.into(),
        i: value,
        r#type: attribute_proto::AttributeType::Int as i32,
        ..Default::default()
    }
}

#[test]
fn test_ir_versions_and_versioned_attributes_are_checked_before_lowering() {
    for version in [0, 6, 11, i64::MAX] {
        let mut source = model("Identity", 13);
        source.ir_version = version;
        assert!(matches!(import(&source), Err(Error::InvalidModel { .. })));
    }
    for (operator, version, attribute) in [
        ("Shape", 13, integer_attribute("start", 1)),
        ("Reshape", 13, integer_attribute("allowzero", 1)),
        (
            "ReduceSum",
            13,
            AttributeProto {
                name: "axes".into(),
                ints: vec![1],
                r#type: attribute_proto::AttributeType::Ints as i32,
                ..Default::default()
            },
        ),
        (
            "ReduceMean",
            18,
            AttributeProto {
                name: "axes".into(),
                ints: vec![1],
                r#type: attribute_proto::AttributeType::Ints as i32,
                ..Default::default()
            },
        ),
        (
            "Squeeze",
            13,
            AttributeProto {
                name: "axes".into(),
                ints: vec![0],
                r#type: attribute_proto::AttributeType::Ints as i32,
                ..Default::default()
            },
        ),
        (
            "Clip",
            13,
            AttributeProto {
                name: "min".into(),
                f: 0.,
                r#type: attribute_proto::AttributeType::Float as i32,
                ..Default::default()
            },
        ),
    ] {
        let mut source = model(operator, version);
        if operator == "Reshape" {
            source.graph.as_mut().unwrap().node[0]
                .input
                .push("shape".into());
        }
        source.graph.as_mut().unwrap().node[0]
            .attribute
            .push(attribute);
        assert!(
            matches!(import(&source), Err(Error::InvalidAttribute { .. })),
            "{operator}"
        );
    }
}

#[test]
fn test_absent_and_empty_reduction_axes_preserve_composite_steps() {
    for (operator, expected) in [
        ("ReduceMean", vec![-2_f32, 3., -4., 5.]),
        ("ReduceSum", vec![-2., 3., -4., 5.]),
        ("ReduceL1", vec![2., 3., 4., 5.]),
        ("ReduceL2", vec![2., 3., 4., 5.]),
        ("ReduceSumSquare", vec![4., 9., 16., 25.]),
        (
            "ReduceLogSum",
            vec![f32::NAN, 3_f32.ln(), f32::NAN, 5_f32.ln()],
        ),
        ("ReduceLogSumExp", vec![-2., 3., -4., 5.]),
    ] {
        for explicit_empty in [false, true] {
            let mut source = model(operator, 18);
            let graph = source.graph.as_mut().unwrap();
            graph.node[0]
                .attribute
                .push(integer_attribute("noop_with_empty_axes", 1));
            if explicit_empty {
                graph.node[0].input.push("axes".into());
                graph.initializer.push(TensorProto {
                    name: "axes".into(),
                    dims: vec![0],
                    data_type: tensor_proto::DataType::Int64 as i32,
                    ..Default::default()
                });
            }
            let imported = import(&source).unwrap();
            let input = Tensor::from_vec(
                Device::Cpu,
                Shape::create([1, 2, 2]).unwrap(),
                vec![-2_f32, 3., -4., 5.],
            )
            .unwrap();
            let output = imported
                .run_ordered(&HashMap::from([("x".into(), input)]))
                .unwrap()[0]
                .to_vec::<f32>()
                .unwrap();
            for (actual, expected) in output.iter().zip(&expected) {
                assert!(
                    (actual - expected).abs() < 1e-5 || actual.is_nan() && expected.is_nan(),
                    "{operator}: {actual}, {expected}"
                );
            }
        }
    }
}

#[test]
fn test_omitted_optional_outputs_work_and_required_slots_cannot_be_empty() {
    let mut source = model("MaxPool", 13);
    let graph = source.graph.as_mut().unwrap();
    graph.node[0].output.push(String::new());
    graph.node[0].attribute.push(AttributeProto {
        name: "kernel_shape".into(),
        ints: vec![1, 1],
        r#type: attribute_proto::AttributeType::Ints as i32,
        ..Default::default()
    });
    graph.input = vec![value_info("x", &[1, 1, 2, 2])];
    graph.output = vec![value_info("y", &[1, 1, 2, 2])];
    assert!(import(&source).is_ok());
    for empty_input in [true, false] {
        let mut source = model("Identity", 13);
        let node = &mut source.graph.as_mut().unwrap().node[0];
        if empty_input {
            node.input[0].clear();
        } else {
            node.output[0].clear();
        }
        assert!(matches!(
            import(&source),
            Err(Error::UnsupportedOperator { .. })
        ));
    }
}

#[test]
fn test_contradictory_shapes_and_value_namespace_collisions_are_rejected() {
    let mut source = model("Identity", 13);
    source.graph.as_mut().unwrap().output[0] = value_info("y", &[4]);
    assert!(matches!(
        import(&source),
        Err(Error::UnsupportedShape { .. })
    ));
    let mut source = model("Identity", 13);
    source.graph.as_mut().unwrap().input[0] = value_info("x", &[-1]);
    assert!(import(&source).is_err());
    let mut source = model("Identity", 13);
    let graph = source.graph.as_mut().unwrap();
    graph.node[0].output[0] = "x".into();
    assert!(matches!(import(&source), Err(Error::DuplicateValue(_))));
    let mut source = model("Identity", 13);
    let graph = source.graph.as_mut().unwrap();
    graph.input.push(graph.input[0].clone());
    assert!(matches!(import(&source), Err(Error::DuplicateValue(_))));
    let mut source = model("Identity", 13);
    let graph = source.graph.as_mut().unwrap();
    graph.output.push(graph.output[0].clone());
    assert!(matches!(import(&source), Err(Error::DuplicateValue(_))));
}

fn assert_provider_limits_are_reported_with_onnx_provenance(device: Device) {
    let mut source = model("MaxPool", 13);
    let graph = source.graph.as_mut().unwrap();
    graph.node[0].name = "activation".into();
    graph.node[0].attribute.push(AttributeProto {
        name: "kernel_shape".into(),
        ints: vec![1, 1],
        r#type: attribute_proto::AttributeType::Ints as i32,
        ..Default::default()
    });
    // Neither backend pools integers, which ONNX `MaxPool` accepts.
    graph.input = vec![typed_value_info(
        "x",
        &[1, 1, 2, 2],
        tensor_proto::DataType::Int8,
    )];
    graph.output = vec![typed_value_info(
        "y",
        &[1, 1, 2, 2],
        tensor_proto::DataType::Int8,
    )];
    let imported = import(&source).unwrap();
    // Logical analysis must not reject a dtype based on provider limits.
    imported.program().fingerprint().unwrap();
    let error = imported
        .prepare(GraphPrepareOptions::new(device))
        .unwrap_err();
    let SingeError::GraphSource { location, source } = error else {
        panic!("missing source provenance: {error}")
    };
    assert!(
        location.contains("`activation`")
            && location.contains("domain ``")
            && location.contains("MaxPool opset 13")
            && location.contains("x: I8 [1, 1, 2, 2]")
    );
    assert!(matches!(*source, SingeError::Unsupported { .. }));
}

#[test]
fn test_provider_limits_are_reported_during_preparation_with_onnx_provenance() {
    assert_provider_limits_are_reported_with_onnx_provenance(Device::Cpu);
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_provider_limits_are_reported_during_preparation_with_onnx_provenance() {
    assert_provider_limits_are_reported_with_onnx_provenance(Device::Cuda(0));
}

#[test]
fn test_onnx_types_do_not_inherit_framework_binary_promotion() {
    let mut source = model("Add", 13);
    let graph = source.graph.as_mut().unwrap();
    graph.node[0].input.push("integer".into());
    graph.input.push(typed_value_info(
        "integer",
        &[1, 2, 2],
        tensor_proto::DataType::Int64,
    ));
    assert!(matches!(
        import(&source),
        Err(Error::UnsupportedOperator { .. })
    ));
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_graphs_prepare_onnx_topk_with_both_results_as_its_decomposition() {
    let mut source = model("TopK", 13);
    let graph = source.graph.as_mut().unwrap();
    graph.node[0].name = "choose".into();
    graph.node[0].input.push("k".into());
    graph.node[0].output.push("indices".into());
    graph.output = vec![
        value_info("y", &[1, 2, 1]),
        typed_value_info("indices", &[1, 2, 1], tensor_proto::DataType::Int64),
    ];
    graph.initializer.push(TensorProto {
        name: "k".into(),
        data_type: tensor_proto::DataType::Int64 as i32,
        dims: vec![1],
        int64_data: vec![1],
        ..Default::default()
    });
    let imported = import(&source).unwrap();
    imported
        .prepare(GraphPrepareOptions::new(Device::Cuda(0)))
        .unwrap();
}

#[test]
fn test_conflicting_shape_overrides_and_output_symbols_are_rejected() {
    let source = model("Identity", 13);
    assert!(matches!(
        OnnxModel::from_reader(
            source.encode_to_vec().as_slice(),
            OnnxOptions::new().with_input_shape("x", vec![99, 2, 2])
        ),
        Err(Error::UnsupportedShape { .. })
    ));
    let mut source = model("Transpose", 13);
    let graph = source.graph.as_mut().unwrap();
    graph.input = vec![value_info("x", &[2, 3])];
    graph.output = vec![value_info("y", &[3, 2])];
    for info in [&mut graph.input[0], &mut graph.output[0]] {
        let Some(type_proto::Value::TensorType(tensor)) = &mut info.r#type.as_mut().unwrap().value
        else {
            unreachable!()
        };
        tensor.shape.as_mut().unwrap().dim[0].value =
            Some(tensor_shape_proto::dimension::Value::DimParam("n".into()));
    }
    assert!(matches!(
        OnnxModel::from_reader(
            source.encode_to_vec().as_slice(),
            OnnxOptions::new().with_symbolic_dimension("n", 2)
        ),
        Err(Error::UnsupportedShape { .. })
    ));
    let mut source = model("Identity", 13);
    source.graph.as_mut().unwrap().node[0].overload = "custom".into();
    assert!(matches!(import(&source), Err(Error::InvalidModel { .. })));
}

#[test]
fn test_reimporting_the_same_export_preserves_logical_identity_and_signature_order() {
    let bytes = include_bytes!("../references/layers/encoder.onnx");
    let first = OnnxModel::from_reader(bytes.as_slice(), OnnxOptions::new()).unwrap();
    let fingerprint = first.program().fingerprint().unwrap();
    let names: Vec<_> = first
        .program()
        .signature()
        .inputs()
        .iter()
        .map(ProgramInput::name)
        .collect();
    for _ in 0..5 {
        let other = OnnxModel::from_reader(bytes.as_slice(), OnnxOptions::new()).unwrap();
        assert_eq!(other.program().fingerprint().unwrap(), fingerprint);
        assert_eq!(
            other
                .program()
                .signature()
                .inputs()
                .iter()
                .map(ProgramInput::name)
                .collect::<Vec<_>>(),
            names
        );
    }
}

fn fp16_constant_relu_model() -> ModelProto {
    let mut source = model("Relu", 13);
    let graph = source.graph.as_mut().unwrap();
    graph.input = vec![typed_value_info(
        "x",
        &[1, 2, 2],
        tensor_proto::DataType::Float16,
    )];
    graph.output = vec![typed_value_info(
        "y",
        &[1, 2, 2],
        tensor_proto::DataType::Float16,
    )];
    graph.node = vec![
        NodeProto {
            op_type: "Constant".into(),
            output: vec!["constant".into()],
            attribute: vec![AttributeProto {
                name: "value".into(),
                r#type: attribute_proto::AttributeType::Tensor as i32,
                t: Some(TensorProto {
                    data_type: tensor_proto::DataType::Float16 as i32,
                    dims: vec![1, 2, 2],
                    // FP16 encodings of [-1, 2, -3, 4].
                    int32_data: vec![0xbc00, 0x4000, 0xc200, 0x4400],
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        },
        NodeProto {
            op_type: "Relu".into(),
            input: vec!["constant".into()],
            output: vec!["activated".into()],
            ..Default::default()
        },
        NodeProto {
            op_type: "Add".into(),
            input: vec!["activated".into(), "x".into()],
            output: vec!["y".into()],
            ..Default::default()
        },
    ];
    source
}

#[test]
fn test_fp16_constant_folding_keeps_logical_fingerprints_and_runs_on_the_cpu() {
    let imported = import(&fp16_constant_relu_model()).unwrap();
    imported.program().fingerprint().unwrap();
    // The CPU computes `F16` through the kernel IR interpreter.
    assert!(
        imported
            .prepare(GraphPrepareOptions::new(Device::Cpu))
            .is_ok()
    );
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_executes_fp16_constant_operations_without_cpu_kernel_support() {
    use half::f16;
    use singe::graph::GraphExecutionMode;

    let imported = import(&fp16_constant_relu_model()).unwrap();
    for mode in [GraphExecutionMode::Eager, GraphExecutionMode::Compiled] {
        let mut executor = imported
            .prepare(GraphPrepareOptions::new(Device::Cuda(0)).with_execution_mode(mode))
            .unwrap();
        let input = Tensor::from_vec(
            Device::Cuda(0),
            Shape::create([1, 2, 2]).unwrap(),
            [-2., 3., -4., 5.].map(f16::from_f32).to_vec(),
        )
        .unwrap();
        let output = executor.run(&HashMap::from([("x".into(), input)])).unwrap();
        assert_eq!(
            output[0].to_vec::<f16>().unwrap(),
            [-2., 5., -4., 9.].map(f16::from_f32),
            "{mode:?}"
        );
    }
}

fn intermediate_metadata_model(info: ValueInfoProto) -> ModelProto {
    let mut source = model("Relu", 13);
    let graph = source.graph.as_mut().unwrap();
    graph.node[0].output = vec!["hidden".into()];
    graph.node.push(NodeProto {
        op_type: "Identity".into(),
        input: vec!["hidden".into()],
        output: vec!["y".into()],
        ..Default::default()
    });
    graph.value_info.push(info);
    source
}

#[test]
fn test_untyped_intermediate_metadata_preserves_inferred_values() {
    let imported = import(&intermediate_metadata_model(ValueInfoProto {
        name: "hidden".into(),
        ..Default::default()
    }))
    .unwrap();
    let mut executor = imported
        .prepare(GraphPrepareOptions::new(Device::Cpu))
        .unwrap();
    let input = Tensor::from_vec(
        Device::Cpu,
        Shape::create([1, 2, 2]).unwrap(),
        vec![-1_f32, 2., -3., 4.],
    )
    .unwrap();
    let inputs = HashMap::from([("x".into(), input)]);
    for output in [
        imported.run_ordered(&inputs).unwrap(),
        executor.run(&inputs).unwrap(),
    ] {
        assert_eq!(output[0].to_vec::<f32>().unwrap(), [0., 2., 0., 4.]);
    }
}

#[test]
fn test_supplied_intermediate_metadata_must_match_inferred_values() {
    for info in [
        typed_value_info("hidden", &[1, 2, 2], tensor_proto::DataType::Int64),
        value_info("hidden", &[4]),
        value_info("hidden", &[1, 2, 3]),
    ] {
        assert!(matches!(
            import(&intermediate_metadata_model(info)),
            Err(Error::UnsupportedShape { .. })
        ));
    }
}

#[test]
fn test_graph_inputs_and_outputs_still_require_types() {
    for missing_input_type in [true, false] {
        let mut source = model("Relu", 13);
        let graph = source.graph.as_mut().unwrap();
        if missing_input_type {
            graph.input[0].r#type = None;
        } else {
            graph.output[0].r#type = None;
        }
        assert!(matches!(
            import(&source),
            Err(Error::UnsupportedShape { reason, .. }) if reason == "missing value type"
        ));
    }
}

#[test]
fn test_cast_19_saturation_is_irrelevant_to_supported_types_and_new_encodings_stay_rejected() {
    use half::f16;
    use singe_onnx::TensorDataType;

    fn cast(version: i64, dtype: tensor_proto::DataType, saturate: i64) -> ModelProto {
        let mut source = model("Cast", version);
        let graph = source.graph.as_mut().unwrap();
        graph.output = vec![typed_value_info("y", &[1, 2, 2], dtype)];
        graph.node[0].attribute = vec![
            AttributeProto {
                name: "to".into(),
                r#type: attribute_proto::AttributeType::Int as i32,
                i: dtype as i64,
                ..Default::default()
            },
            AttributeProto {
                name: "saturate".into(),
                r#type: attribute_proto::AttributeType::Int as i32,
                i: saturate,
                ..Default::default()
            },
        ];
        source
    }
    for (version, saturation) in [(19, 0), (19, 1), (20, 0), (20, 1)] {
        let imported = import(&cast(version, tensor_proto::DataType::Float16, saturation)).unwrap();
        let input = Tensor::from_vec(
            Device::Cpu,
            Shape::create([1, 2, 2]).unwrap(),
            vec![-1.5_f32, 0., 0.5, 2.],
        )
        .unwrap();
        assert_eq!(
            imported
                .run_ordered(&HashMap::from([("x".into(), input)]))
                .unwrap()[0]
                .to_vec::<f16>()
                .unwrap(),
            [-1.5_f32, 0., 0.5, 2.].map(f16::from_f32)
        );
    }
    assert!(matches!(
        import(&cast(20, tensor_proto::DataType::Float16, 2)),
        Err(Error::InvalidAttribute { .. })
    ));
    assert!(matches!(
        import(&cast(
            20,
            tensor_proto::DataType::try_from(TensorDataType::Float8e4m3fn as i32).unwrap(),
            0
        )),
        Err(Error::UnsupportedOperator { .. })
    ));
    assert!(matches!(
        import(&cast(21, tensor_proto::DataType::Float16, 1)),
        Err(Error::UnsupportedOperator { .. })
    ));
}
