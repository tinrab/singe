#![cfg(feature = "onnx")]

use std::collections::HashMap;

use singe::{
    DataType, Device, OnnxModel, OnnxOptions, Shape, Tensor,
    backend::registry::{Operation, OperationAttributes},
    dispatch::{CustomOperation, DispatchKey, Kernel, call_one},
    graph::{CaptureBindings, CaptureOptions, Graph, GraphInputSpec, GraphProgram},
    nn::StateBuilder,
    onnx::{self, Error},
    random::Generator,
    value::ValueSpec,
};
use singe_onnx::{
    decode_proto, encode_proto,
    proto::onnx::{AttributeProto, NodeProto, StringStringEntryProto, attribute_proto},
};

// The program of one operation on one input, with one named output.
fn one_operation(
    spec: impl Into<GraphInputSpec>,
    operation: Operation,
    attributes: OperationAttributes,
) -> GraphProgram {
    Graph::build([("x", spec)], CaptureBindings::new(), |inputs| {
        Ok(vec![call_one(operation, attributes, &[&inputs[0]])?])
    })
    .unwrap()
    .with_output_names(["y"])
    .unwrap()
    .program()
    .unwrap()
}

fn tensor(values: Vec<f32>, shape: &[i64]) -> Tensor {
    Tensor::from_vec(Device::Cpu, Shape::create(shape.to_vec()).unwrap(), values).unwrap()
}

#[test]
fn test_named_state_publications_strides_ties_and_output_projections_survive_as_values() {
    let weight = tensor(vec![1., 2., 3., 4.], &[2, 2])
        .transpose(0, 1)
        .unwrap();
    let mut state = StateBuilder::new();
    let key = state.parameter("weight", weight).unwrap();
    state.tie("tied.weight", &key).unwrap();
    let state = state.finish().unwrap();
    let meta = state.to(Device::Meta, None).unwrap();
    let graph = Graph::build(
        [
            (
                "singe_0",
                ValueSpec::new(DataType::F32, Shape::create([2, 2]).unwrap()),
            ),
            ("unused", ValueSpec::new(DataType::I64, Shape::scalar())),
        ],
        meta.capture_bindings().unwrap(),
        |inputs| {
            let output = call_one(
                Operation::Add,
                OperationAttributes::None,
                &[&inputs[0], meta.named_tensor("weight").unwrap()],
            )?;
            Ok(vec![output.clone(), output])
        },
    )
    .unwrap()
    .with_output_names(["result", "again"])
    .unwrap();
    let program = state.program(&graph).unwrap();
    let identity = program.fingerprint().unwrap();
    let original = onnx::export(&program, state.snapshot()).unwrap();
    let changed = state
        .snapshot()
        .replace(&HashMap::from([(
            "weight".into(),
            tensor(vec![10., 20., 30., 40.], &[2, 2]),
        )]))
        .unwrap();
    let changed_bytes = onnx::export(&program, &changed).unwrap();
    assert_ne!(original, changed_bytes);
    assert_eq!(identity, program.fingerprint().unwrap());
    assert_eq!(original, onnx::export(&program, state.snapshot()).unwrap());
    for (bytes, expected) in [
        (&original, vec![2., 4., 3., 5.]),
        (&changed_bytes, vec![11., 21., 31., 41.]),
    ] {
        let imported = OnnxModel::from_reader(bytes.as_slice(), OnnxOptions::new()).unwrap();
        assert_eq!(
            imported
                .outputs()
                .iter()
                .map(|output| output.name.as_str())
                .collect::<Vec<_>>(),
            ["result", "again"]
        );
        assert_eq!(
            imported
                .initializers()
                .tensor("weight")
                .unwrap()
                .to_bytes()
                .unwrap(),
            imported
                .initializers()
                .tensor("tied.weight")
                .unwrap()
                .to_bytes()
                .unwrap()
        );
        let inputs = HashMap::from([
            ("singe_0".into(), tensor(vec![1.; 4], &[2, 2])),
            (
                "unused".into(),
                Tensor::from_vec(Device::Cpu, Shape::scalar(), vec![7_i64]).unwrap(),
            ),
        ]);
        let outputs = imported.run(&inputs).unwrap();
        assert_eq!(outputs["result"].to_vec::<f32>().unwrap(), expected);
        assert_eq!(
            outputs["again"].to_bytes().unwrap(),
            outputs["result"].to_bytes().unwrap()
        );
        assert!(
            imported
                .run(&HashMap::from([(
                    "singe_0".into(),
                    inputs["singe_0"].clone()
                )]))
                .is_err()
        );
    }
}

#[test]
fn test_overridable_defaults_remain_graph_inputs_and_initializers() {
    let mut graph = Graph::new();
    let input = graph.add_input(
        "fallback",
        DataType::F32,
        Shape::create([2]).unwrap(),
        Device::Cpu,
    );
    let value = tensor(vec![1., 2.], &[2]);
    graph.add_initializer("fallback", value.clone()).unwrap();
    graph
        .set_named_output_projections(vec![("result".into(), input, 0)])
        .unwrap();
    let program = graph.program().unwrap();
    let state = program
        .bind_state(&HashMap::from([("fallback".into(), value)]))
        .unwrap();
    let bytes = onnx::export(&program, &state).unwrap();
    let proto = decode_proto(&bytes).unwrap().graph.unwrap();
    assert_eq!(proto.input[0].name, "fallback");
    assert_eq!(proto.initializer[0].name, "fallback");
    let imported = OnnxModel::from_reader(bytes.as_slice(), OnnxOptions::new()).unwrap();
    assert_eq!(
        imported.run(&HashMap::new()).unwrap()["result"]
            .to_vec::<f32>()
            .unwrap(),
        [1., 2.]
    );
    assert_eq!(
        imported
            .run(&HashMap::from([(
                "fallback".into(),
                tensor(vec![3., 4.], &[2])
            )]))
            .unwrap()["result"]
            .to_vec::<f32>()
            .unwrap(),
        [3., 4.]
    );
}

#[test]
fn test_export_rejects_unsupported_operations_mixed_types_and_foreign_snapshots() {
    let spec = ValueSpec::new(DataType::F32, Shape::create([2]).unwrap());
    let program = one_operation(spec.clone(), Operation::Square, OperationAttributes::None);
    let state = program.bind_state(&HashMap::new()).unwrap();
    assert!(matches!(
        onnx::export(&program, &state),
        Err(Error::UnsupportedExport { .. })
    ));
    let mixed = Graph::build(
        [
            ("x", spec),
            (
                "z",
                ValueSpec::new(DataType::F64, Shape::create([2]).unwrap()),
            ),
        ],
        CaptureBindings::new(),
        |inputs| {
            Ok(vec![call_one(
                Operation::Add,
                OperationAttributes::None,
                &[&inputs[0], &inputs[1]],
            )?])
        },
    )
    .unwrap()
    .program()
    .unwrap();
    assert!(matches!(
        onnx::export(&mixed, &mixed.bind_state(&HashMap::new()).unwrap()),
        Err(Error::UnsupportedExport { .. })
    ));
    assert!(matches!(onnx::export(&mixed, &state), Err(Error::Singe(_))));
}

#[test]
fn test_export_rejects_custom_operations_by_name() {
    let custom = CustomOperation::register(
        "test.export_identity",
        1,
        1,
        &[DataType::F32],
        |inputs, _| Ok(vec![inputs[0].clone()]),
    )
    .unwrap()
    .register_kernel(
        DispatchKey::Cpu,
        Kernel::single(|call| call.input(0).cloned()),
    )
    .unwrap();
    let program = one_operation(
        ValueSpec::new(DataType::F32, Shape::create([2]).unwrap()),
        Operation::Custom(custom),
        OperationAttributes::Custom(Vec::new()),
    );
    let state = program.bind_state(&HashMap::new()).unwrap();
    let error = onnx::export(&program, &state).unwrap_err();
    assert!(
        matches!(&error, Error::UnsupportedExport { .. })
            && error.to_string().contains("test.export_identity"),
        "unexpected error {error}"
    );
}

#[test]
fn test_gelu_formulas_are_explicit_and_roundtrip_independently() {
    for operation in [Operation::Gelu, Operation::GeluExact] {
        let program = one_operation(
            ValueSpec::new(DataType::F32, Shape::create([2]).unwrap()),
            operation,
            OperationAttributes::None,
        );
        let bytes = onnx::export(&program, &program.bind_state(&HashMap::new()).unwrap()).unwrap();
        let proto = decode_proto(&bytes).unwrap();
        let gelu = proto
            .graph
            .as_ref()
            .unwrap()
            .node
            .iter()
            .find(|node| node.op_type == "Gelu")
            .unwrap();
        let approximation = &gelu
            .attribute
            .iter()
            .find(|attr| attr.name == "approximate")
            .unwrap()
            .s;
        assert_eq!(
            approximation,
            if operation == Operation::Gelu {
                b"tanh".as_slice()
            } else {
                b"none".as_slice()
            }
        );
        let imported = OnnxModel::from_reader(bytes.as_slice(), OnnxOptions::new()).unwrap();
        let input = tensor(vec![-2., 2.], &[2]);
        let expected = if operation == Operation::Gelu {
            input.gelu()
        } else {
            input.gelu_exact()
        }
        .unwrap();
        assert_eq!(
            imported.run(&HashMap::from([("x".into(), input)])).unwrap()["y"]
                .to_bytes()
                .unwrap(),
            expected.to_bytes().unwrap()
        );
    }
}

#[test]
fn test_shape_contract_preserves_ranges_when_callers_supply_weaker_bounds_and_rejects_new_versions()
{
    let program = Graph::build(
        [(
            "x",
            GraphInputSpec::new(ValueSpec::new(DataType::F32, Shape::create([2]).unwrap()))
                .with_dimension(0, "batch", Some(1), Some(3))
                .unwrap(),
        )],
        CaptureBindings::new(),
        |inputs| Ok(vec![inputs[0].clone()]),
    )
    .unwrap()
    .with_output_names(["x"])
    .unwrap()
    .program()
    .unwrap();
    let bytes = onnx::export(&program, &program.bind_state(&HashMap::new()).unwrap()).unwrap();
    let imported = OnnxModel::from_reader(
        bytes.as_slice(),
        OnnxOptions::new().with_symbolic_dimension_range("batch", 2, Some(0), Some(100)),
    )
    .unwrap();
    assert!(
        imported
            .run(&HashMap::from([("x".into(), tensor(vec![1.; 4], &[4]))]))
            .is_err()
    );
    assert_eq!(
        imported
            .run(&HashMap::from([("x".into(), tensor(vec![1.; 3], &[3]))]))
            .unwrap()["x"]
            .shape()
            .dimensions(),
        [3]
    );
    let mut proto = decode_proto(&bytes).unwrap();
    proto
        .metadata_props
        .iter_mut()
        .find(|entry| entry.key == "singe.shape_contract")
        .unwrap()
        .value = "2".into();
    assert!(matches!(
        OnnxModel::from_reader(encode_proto(&proto).as_slice(), OnnxOptions::new()),
        Err(Error::InvalidModel { .. })
    ));
}

#[test]
fn test_literal_payloads_preserve_integer_precision_nan_bits_signed_zero_and_strides() {
    let values = [
        Tensor::from_vec(
            Device::Cpu,
            Shape::create([2, 2]).unwrap(),
            vec![
                f64::from_bits(0x7ff8_0000_0000_1234),
                -0_f64,
                1.,
                f64::INFINITY,
            ],
        )
        .unwrap()
        .transpose(0, 1)
        .unwrap(),
        Tensor::from_vec(
            Device::Cpu,
            Shape::create([2, 2]).unwrap(),
            vec![(1_i64 << 60) + 3, i64::MIN, i64::MAX, -7],
        )
        .unwrap()
        .transpose(0, 1)
        .unwrap(),
    ];
    for value in values {
        let mut graph = Graph::new();
        let output = graph.add_constant(value.clone());
        graph
            .set_named_output_projections(vec![("literal".into(), output, 0)])
            .unwrap();
        let program = graph.program().unwrap();
        let bytes = onnx::export(&program, &program.bind_state(&HashMap::new()).unwrap()).unwrap();
        let imported = OnnxModel::from_reader(bytes.as_slice(), OnnxOptions::new()).unwrap();
        let output = imported.run(&HashMap::new()).unwrap();
        assert_eq!(output["literal"].data_type(), value.data_type());
        assert_eq!(
            output["literal"].to_bytes().unwrap(),
            value.to_bytes().unwrap()
        );
    }
}

#[test]
fn test_live_metadata_shape_values_reexport_as_runtime_integer_operations() {
    let mut source =
        decode_proto(include_bytes!("../references/shapes/shape_as_data.onnx")).unwrap();
    let graph = source.graph.as_mut().unwrap();
    let one = graph.initializer.pop().unwrap();
    // This program owns a literal shape operand.
    // The separate state-guard test retains the initializer form and checks that its guard cannot be dropped.
    graph.node.insert(
        0,
        NodeProto {
            op_type: "Constant".into(),
            output: vec!["one".into()],
            attribute: vec![AttributeProto {
                name: "value".into(),
                r#type: attribute_proto::AttributeType::Tensor as i32,
                t: Some(one),
                ..Default::default()
            }],
            ..Default::default()
        },
    );
    let original = OnnxModel::from_reader(
        encode_proto(&source).as_slice(),
        OnnxOptions::new()
            .with_symbolic_dimension_range("batch", 2, Some(1), Some(4))
            .with_symbolic_dimension_range("sequence", 3, Some(1), Some(8)),
    )
    .unwrap();
    let bytes = onnx::export(original.program(), original.initializers()).unwrap();
    let imported = OnnxModel::from_reader(bytes.as_slice(), OnnxOptions::new()).unwrap();
    for (batch, sequence) in [(2, 3), (1, 1), (4, 8)] {
        let input = Tensor::zeroed_on(
            Device::Cpu,
            DataType::F32,
            Shape::create([batch, sequence, 4]).unwrap(),
        )
        .unwrap();
        let output = imported
            .run_ordered(&HashMap::from([("x".into(), input)]))
            .unwrap();
        assert_eq!(
            output[0].to_vec::<i64>().unwrap(),
            [batch + 1, sequence + 1, 5]
        );
    }
}

#[test]
fn test_value_specialized_capture_and_imported_state_are_rejected() {
    let (captured, _) = Graph::capture_with_options(
        [("x", tensor(vec![2., 3.], &[2]))],
        CaptureBindings::new(),
        CaptureOptions::new().with_value_specialization(true),
        |inputs| {
            let coefficient = inputs[0].to_vec::<f32>()?[0];
            Ok(vec![inputs[0].mul_scalar(coefficient)?])
        },
    )
    .unwrap();
    let program = captured.program().unwrap();
    assert!(matches!(
        onnx::export(&program, &program.bind_state(&HashMap::new()).unwrap()),
        Err(Error::UnsupportedExport { .. })
    ));
    let original = OnnxModel::from_reader(
        include_bytes!("../references/shapes/shape_as_data.onnx").as_slice(),
        OnnxOptions::new()
            .with_symbolic_dimension_range("batch", 2, Some(1), Some(4))
            .with_symbolic_dimension_range("sequence", 3, Some(1), Some(8)),
    )
    .unwrap();
    assert!(matches!(
        onnx::export(original.program(), original.initializers()),
        Err(Error::UnsupportedExport { .. })
    ));
}

#[test]
fn test_negative_permutation_axes_export_as_a_full_nonnegative_permutation() {
    let input = tensor(vec![1., 2., 3., 4., 5., 6.], &[2, 3]);
    let (captured, _) = Graph::capture([("x", input.clone())], |inputs| {
        Ok(vec![inputs[0].permute(&[-1, -2])?])
    })
    .unwrap();
    let program = captured.program().unwrap();
    let bytes = onnx::export(&program, &program.bind_state(&HashMap::new()).unwrap()).unwrap();
    let imported = OnnxModel::from_reader(bytes.as_slice(), OnnxOptions::new()).unwrap();
    assert_eq!(
        imported
            .run_ordered(&HashMap::from([("x".into(), input.clone())]))
            .unwrap()[0]
            .to_bytes()
            .unwrap(),
        input.permute(&[1, 0]).unwrap().to_bytes().unwrap()
    );
}

#[test]
fn test_captured_scalar_softmax_roundtrips_as_rank_zero() {
    let input = tensor(vec![2.], &[]);
    let (captured, _) = Graph::capture([("x", input.clone())], |inputs| {
        Ok(vec![inputs[0].softmax(0)?])
    })
    .unwrap();
    let program = captured.program().unwrap();
    let bytes = onnx::export(&program, &program.bind_state(&HashMap::new()).unwrap()).unwrap();
    let imported = OnnxModel::from_reader(bytes.as_slice(), OnnxOptions::new()).unwrap();
    let output = imported
        .run_ordered(&HashMap::from([("x".into(), input)]))
        .unwrap();
    assert!(output[0].shape().dimensions().is_empty());
    assert_eq!(output[0].to_vec::<f32>().unwrap(), [1.]);
}

#[cfg(feature = "cuda")]
#[test]
fn test_half_silu_export_retains_fp32_intermediates_in_eager_and_native_replay() {
    use singe::graph::{GraphExecutionMode, GraphPrepareOptions};
    for dtype in [DataType::F16, DataType::Bf16] {
        let input = tensor(vec![-7., -1.5, -0.5, 0., 0.5, 1.5, 7.], &[7])
            .cast(dtype)
            .unwrap()
            .to_device(Device::Cuda(0))
            .unwrap();
        let program = one_operation(
            input.value_spec(),
            Operation::Silu,
            OperationAttributes::None,
        );
        let bytes = onnx::export(&program, &program.bind_state(&HashMap::new()).unwrap()).unwrap();
        let imported = OnnxModel::from_reader(bytes.as_slice(), OnnxOptions::new()).unwrap();
        let expected = input
            .silu()
            .unwrap()
            .to_device(Device::Cpu)
            .unwrap()
            .cast(DataType::F32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        for mode in [GraphExecutionMode::Eager, GraphExecutionMode::Compiled] {
            let mut executor = imported
                .prepare(GraphPrepareOptions::new(Device::Cuda(0)).with_execution_mode(mode))
                .unwrap();
            let output = executor
                .run(&HashMap::from([("x".into(), input.clone())]))
                .unwrap()
                .remove(0);
            assert_eq!(output.data_type(), dtype);
            let actual = output
                .to_device(Device::Cpu)
                .unwrap()
                .cast(DataType::F32)
                .unwrap()
                .to_vec::<f32>()
                .unwrap();
            for (actual, expected) in actual.iter().zip(&expected) {
                assert!(
                    (actual - expected).abs() <= f32::mul_add(expected.abs(), 1e-3, 1e-3),
                    "{dtype:?}: actual {actual}, expected {expected}"
                );
            }
        }
    }
}

#[test]
fn test_malformed_versioned_shape_metadata_is_rejected_at_import() {
    let program = Graph::build(
        [(
            "x",
            GraphInputSpec::new(ValueSpec::new(DataType::F32, Shape::create([2]).unwrap()))
                .with_dimension(0, "n", Some(1), Some(3))
                .unwrap(),
        )],
        CaptureBindings::new(),
        |inputs| Ok(vec![inputs[0].clone()]),
    )
    .unwrap()
    .program()
    .unwrap();
    let bytes = onnx::export(&program, &program.bind_state(&HashMap::new()).unwrap()).unwrap();
    for (key, value) in [
        ("singe.axis.0.min", "nonsense"),
        ("singe.axis.99.min", "1"),
        ("singe.axis.0.unknown", "1"),
    ] {
        let mut proto = decode_proto(&bytes).unwrap();
        let properties = &mut proto.graph.as_mut().unwrap().input[0].metadata_props;
        if let Some(property) = properties.iter_mut().find(|property| property.key == key) {
            property.value = value.into();
        } else {
            properties.push(StringStringEntryProto {
                key: key.into(),
                value: value.into(),
            });
        }
        assert!(matches!(
            OnnxModel::from_reader(encode_proto(&proto).as_slice(), OnnxOptions::new()),
            Err(Error::InvalidModel { .. })
        ));
    }
}

#[test]
fn test_graphs_that_write_inputs_export_present_outputs_and_round_trip() {
    let cache = tensor(vec![0.; 6], &[3, 2]);
    let row = tensor(vec![1., 2.], &[1, 2]);
    let step = |cache: &Tensor, row: &Tensor| -> singe::Result<Tensor> {
        cache.narrow(0, 1, 1)?.assign(row)?;
        cache.add(cache)
    };
    let (captured, _) = Graph::capture([("cache", cache), ("row", row)], |inputs| {
        Ok(vec![step(&inputs[0], &inputs[1])?])
    })
    .unwrap();
    let program = captured.program().unwrap();
    let state = program.bind_state(&HashMap::new()).unwrap();
    let bytes = onnx::export(&program, &state).unwrap();
    let model = decode_proto(&bytes).unwrap();
    let graph = model.graph.unwrap();
    assert!(
        graph
            .output
            .iter()
            .any(|output| output.name == "present.cache")
    );

    let imported = OnnxModel::from_reader(bytes.as_slice(), OnnxOptions::new()).unwrap();
    let past = tensor(vec![5., 6., 7., 8., 9., 10.], &[3, 2]);
    let next = tensor(vec![3., 4.], &[1, 2]);
    let outputs = imported
        .run_ordered(&HashMap::from([
            ("cache".into(), past.clone()),
            ("row".into(), next.clone()),
        ]))
        .unwrap();
    let expected = step(&past, &next).unwrap();
    assert_eq!(outputs.len(), 2);
    assert_eq!(
        outputs[0].to_vec::<f32>().unwrap(),
        expected.to_vec::<f32>().unwrap()
    );
    assert_eq!(
        outputs[1].to_vec::<f32>().unwrap(),
        past.to_vec::<f32>().unwrap()
    );
}

#[test]
fn test_random_operations_export_as_onnx_random_operators() {
    let generator = Generator::new(3);
    let weights = tensor(vec![1., 2., 3., 4., 3., 2.], &[2, 3]);
    let (captured, _) = Graph::capture([("weights", weights)], |inputs| {
        Ok(vec![
            Tensor::rand(
                Device::Cpu,
                Shape::create([2, 3]).unwrap(),
                DataType::F32,
                Some(&generator),
            )?,
            Tensor::randn(
                Device::Cpu,
                Shape::create([4]).unwrap(),
                DataType::F32,
                Some(&generator),
            )?,
            inputs[0].mul_scalar(0.1)?.bernoulli(Some(&generator))?,
            inputs[0].multinomial(2, Some(&generator))?,
        ])
    })
    .unwrap();
    let program = captured.program().unwrap();
    let bytes = onnx::export(&program, &program.bind_state(&HashMap::new()).unwrap()).unwrap();
    let graph = decode_proto(&bytes).unwrap().graph.unwrap();
    for operator in ["RandomUniform", "RandomNormal", "Bernoulli", "Multinomial"] {
        assert!(
            graph.node.iter().any(|node| node.op_type == operator),
            "missing {operator}"
        );
    }
}
