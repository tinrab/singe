#![cfg(feature = "onnx")]

use std::{collections::HashMap, env, fs, path::PathBuf};

use serde_json::{Value, json};
use singe::{
    DataType, Device, OnnxModel, OnnxOptions, Shape, Tensor,
    backend::registry::{
        Conv2dAttributes, FlattenAttributes, Operation, OperationAttributes, Pool2dAttributes,
        ReductionAttributes, TransposeAttributes,
    },
    dispatch::call_one,
    graph::{CaptureBindings, Graph, GraphInputSpec, GraphPrepareOptions, GraphProgram},
    onnx::{self, Error},
    value::ValueSpec,
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

fn check_export(
    name: &str,
    operation: Operation,
    attributes: OperationAttributes,
    input_shape: &[i64],
    output_shape: &[i64],
) {
    let shape = Shape::create(input_shape.to_vec()).unwrap();
    let values = (0..shape.element_count())
        .map(|index| index as f32 + 1.)
        .collect::<Vec<_>>();
    let input = Tensor::from_vec(Device::Cpu, shape, values.clone()).unwrap();
    let program = one_operation(input.value_spec(), operation, attributes);
    let bytes = onnx::export(&program, &program.bind_state(&HashMap::new()).unwrap()).unwrap();
    let imported = OnnxModel::from_reader(bytes.as_slice(), OnnxOptions::new()).unwrap();
    let outputs = imported
        .run_ordered(&HashMap::from([("x".into(), input)]))
        .unwrap();
    assert_eq!(outputs[0].shape().dimensions(), output_shape, "{name}");
    assert_eq!(outputs[0].to_vec::<f32>().unwrap(), values, "{name}");
    record_export(
        name,
        &bytes,
        &json!([{
            "input_shape": input_shape,
            "input": values,
            "output_shape": output_shape,
            "output": values,
        }]),
    );
}

fn record_export(name: &str, bytes: &[u8], cases: &Value) {
    if let Some(directory) = env::var_os("SINGE_ONNX_EXPORT_DIR") {
        let directory = PathBuf::from(directory).join("shapes");
        fs::create_dir_all(&directory).unwrap();
        fs::write(directory.join(format!("{name}.onnx")), bytes).unwrap();
        fs::write(
            directory.join(format!("{name}.json")),
            serde_json::to_vec(cases).unwrap(),
        )
        .unwrap();
    }
}

#[test]
fn test_dynamic_squeeze_exports_only_invariant_singleton_status() {
    for singleton in [false, true] {
        let (representative, min, max, axis) = if singleton {
            (2, Some(1), Some(4), 1)
        } else {
            (2, Some(2), Some(4), 0)
        };
        let program = one_operation(
            GraphInputSpec::new(ValueSpec::new(
                DataType::F32,
                Shape::create([representative, 1]).unwrap(),
            ))
            .with_dimension(0, "batch", min, max)
            .unwrap(),
            Operation::Squeeze,
            OperationAttributes::Dimension(axis),
        );
        let state = program.bind_state(&HashMap::new()).unwrap();
        let bytes = onnx::export(&program, &state).unwrap();
        let imported = OnnxModel::from_reader(bytes.as_slice(), OnnxOptions::new()).unwrap();
        let mut native = program
            .prepare(&state, GraphPrepareOptions::new(Device::Cpu))
            .unwrap();
        let mut cases = Vec::new();
        for batch in [2, 3, 4] {
            let values = (0..batch)
                .map(|value| value as f32 + 1.)
                .collect::<Vec<_>>();
            let shape = Shape::create([batch, 1]).unwrap();
            let inputs = HashMap::from([(
                "x".into(),
                Tensor::from_vec(Device::Cpu, shape, values.clone()).unwrap(),
            )]);
            let expected_shape = if singleton {
                vec![batch]
            } else {
                vec![batch, 1]
            };
            for outputs in [
                native.run(&inputs).unwrap(),
                imported.run_ordered(&inputs).unwrap(),
            ] {
                assert_eq!(outputs[0].shape().dimensions(), expected_shape);
                assert_eq!(outputs[0].to_vec::<f32>().unwrap(), values);
            }
            cases.push(json!({
                "input_shape": [batch, 1],
                "input": values,
                "output_shape": expected_shape,
                "output": values,
            }));
        }
        record_export(
            &format!("squeeze_dynamic_{singleton}"),
            &bytes,
            &json!(cases),
        );
    }
}

#[test]
fn test_scalar_transpose_exports_all_virtual_axis_pairs_without_panicking() {
    for dim0 in [0, -1] {
        for dim1 in [0, -1] {
            check_export(
                &format!("transpose_{dim0}_{dim1}"),
                Operation::Transpose,
                OperationAttributes::Transpose(TransposeAttributes { dim0, dim1 }),
                &[],
                &[],
            );
        }
    }
}

#[test]
fn test_scalar_flatten_exports_a_vector_for_both_virtual_axes() {
    for start_dim in [0, -1] {
        for end_dim in [0, -1] {
            check_export(
                &format!("flatten_{start_dim}_{end_dim}"),
                Operation::Flatten,
                OperationAttributes::Flatten(FlattenAttributes { start_dim, end_dim }),
                &[],
                &[1],
            );
        }
    }
}

#[test]
fn test_squeeze_exports_noops_and_singletons_with_native_shapes() {
    for dim in [0, -2] {
        check_export(
            &format!("squeeze_noop_{dim}"),
            Operation::Squeeze,
            OperationAttributes::Dimension(dim),
            &[2, 3],
            &[2, 3],
        );
    }
    for dim in [1, -1] {
        check_export(
            &format!("squeeze_singleton_{dim}"),
            Operation::Squeeze,
            OperationAttributes::Dimension(dim),
            &[2, 1],
            &[2],
        );
    }
    for dim in [0, -1] {
        check_export(
            &format!("squeeze_scalar_{dim}"),
            Operation::Squeeze,
            OperationAttributes::Dimension(dim),
            &[],
            &[],
        );
    }
}

#[test]
fn test_scalar_reductions_export_normalized_virtual_axes() {
    for operation in [
        Operation::Sum,
        Operation::Mean,
        Operation::Min,
        Operation::Max,
    ] {
        for dim in [0, -1] {
            for keepdim in [false, true] {
                check_export(
                    &format!("reduce_{operation:?}_{dim}_{keepdim}"),
                    operation,
                    OperationAttributes::Reduction(ReductionAttributes {
                        dimensions: vec![dim],
                        keepdim,
                    }),
                    &[],
                    &[],
                );
            }
        }
    }
}

#[test]
fn test_bf16_spatial_operators_are_rejected_before_serialization() {
    for operation in [
        Operation::Conv2d,
        Operation::MaxPool2d,
        Operation::AvgPool2d,
    ] {
        let image = ValueSpec::new(DataType::Bf16, Shape::create([1, 1, 4, 4]).unwrap());
        let program = if operation == Operation::Conv2d {
            Graph::build(
                [
                    ("x", image),
                    (
                        "weight",
                        ValueSpec::new(DataType::Bf16, Shape::create([1, 1, 2, 2]).unwrap()),
                    ),
                ],
                CaptureBindings::new(),
                |inputs| {
                    Ok(vec![call_one(
                        operation,
                        OperationAttributes::Conv2d(Conv2dAttributes {
                            stride: (1, 1),
                            padding: (0, 0, 0, 0),
                            dilation: (1, 1),
                            groups: 1,
                            has_bias: false,
                        }),
                        &[&inputs[0], &inputs[1]],
                    )?])
                },
            )
            .unwrap()
            .program()
            .unwrap()
        } else {
            one_operation(
                image,
                operation,
                OperationAttributes::Pool2d(Pool2dAttributes {
                    kernel_size: (2, 2),
                    stride: (1, 1),
                    padding: (0, 0, 0, 0),
                    dilation: (1, 1),
                    ceil_mode: false,
                    count_include_pad: (operation == Operation::AvgPool2d).then_some(false),
                }),
            )
        };
        assert!(
            matches!(
                onnx::export(&program, &program.bind_state(&HashMap::new()).unwrap()),
                Err(Error::UnsupportedExport { .. })
            ),
            "{operation:?}"
        );
    }
}

#[test]
fn test_variable_squeeze_dimensions_cannot_silently_change_exported_rank() {
    for representative in [1, 2] {
        let program = one_operation(
            GraphInputSpec::new(ValueSpec::new(
                DataType::F32,
                Shape::create([representative, 3]).unwrap(),
            ))
            .with_dimension(0, "batch", Some(1), Some(4))
            .unwrap(),
            Operation::Squeeze,
            OperationAttributes::Dimension(0),
        );
        assert!(matches!(
            onnx::export(&program, &program.bind_state(&HashMap::new()).unwrap()),
            Err(Error::UnsupportedExport { .. })
        ));
    }
}
