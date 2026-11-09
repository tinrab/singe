#![cfg(feature = "onnx")]

use std::collections::HashMap;

use serde_json::Value;
use singe::{
    Device, OnnxModel, OnnxOptions, Shape, Tensor,
    error::ErrorChain,
    graph::{GraphExecutionMode, GraphPrepareOptions},
};

#[cfg(feature = "testing")]
use singe::testing::TensorActivity;
#[cfg(all(feature = "cuda", feature = "testing"))]
use singe::{
    backend::cuda::export::CudaExport,
    testing::{CompiledProgram, maybe_lock_device, program_target},
};

const MODELS: [(&str, &[u8]); 4] = [
    (
        "shape_as_data",
        include_bytes!("../references/shapes/shape_as_data.onnx"),
    ),
    (
        "scalar_shape_gather",
        include_bytes!("../references/shapes/scalar_shape_gather.onnx"),
    ),
    (
        "signed_shape_division",
        include_bytes!("../references/shapes/signed_shape_division.onnx"),
    ),
    (
        "reshape_gathered_shape",
        include_bytes!("../references/shapes/reshape_gathered_shape.onnx"),
    ),
];

fn options(batch: i64, sequence: i64) -> OnnxOptions {
    OnnxOptions::new()
        .with_symbolic_dimension_range("batch", batch, Some(1), Some(4))
        .with_symbolic_dimension_range("sequence", sequence, Some(1), Some(8))
}

#[test]
fn test_compiled_shape_constants_reject_changed_symbolic_dimensions() {
    check_compiled_shape_constants(Device::Cpu);
}

#[cfg(feature = "cuda")]
#[test]
fn test_compiled_shape_constants_reject_changed_symbolic_dimensions_on_cuda() {
    check_compiled_shape_constants(Device::Cuda(0));
}

fn check_compiled_shape_constants(device: Device) {
    for (_, bytes) in MODELS {
        let model = OnnxModel::from_reader(bytes, options(2, 3).with_device(device)).unwrap();
        let compiled = model.compile().unwrap();
        let original = HashMap::from([(
            "x".into(),
            Tensor::zeroed_on(
                device,
                singe::DataType::F32,
                Shape::create([2, 3, 4]).unwrap(),
            )
            .unwrap(),
        )]);
        compiled.run(&original).unwrap();
        for dimensions in [[3, 5, 4], [3, 3, 4], [2, 5, 4]] {
            let changed = HashMap::from([(
                "x".into(),
                Tensor::zeroed_on(
                    device,
                    singe::DataType::F32,
                    Shape::create(dimensions).unwrap(),
                )
                .unwrap(),
            )]);
            #[cfg(feature = "testing")]
            let before = TensorActivity::current();
            assert!(compiled.run(&changed).is_err());
            #[cfg(feature = "testing")]
            assert_eq!(TensorActivity::current(), before);
            // The common executor retains the logical shape program and can evaluate another specialization rather than returning stale data.
            model.run_ordered(&changed).unwrap();
        }
        compiled.run(&original).unwrap();
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "fixture values are float32 values stored as f64 JSON numbers"
)]
fn value_f32(value: &Value) -> f32 {
    value.as_f64().unwrap() as f32
}

fn input(case: &Value, device: Device) -> Tensor {
    Tensor::from_vec(
        device,
        Shape::create(
            case["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|dim| dim.as_i64().unwrap()),
        )
        .unwrap(),
        case["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(value_f32)
            .collect(),
    )
    .unwrap()
}

fn compare(output: &Tensor, expected: &Value, dtype: &str) {
    assert_eq!(
        output.shape().dimensions(),
        expected["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|dim| dim.as_i64().unwrap())
            .collect::<Vec<_>>()
    );
    if dtype == "i64" {
        assert_eq!(
            output.to_vec::<i64>().unwrap(),
            expected["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_i64().unwrap())
                .collect::<Vec<_>>()
        );
    } else {
        for (actual, expected) in output
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .zip(expected["values"].as_array().unwrap())
        {
            assert!((*actual - value_f32(expected)).abs() <= 1e-6);
        }
    }
}

fn exercise(device: Device) {
    let references: Value = singe_testing::reference!("shapes/shape_cases").resolved();
    for (name, bytes) in MODELS {
        #[cfg(feature = "testing")]
        let dispatched = TensorActivity::current().checked_invocations;
        let imported = OnnxModel::from_reader(bytes, options(2, 3)).unwrap();
        #[cfg(feature = "testing")]
        assert_eq!(TensorActivity::current().checked_invocations, dispatched);
        let second = OnnxModel::from_reader(bytes, options(3, 5)).unwrap();
        assert_eq!(
            imported.program().fingerprint().unwrap(),
            second.program().fingerprint().unwrap(),
            "{name}"
        );
        let reference = &references[name];
        let modes = if device.is_cpu() {
            vec![GraphExecutionMode::Compiled]
        } else {
            vec![GraphExecutionMode::Eager, GraphExecutionMode::Compiled]
        };
        for mode in modes {
            let mut executor = imported
                .prepare(
                    GraphPrepareOptions::new(device)
                        .with_execution_mode(mode)
                        .with_cache_capacity(2),
                )
                .unwrap();
            let mut retained = Vec::new();
            for case in reference["cases"].as_array().unwrap() {
                let inputs = HashMap::from([("x".into(), input(&case["input"], device))]);
                #[cfg(feature = "testing")]
                let readbacks = TensorActivity::current().cuda_readbacks;
                let output = executor.run(&inputs).unwrap().remove(0);
                #[cfg(feature = "testing")]
                assert_eq!(TensorActivity::current().cuda_readbacks, readbacks);
                compare(
                    &output,
                    &case["output"],
                    reference["dtype"].as_str().unwrap(),
                );
                retained.push((output, case["output"].clone()));
            }
            assert!(executor.cache_stats().entries() <= 2);
            for (output, expected) in retained {
                compare(&output, &expected, reference["dtype"].as_str().unwrap());
            }
        }
        if name == "reshape_gathered_shape" {
            let replacement =
                Tensor::from_vec(Device::Cpu, Shape::create([3]).unwrap(), vec![0_i64, 1, 2])
                    .unwrap();
            let changed = imported
                .initializers()
                .replace(&HashMap::from([("order".into(), replacement)]))
                .unwrap();
            assert!(
                imported
                    .program()
                    .prepare(&changed, GraphPrepareOptions::new(device))
                    .is_err()
            );
            assert!(
                imported
                    .prepare(GraphPrepareOptions::new(device))
                    .unwrap()
                    .report()
                    .unwrap()
                    .contains("value-specialized state bindings")
            );
        }
    }
}

#[test]
fn test_onnx_shape_values_match_reference_evaluator_on_cpu() {
    exercise(Device::Cpu);
}

#[cfg(feature = "cuda")]
#[test]
fn test_onnx_shape_values_match_reference_evaluator_without_cuda_readbacks() {
    exercise(Device::Cuda(0));
}

fn exercise_rejections(device: Device) {
    for bytes in [
        include_bytes!("../references/shapes/data_dependent_reshape.onnx").as_slice(),
        include_bytes!("../references/shapes/unsupported_float_shape.onnx").as_slice(),
    ] {
        #[cfg(feature = "testing")]
        let before = TensorActivity::current().checked_invocations;
        assert!(OnnxModel::from_reader(bytes, options(2, 3).with_device(device)).is_err());
        #[cfg(feature = "testing")]
        assert_eq!(TensorActivity::current().checked_invocations, before);
    }
    for bytes in [
        include_bytes!("../references/shapes/fixed_output_annotation.onnx").as_slice(),
        include_bytes!("../references/shapes/fixed_intermediate_annotation.onnx").as_slice(),
    ] {
        let imported = OnnxModel::from_reader(bytes, options(2, 3)).unwrap();
        let mut executor = imported.prepare(GraphPrepareOptions::new(device)).unwrap();
        let inputs = HashMap::from([(
            "x".into(),
            Tensor::full(device, Shape::create([3, 3, 4]).unwrap(), 1_f32).unwrap(),
        )]);
        #[cfg(feature = "testing")]
        let before = TensorActivity::current();
        let cache = executor.cache_stats();
        let error = executor.run(&inputs).unwrap_err();
        assert!(error.to_string().contains("annotation"), "{error}");
        #[cfg(feature = "testing")]
        assert_eq!(TensorActivity::current(), before);
        assert_eq!(executor.cache_stats(), cache);
    }
    for (bytes, representative, actual) in [
        (
            include_bytes!("../references/shapes/shape_overflow.onnx").as_slice(),
            0,
            2,
        ),
        (
            include_bytes!("../references/shapes/shape_division_by_zero.onnx").as_slice(),
            2,
            1,
        ),
    ] {
        let opts = OnnxOptions::new()
            .with_symbolic_dimension_range("batch", representative, Some(0), Some(4))
            .with_symbolic_dimension_range("sequence", 1, Some(1), Some(8));
        let imported = OnnxModel::from_reader(bytes, opts).unwrap();
        let mut executor = imported.prepare(GraphPrepareOptions::new(device)).unwrap();
        let inputs = HashMap::from([(
            "x".into(),
            Tensor::full(device, Shape::create([actual, 1, 4]).unwrap(), 0_f32).unwrap(),
        )]);
        #[cfg(feature = "testing")]
        let before = TensorActivity::current();
        let cache = executor.cache_stats();
        let error = executor.run(&inputs).unwrap_err();
        let report = ErrorChain(&error).to_string();
        assert!(report.contains("shape arithmetic"), "{report}");
        assert!(report.contains("onnx node"), "{report}");
        #[cfg(feature = "testing")]
        assert_eq!(TensorActivity::current(), before);
        assert_eq!(executor.cache_stats(), cache);
    }
}

#[test]
fn test_onnx_shape_rejections_precede_dispatch_and_cache_publication() {
    exercise_rejections(Device::Cpu);
}

#[cfg(feature = "cuda")]
#[test]
fn test_onnx_shape_rejections_precede_cuda_submission_and_readback() {
    exercise_rejections(Device::Cuda(0));
}

#[test]
fn test_fixed_metadata_remains_usable_by_static_shape_operands() {
    let imported = OnnxModel::from_reader(
        include_bytes!("../references/shapes/static_metadata_expand.onnx").as_slice(),
        OnnxOptions::new(),
    )
    .unwrap();
    let tensor = Tensor::full(Device::Cpu, Shape::create([2, 3, 4]).unwrap(), 0_f32).unwrap();
    let output = imported
        .run_ordered(&HashMap::from([("x".into(), tensor)]))
        .unwrap();
    assert_eq!(output[0].to_vec::<f32>().unwrap(), vec![1.; 24]);
}

#[test]
fn test_representative_symbol_values_do_not_pin_runtime_dimensions() {
    let bytes = include_bytes!("../references/shapes/shape_as_data.onnx").as_slice();
    let imported = OnnxModel::from_reader(
        bytes,
        OnnxOptions::new()
            .with_symbolic_dimension("batch", 2)
            .with_symbolic_dimension("sequence", 3),
    )
    .unwrap();
    let mut executor = imported
        .prepare(GraphPrepareOptions::new(Device::Cpu))
        .unwrap();
    let inputs = HashMap::from([(
        "x".into(),
        Tensor::full(Device::Cpu, Shape::create([5, 9, 4]).unwrap(), 0_f32).unwrap(),
    )]);
    assert_eq!(
        executor.run(&inputs).unwrap()[0].to_vec::<i64>().unwrap(),
        [6, 10, 5]
    );

    let fixed = OnnxModel::from_reader(
        bytes,
        OnnxOptions::new()
            .with_symbolic_dimension_range("batch", 2, Some(2), Some(2))
            .with_symbolic_dimension_range("sequence", 3, Some(3), Some(3)),
    )
    .unwrap();
    assert_ne!(
        imported.program().fingerprint().unwrap(),
        fixed.program().fingerprint().unwrap()
    );
    assert!(
        fixed
            .prepare(GraphPrepareOptions::new(Device::Cpu))
            .unwrap()
            .run(&inputs)
            .is_err()
    );
}

// A model's symbolic dimensions take their values from its options, and a program bakes those in, including the shape arithmetic the model does on them.
#[cfg(all(feature = "cuda", feature = "testing"))]
#[test]
fn test_programs_export_models_with_symbolic_dimensions_at_any_chosen_size() {
    let device = Device::Cuda(0);
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let Some(architecture) = program_target(device) else {
        return;
    };
    let models = MODELS.into_iter().chain([(
        "encoder",
        include_bytes!("../references/shapes/encoder.onnx").as_slice(),
    )]);
    for (name, bytes) in models {
        for (batch, sequence) in [(2, 3), (3, 5)] {
            let model = OnnxModel::from_reader(bytes, options(batch, sequence).with_device(device))
                .unwrap();
            let shape = Shape::create([batch, sequence, 4]).unwrap();
            let count = i32::try_from(shape.element_count()).unwrap();
            let values = (0..count)
                .map(|index| f32::from(i16::try_from(index % 17 - 8).unwrap()) / 8.0)
                .collect::<Vec<_>>();
            let input = Tensor::from_vec(device, shape, values).unwrap();
            let bindings = HashMap::from([("x".to_owned(), input.clone())]);
            let expected = model.run_ordered(&bindings).unwrap();
            let program = CudaExport::new(architecture)
                .without_libraries()
                .export(&model.graph().unwrap())
                .unwrap();
            let label = format!("{name}-{batch}-{sequence}");
            let compiled = CompiledProgram::create(&program, &label).unwrap();
            let actual = compiled.run(&[input], 1).unwrap();
            assert_eq!(actual.len(), expected.len(), "{label}");
            for (actual, expected) in actual.iter().zip(&expected) {
                assert_eq!(actual.shape(), expected.shape(), "{label}");
                let actual = actual
                    .cast(singe::DataType::F32)
                    .unwrap()
                    .to_vec::<f32>()
                    .unwrap();
                let expected = expected
                    .cast(singe::DataType::F32)
                    .unwrap()
                    .to_vec::<f32>()
                    .unwrap();
                for (actual, expected) in actual.iter().zip(&expected) {
                    assert!(
                        (actual - expected).abs() <= 1e-3,
                        "{label}: {actual} != {expected}"
                    );
                }
            }
        }
    }
}
