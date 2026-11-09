#![cfg(feature = "onnx")]

use std::collections::HashMap;
#[cfg(all(feature = "cuda", feature = "testing"))]
use std::slice;

use serde_json::Value;
use singe::{
    Device, OnnxModel, OnnxOptions, Shape, Tensor,
    graph::{GraphExecutionMode, GraphPrepareOptions},
};
#[cfg(all(feature = "cuda", feature = "testing"))]
use singe::{
    backend::cuda::export::{CudaExport, Library, StateKind},
    testing::{CompiledProgram, maybe_lock_device, program_target},
};

// Missing required exports fail compilation.
// Regenerate with the pinned exporter.
const MODELS: [(&str, &[u8]); 3] = [
    ("mlp", include_bytes!("../references/layers/mlp.onnx")),
    ("cnn", include_bytes!("../references/layers/cnn.onnx")),
    (
        "encoder",
        include_bytes!("../references/layers/encoder.onnx"),
    ),
];

#[expect(
    clippy::cast_possible_truncation,
    reason = "fixture values are float32 values stored as f64 JSON numbers"
)]
fn values(value: &Value) -> Vec<f32> {
    value["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

fn assert_output(actual: &Tensor, expected: &Value) {
    let dimensions: Vec<_> = expected["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect();
    assert_eq!(actual.shape().dimensions(), dimensions);
    let actual = actual.to_vec::<f32>().unwrap();
    let expected = values(expected);
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 3e-4,
            "actual {actual}, expected {expected}"
        );
    }
}

fn check_exports(device: Device) {
    let references: Value = singe_testing::reference!("layers").resolved();
    for (name, bytes) in MODELS {
        let model = OnnxModel::from_reader(bytes, OnnxOptions::new().with_device(device)).unwrap();
        assert_eq!(model.inputs().len(), 1, "{name}");
        assert_eq!(model.outputs()[0].name, "y");
        assert!(
            model
                .initializers()
                .bindings()
                .all(|(_, tensor)| tensor.device() == Device::Cpu)
        );
        let revision = model.initializers().revision();
        for mode in [GraphExecutionMode::Eager, GraphExecutionMode::Compiled] {
            let mut executor = model
                .prepare(GraphPrepareOptions::new(device).with_execution_mode(mode))
                .unwrap();
            let cases = references["models"][name]["cases"].as_array().unwrap();
            let mut retained = None;
            for case in cases {
                let dimensions: Vec<_> = case["input"]["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_i64().unwrap())
                    .collect();
                let input = Tensor::from_vec(
                    device,
                    Shape::create(dimensions).unwrap(),
                    values(&case["input"]),
                )
                .unwrap();
                let bindings = HashMap::from([("x".to_owned(), input)]);
                let output = executor.run(&bindings).unwrap();
                assert_output(&output[0], &case["output"]);
                assert_output(&model.run(&bindings).unwrap()["y"], &case["output"]);
                if retained.is_none() {
                    retained = Some(output[0].clone());
                }
            }
            assert_output(&retained.unwrap(), &cases[0]["output"]);
            assert_eq!(executor.cache_stats().analysis_builds(), 1);
            assert_eq!(executor.runs(), 2);
            assert_eq!(model.initializers().revision(), revision);
        }
    }
}

#[test]
fn test_required_pytorch_exports_run_eager_and_prepared_on_cpu() {
    check_exports(Device::Cpu);
}

#[cfg(feature = "cuda")]
#[test]
fn test_required_pytorch_exports_run_eager_and_replay_on_cuda() {
    check_exports(Device::Cuda(0));
}

// Each model as a C++ program, once from generated kernels alone and once with every library allowed.
#[cfg(all(feature = "cuda", feature = "testing"))]
#[test]
fn test_required_pytorch_exports_run_as_cuda_programs() {
    let device = Device::Cuda(0);
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let Some(architecture) = program_target(device) else {
        return;
    };
    let references: Value = singe_testing::reference!("layers").resolved();
    for (name, bytes) in MODELS {
        let model = OnnxModel::from_reader(bytes, OnnxOptions::new().with_device(device)).unwrap();
        let graph = model.graph().unwrap();
        for (label, libraries) in [
            ("generated", Vec::new()),
            ("libraries", Library::ALL.to_vec()),
        ] {
            let program = CudaExport::new(architecture)
                .with_libraries(libraries)
                .with_state_functions(true)
                .export(&graph)
                .unwrap();
            let compiled = CompiledProgram::create(&program, &format!("{name}-{label}")).unwrap();
            let mut checked = 0;
            for case in references["models"][name]["cases"].as_array().unwrap() {
                let dimensions: Vec<_> = case["input"]["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_i64().unwrap())
                    .collect();
                // Programs bake in their shapes.
                if program.inputs[0].layout.dimensions() != dimensions {
                    continue;
                }
                let input = Tensor::from_vec(
                    device,
                    Shape::create(dimensions).unwrap(),
                    values(&case["input"]),
                )
                .unwrap();
                let outputs = compiled.run(&[input], 1).unwrap();
                assert_output(&outputs[0], &case["output"]);
                checked += 1;
            }
            assert!(checked > 0, "{name} {label} ran no reference case");
            // The model's initializers are the program's tensors: clearing them changes its results, and restoring them restores them.
            let instance = compiled.instance().unwrap();
            let case = &references["models"][name]["cases"].as_array().unwrap()[0];
            if program.inputs[0].layout.dimensions()
                == case["input"]["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_i64().unwrap())
                    .collect::<Vec<_>>()
            {
                let input = Tensor::from_vec(
                    device,
                    Shape::create(program.inputs[0].layout.dimensions().to_vec()).unwrap(),
                    values(&case["input"]),
                )
                .unwrap();
                let saved = program
                    .state
                    .iter()
                    .filter(|state| state.kind == StateKind::Initializer)
                    .map(|state| (state.name.clone(), instance.get(&state.name).unwrap()))
                    .collect::<Vec<_>>();
                assert!(!saved.is_empty(), "{name} has no initializers");
                for (state, tensor) in &saved {
                    let zeros =
                        Tensor::zeroed_on(device, tensor.data_type(), tensor.shape().clone())
                            .unwrap();
                    instance.set(state, &zeros).unwrap();
                }
                let cleared = instance.run(slice::from_ref(&input), 1).unwrap().remove(0);
                let expected = values(&case["output"]);
                assert!(
                    cleared
                        .to_vec::<f32>()
                        .unwrap()
                        .iter()
                        .zip(&expected)
                        .any(|(a, b)| (a - b).abs() > 1e-3),
                    "{name} ignored its cleared initializers"
                );
                for (state, tensor) in &saved {
                    instance.set(state, tensor).unwrap();
                }
                let restored = instance.run(slice::from_ref(&input), 1).unwrap();
                assert_output(&restored[0], &case["output"]);
            }
        }
    }
}
