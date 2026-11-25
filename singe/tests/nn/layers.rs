use std::{
    collections::HashMap,
    env, fs,
    path::{Path, PathBuf},
    process, ptr,
};

use serde_json::Value;
#[cfg(feature = "onnx")]
use serde_json::json;
use singe::{
    DataType, Device, Error, Shape, Tensor,
    error::CheckpointError,
    graph::{GraphInputSpec, GraphPrepareOptions},
    nn::{
        Activation, BatchNorm, Checkpoint, CheckpointOptions, Conv2d, Conv2dConfig, EncoderBlock,
        Flatten, LayerNorm, Linear, LoadPlan, Model, ModelState, Pool2d, Pool2dConfig, Pool2dKind,
        SelfAttention, Sequential, StateBuilder, StateKey, StatePiece, TensorSource,
    },
    shape::spatial::Spatial2d,
    value::ValueSpec,
};
#[cfg(feature = "testing")]
use singe::{storage::cpu::CpuStorage, testing::TensorActivity};
use singe_testing::{Tolerance, assert_matches, host, integers, reference};

fn value_f32(value: &Value) -> f32 {
    value.as_f64().unwrap() as f32
}

fn tensor(dimensions: &[i64], values: Vec<f32>) -> Tensor {
    Tensor::from_vec(
        Device::Cpu,
        Shape::create(dimensions.to_vec()).unwrap(),
        values,
    )
    .unwrap()
}

fn fixture_tensor(value: &Value, device: Device) -> Tensor {
    tensor(
        &value["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap())
            .collect::<Vec<_>>(),
        value["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(value_f32)
            .collect(),
    )
    .to_device(device)
    .unwrap()
}

fn checkpoint(name: &str) -> &'static [u8] {
    match name {
        "mlp" => include_bytes!("../references/layers/mlp.safetensors"),
        "cnn" => include_bytes!("../references/layers/cnn.safetensors"),
        "encoder" => include_bytes!("../references/layers/encoder.safetensors"),
        _ => unreachable!(),
    }
}

fn fixture_model(name: &str, reference: &Value) -> Model<Sequential> {
    let mut builder = StateBuilder::new();
    let mut keys = HashMap::new();
    for (name, value) in reference["state"].as_object().unwrap() {
        let dimensions = value["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap())
            .collect::<Vec<_>>();
        let tensor = Tensor::zeroed_on(
            Device::Cpu,
            DataType::F32,
            Shape::create(dimensions).unwrap(),
        )
        .unwrap();
        let key = if name.contains("running_") {
            builder.buffer(name, tensor)
        } else {
            builder.parameter(name, tensor)
        }
        .unwrap();
        keys.insert(name.clone(), key);
    }
    let key = |name: &str| keys[name].clone();
    let linear = |name: &str| {
        Linear::new(
            key(&format!("{name}.weight")),
            Some(key(&format!("{name}.bias"))),
        )
    };
    let norm = |name: &str| {
        LayerNorm::new(
            vec![4],
            Some(key(&format!("{name}.weight"))),
            Some(key(&format!("{name}.bias"))),
            1e-5,
        )
    };
    let module = match name {
        "mlp" => Sequential::new()
            .with(linear("first"))
            .with(Activation::Relu)
            .with(linear("second")),
        "cnn" => Sequential::new()
            .with(Conv2d::new(
                key("conv.weight"),
                Some(key("conv.bias")),
                Conv2dConfig::default(),
            ))
            .with(BatchNorm::new(
                Some(key("norm.weight")),
                Some(key("norm.bias")),
                key("norm.running_mean"),
                key("norm.running_var"),
                1e-5,
            ))
            .with(Activation::Relu)
            .with(Pool2d::new(
                Pool2dKind::Average {
                    count_include_pad: true,
                },
                Pool2dConfig {
                    stride: Spatial2d::square(1),
                    ..Pool2dConfig::new((2, 2))
                },
            ))
            .with(Flatten::new(1, -1))
            .with(linear("head")),
        "encoder" => Sequential::new().with(EncoderBlock::new(
            SelfAttention::create(
                linear("query"),
                linear("key"),
                linear("value"),
                linear("output"),
                2,
                4,
            )
            .unwrap(),
            norm("norm1"),
            linear("expand"),
            linear("contract"),
            norm("norm2"),
        )),
        _ => unreachable!(),
    };
    let state = builder
        .finish()
        .unwrap()
        .load_safetensors_bytes(checkpoint(name))
        .unwrap();
    Model::new(module, state)
}

const LAYERS: Tolerance =
    Tolerance::relative(2e-5, 2e-5, "F32 layers sum in another order than PyTorch");

fn assert_close(actual: &Tensor, expected: &Value) {
    assert_eq!(actual.shape().dimensions(), integers(&expected["shape"]));
    let expected = expected["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(value_f32)
        .collect::<Vec<_>>();
    assert_matches(&host(actual), &expected, LAYERS, "layer output");
}

#[cfg(feature = "onnx")]
fn record_onnx_export(name: &str, device: Device, bytes: &[u8], cases: &Value) {
    if let Some(directory) = env::var_os("SINGE_ONNX_EXPORT_DIR") {
        let directory = PathBuf::from(directory);
        fs::create_dir_all(&directory).unwrap();
        let device = if device == Device::Cpu { "cpu" } else { "cuda" };
        let name = format!("{name}_{device}");
        fs::write(directory.join(format!("{name}.onnx")), bytes).unwrap();
        let reference = json!({"cases": cases, "input_dtype": "float32", "atol": 2e-5, "rtol": 2e-5, "reference_evaluator": true});
        fs::write(
            directory.join(format!("{name}.json")),
            serde_json::to_vec_pretty(&reference).unwrap(),
        )
        .unwrap();
    }
}

#[cfg(feature = "onnx")]
fn exercise_native_onnx_exports(device: Device) {
    use singe::{
        OnnxModel, OnnxOptions,
        graph::{GraphExecutionMode, ProgramInputBinding},
        onnx,
    };
    let references: Value = reference!("layers").resolved();
    for name in ["mlp", "cnn", "encoder"] {
        let reference = &references["models"][name];
        let model = fixture_model(name, reference).to(device, None).unwrap();
        let example = fixture_tensor(&reference["cases"][0]["input"], device);
        let built = model.build("x", example.value_spec()).unwrap();
        let captured = model.capture("x", example).unwrap();
        for (route, source) in [("build", built), ("capture", captured)] {
            #[cfg(feature = "testing")]
            let before = TensorActivity::current();
            let bytes = onnx::export(source.program(), source.state()).unwrap();
            #[cfg(feature = "testing")]
            assert_eq!(
                TensorActivity::current().checked_invocations,
                before.checked_invocations
            );
            assert_eq!(
                bytes,
                onnx::export(source.program(), source.state()).unwrap()
            );
            let imported = OnnxModel::from_reader(bytes.as_slice(), OnnxOptions::new()).unwrap();
            assert_eq!(
                imported
                    .inputs()
                    .iter()
                    .map(|input| input.name.as_str())
                    .collect::<Vec<_>>(),
                ["x"]
            );
            for binding in source.program().signature().inputs() {
                if matches!(binding.binding(), ProgramInputBinding::State(_)) {
                    assert_eq!(
                        imported
                            .initializers()
                            .tensor(binding.name())
                            .unwrap()
                            .to_bytes()
                            .unwrap(),
                        source
                            .state()
                            .tensor(binding.name())
                            .unwrap()
                            .to_bytes()
                            .unwrap()
                    );
                }
            }
            for mode in [GraphExecutionMode::Eager, GraphExecutionMode::Compiled] {
                let mut executor = imported
                    .prepare(GraphPrepareOptions::new(device).with_execution_mode(mode))
                    .unwrap();
                for case in reference["cases"].as_array().unwrap() {
                    let input = fixture_tensor(&case["input"], device);
                    let native = model.forward(&input).unwrap();
                    assert_close(&native, &case["output"]);
                    let actual = executor.run(&HashMap::from([("x".into(), input)])).unwrap();
                    assert_close(&actual[0], &case["output"]);
                }
            }
            record_onnx_export(
                &format!("{name}_{route}"),
                device,
                &bytes,
                &reference["cases"],
            );
        }
    }
    let dynamic: Value = reference!("shapes").resolved();
    let model = fixture_model("encoder", &references["models"]["encoder"])
        .to(device, None)
        .unwrap();
    let spec = GraphInputSpec::new(ValueSpec::new(
        DataType::F32,
        Shape::create([2, 3, 4]).unwrap(),
    ))
    .with_dimension(0, "batch", Some(1), Some(4))
    .unwrap()
    .with_dimension(1, "sequence", Some(1), Some(8))
    .unwrap();
    let source = model.build("x", spec).unwrap();
    let bytes = onnx::export(source.program(), source.state()).unwrap();
    // Representative values and inclusive bounds come from versioned export metadata.
    // Callers need not reconstruct the native input declaration.
    let imported = OnnxModel::from_reader(bytes.as_slice(), OnnxOptions::new()).unwrap();
    for mode in [GraphExecutionMode::Eager, GraphExecutionMode::Compiled] {
        let mut executor = imported
            .prepare(
                GraphPrepareOptions::new(device)
                    .with_execution_mode(mode)
                    .with_cache_capacity(2),
            )
            .unwrap();
        let mut retained = Vec::new();
        for case in dynamic["cases"].as_array().unwrap() {
            let input = fixture_tensor(&case["input"], device);
            assert_close(&model.forward(&input).unwrap(), &case["output"]);
            let actual = executor.run(&HashMap::from([("x".into(), input)])).unwrap();
            assert_close(&actual[0], &case["output"]);
            retained.push(actual[0].clone());
        }
        for (actual, case) in retained.iter().zip(dynamic["cases"].as_array().unwrap()) {
            assert_close(actual, &case["output"]);
        }
        for shape in [[0, 3, 4], [5, 3, 4], [2, 9, 4]] {
            let input =
                Tensor::zeroed_on(device, DataType::F32, Shape::create(shape).unwrap()).unwrap();
            #[cfg(feature = "testing")]
            let before = TensorActivity::current();
            assert!(executor.run(&HashMap::from([("x".into(), input)])).is_err());
            #[cfg(feature = "testing")]
            assert_eq!(
                TensorActivity::current().checked_invocations,
                before.checked_invocations
            );
        }
    }
    record_onnx_export("encoder_dynamic", device, &bytes, &dynamic["cases"]);
}

// Building runs on metadata: it may describe tensors, but allocates and reads nothing.
#[cfg(feature = "testing")]
fn assert_builds_without_work(before: TensorActivity, name: &str) {
    let after = TensorActivity::current();
    assert_eq!(after.storage_requests, before.storage_requests, "{name}");
    assert_eq!(after.cuda_readbacks, before.cuda_readbacks, "{name}");
}

#[cfg(feature = "onnx")]
#[test]
fn test_native_onnx_export_roundtrips_models_cpu() {
    exercise_native_onnx_exports(Device::Cpu);
}

fn exercise_models(device: Device) {
    let references: Value = reference!("layers").resolved();
    assert_eq!(references["reference"]["version"], "2.12.0");
    for name in ["mlp", "cnn", "encoder"] {
        let reference = &references["models"][name];
        let model = fixture_model(name, reference).to(device, None).unwrap();
        let revision = model.state().snapshot().revision();
        let capture = model
            .capture("x", fixture_tensor(&reference["cases"][0]["input"], device))
            .unwrap();
        let spec = capture
            .program()
            .signature()
            .inputs()
            .iter()
            .find(|input| input.name() == "x")
            .unwrap()
            .spec()
            .clone();
        #[cfg(feature = "testing")]
        let before = (TensorActivity::current(), CpuStorage::copy_stats());
        let built = model.build("x", spec).unwrap();
        #[cfg(feature = "testing")]
        {
            assert_builds_without_work(before.0, name);
            let copies = CpuStorage::copy_stats();
            assert_eq!(
                (copies.imports, copies.exports, copies.clones),
                (before.1.imports, before.1.exports, before.1.clones)
            );
        }
        assert_eq!(built.state().revision(), revision);
        assert_eq!(capture.state().revision(), revision);
        assert_eq!(
            built.program().fingerprint().unwrap(),
            capture.program().fingerprint().unwrap(),
            "{name}"
        );
        for (binding, tensor) in model.state().bindings() {
            assert_eq!(
                built.state().tensor(binding.name()).unwrap().id(),
                tensor.id()
            );
        }
        let mut metadata_executor = built.prepare(GraphPrepareOptions::new(device)).unwrap();
        let mut executor = capture.prepare(GraphPrepareOptions::new(device)).unwrap();
        let analysis = executor.analysis().clone();
        let mut retained = Vec::new();
        for case in reference["cases"].as_array().unwrap() {
            let input = fixture_tensor(&case["input"], device);
            assert_close(&model.forward(&input).unwrap(), &case["output"]);
            let metadata_output = metadata_executor
                .run(&HashMap::from([("x".into(), input.clone())]))
                .unwrap()
                .remove(0);
            assert_close(&metadata_output, &case["output"]);
            retained.push((metadata_output, case["output"].clone()));
            let output = executor
                .run(&HashMap::from([("x".into(), input)]))
                .unwrap()
                .remove(0);
            assert_close(&output, &case["output"]);
            retained.push((output, case["output"].clone()));
        }
        assert!(ptr::eq(analysis.ir(), executor.analysis().ir()));
        assert_eq!(model.state().snapshot().revision(), revision);
        for (output, expected) in retained {
            assert_close(&output, &expected);
        }
    }
}

#[test]
fn test_checkpoint_models_match_pytorch_through_eager_and_prepared_cpu() {
    exercise_models(Device::Cpu);
}

fn exercise_dynamic_encoder(device: Device) {
    use singe::graph::GraphExecutionMode;

    let references: Value = reference!("layers").resolved();
    let dynamic: Value = reference!("shapes").resolved();
    let model = fixture_model("encoder", &references["models"]["encoder"])
        .to(device, None)
        .unwrap();
    let spec = GraphInputSpec::new(ValueSpec::new(
        DataType::F32,
        Shape::create([2, 3, 4]).unwrap(),
    ))
    .with_dimension(0, "batch", Some(1), Some(4))
    .unwrap()
    .with_dimension(1, "sequence", Some(1), Some(8))
    .unwrap();
    #[cfg(feature = "testing")]
    let before = TensorActivity::current();
    let program = model.build("x", spec).unwrap();
    #[cfg(feature = "testing")]
    assert_builds_without_work(before, "encoder");
    let execution_modes = if device == Device::Cpu {
        vec![GraphExecutionMode::Compiled]
    } else {
        vec![GraphExecutionMode::Compiled, GraphExecutionMode::Eager]
    };
    for mode in execution_modes {
        let mut executor = program
            .prepare(
                GraphPrepareOptions::new(device)
                    .with_execution_mode(mode)
                    .with_cache_capacity(2),
            )
            .unwrap();
        let fingerprint = executor.specialization_key().program();
        assert!(
            executor
                .report()
                .unwrap()
                .contains("shape specialization: exact, metadata only")
        );
        let mut worker = executor.create_instance().unwrap();
        let case = &dynamic["cases"][2];
        let output = worker
            .run(&HashMap::from([(
                "x".into(),
                fixture_tensor(&case["input"], device),
            )]))
            .unwrap();
        assert_close(&output[0], &case["output"]);
        assert_eq!(executor.cache_stats().analysis_builds(), 1);
        let mut retained = Vec::new();
        for index in [0, 0, 1, 2, 3, 4] {
            let case = &dynamic["cases"][index];
            let input = fixture_tensor(&case["input"], device);
            #[cfg(feature = "testing")]
            let readbacks = TensorActivity::current().cuda_readbacks;
            let output = executor
                .run(&HashMap::from([("x".into(), input)]))
                .unwrap()
                .remove(0);
            #[cfg(feature = "testing")]
            assert_eq!(TensorActivity::current().cuda_readbacks, readbacks);
            assert_close(&output, &case["output"]);
            #[cfg(feature = "testing")]
            if device.is_cuda() {
                assert!(TensorActivity::current().cuda_readbacks > readbacks);
            }
            assert_eq!(executor.specialization_key().program(), fingerprint);
            assert!(executor.cache_stats().entries() <= 2);
            retained.push((output, case["output"].clone()));
        }
        assert_eq!(executor.cache_stats().analysis_builds(), 5);
        assert_eq!(executor.cache_stats().evictions(), 3);
        for dims in [[5, 3, 4], [2, 9, 4], [2, 3, 5], [0, 3, 4]] {
            let input = Tensor::full(device, Shape::create(dims).unwrap(), 0_f32).unwrap();
            #[cfg(feature = "testing")]
            let before = TensorActivity::current();
            assert!(executor.run(&HashMap::from([("x".into(), input)])).is_err());
            #[cfg(feature = "testing")]
            assert_eq!(TensorActivity::current(), before);
        }
        for (output, reference) in retained {
            assert_close(&output, &reference);
        }
    }
}

#[test]
fn test_native_dynamic_encoder_resolves_shapes_and_bounds_cache_on_cpu() {
    exercise_dynamic_encoder(Device::Cpu);
}

#[cfg(feature = "cuda")]
#[test]
fn test_native_dynamic_encoder_resolves_shapes_and_bounds_cache_on_cuda() {
    exercise_dynamic_encoder(Device::Cuda(0));
}

#[cfg(feature = "onnx")]
fn exercise_dynamic_onnx_encoder(device: Device) {
    use singe::{OnnxModel, OnnxOptions, graph::GraphExecutionMode};
    let dynamic: Value = reference!("shapes").resolved();
    let imported = OnnxModel::from_reader(
        include_bytes!("../references/shapes/encoder.onnx").as_slice(),
        OnnxOptions::new()
            .with_symbolic_dimension_range("batch", 2, Some(1), Some(4))
            .with_symbolic_dimension_range("sequence", 3, Some(1), Some(8)),
    )
    .unwrap();
    let modes = if device.is_cpu() {
        vec![GraphExecutionMode::Compiled]
    } else {
        vec![GraphExecutionMode::Compiled, GraphExecutionMode::Eager]
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
        for index in [0, 0, 1, 2, 3, 4] {
            let case = &dynamic["cases"][index];
            let input = fixture_tensor(&case["input"], device);
            #[cfg(feature = "testing")]
            let readbacks = TensorActivity::current().cuda_readbacks;
            let output = executor
                .run(&HashMap::from([("x".into(), input)]))
                .unwrap()
                .remove(0);
            #[cfg(feature = "testing")]
            assert_eq!(TensorActivity::current().cuda_readbacks, readbacks);
            assert_close(&output, &case["output"]);
            retained.push((output, case["output"].clone()));
            assert!(executor.cache_stats().entries() <= 2);
        }
        assert_eq!(executor.cache_stats().analysis_builds(), 5);
        assert_eq!(executor.cache_stats().evictions(), 3);
        for dimensions in [[5, 3, 4], [2, 9, 4], [0, 3, 4], [2, 3, 5]] {
            let input = Tensor::full(device, Shape::create(dimensions).unwrap(), 0_f32).unwrap();
            #[cfg(feature = "testing")]
            let before = TensorActivity::current();
            let cache = executor.cache_stats();
            assert!(executor.run(&HashMap::from([("x".into(), input)])).is_err());
            #[cfg(feature = "testing")]
            assert_eq!(TensorActivity::current(), before);
            assert_eq!(executor.cache_stats(), cache);
        }
        let mut independent = imported
            .prepare(GraphPrepareOptions::new(device).with_execution_mode(mode))
            .unwrap();
        let case = &dynamic["cases"][2];
        let output = independent
            .run(&HashMap::from([(
                "x".into(),
                fixture_tensor(&case["input"], device),
            )]))
            .unwrap();
        assert_close(&output[0], &case["output"]);
        for (output, expected) in retained {
            assert_close(&output, &expected);
        }
    }
}

#[cfg(feature = "onnx")]
#[test]
fn test_exported_dynamic_encoder_resolves_shape_values_on_cpu() {
    exercise_dynamic_onnx_encoder(Device::Cpu);
}

#[cfg(feature = "cuda")]
#[test]
fn test_checkpoint_models_match_pytorch_through_eager_and_native_cuda() {
    exercise_models(Device::Cuda(0));
}

fn tied_model() -> (Model<Sequential>, StateKey, StateKey) {
    let mut builder = StateBuilder::new();
    let first = builder
        .parameter("first.weight", tensor(&[2, 2], vec![1., 0., 0., 2.]))
        .unwrap();
    let second = builder.tie("second.weight", &first).unwrap();
    builder
        .buffer(
            "steps",
            Tensor::from_vec(Device::Cpu, Shape::scalar(), vec![7_i64]).unwrap(),
        )
        .unwrap();
    builder
        .buffer(
            "enabled",
            Tensor::from_vec(Device::Cpu, Shape::scalar(), vec![true]).unwrap(),
        )
        .unwrap();
    let model = Model::new(
        Sequential::new()
            .with(Linear::new(first.clone(), None))
            .with(Linear::new(second.clone(), None)),
        builder.finish().unwrap(),
    );
    (model, first, second)
}

fn exercise_state_conversion(device: Device) {
    let (model, first, second) = tied_model();
    let state = model
        .state()
        .load_safetensors_bytes(&model.state().safetensors_bytes().unwrap())
        .unwrap();
    for dtype in [DataType::F16, DataType::Bf16, DataType::F32, DataType::F64] {
        let converted = state.to(device, Some(dtype)).unwrap();
        assert_eq!(converted.tensor(&first).unwrap().data_type(), dtype);
        assert_eq!(
            converted.tensor(&first).unwrap().id(),
            converted.tensor(&second).unwrap().id()
        );
        assert_eq!(
            converted
                .named_tensor("steps")
                .unwrap()
                .to_vec::<i64>()
                .unwrap(),
            [7]
        );
        assert_eq!(
            converted
                .named_tensor("enabled")
                .unwrap()
                .to_vec::<bool>()
                .unwrap(),
            [true]
        );
    }
    assert!(state.to(Device::Cpu, Some(DataType::I64)).is_err());
}

#[test]
fn test_ties_survive_checkpoint_and_float_conversion_without_converting_integral_buffers() {
    exercise_state_conversion(Device::Cpu);
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_movement_and_float_conversion_preserve_ties_and_integral_buffers() {
    exercise_state_conversion(Device::Cuda(0));
}

fn exercise_replacement(device: Device) {
    let (model, first, second) = tied_model();
    let model = model.to(device, None).unwrap();
    let input = tensor(&[1, 2], vec![2., 3.]).to_device(device).unwrap();
    let capture = model.capture("x", input.clone()).unwrap();
    let mut old = capture.prepare(GraphPrepareOptions::new(device)).unwrap();
    let replacement = tensor(&[2, 2], vec![3., 0., 0., 4.])
        .to_device(device)
        .unwrap();
    let new_state = model
        .state()
        .replace(&HashMap::from([("first.weight".into(), replacement)]))
        .unwrap();
    assert_eq!(
        new_state.tensor(&first).unwrap().id(),
        new_state.tensor(&second).unwrap().id()
    );
    let bytes = new_state.safetensors_bytes().unwrap();
    let loaded = model.state().load_safetensors_bytes(&bytes).unwrap();
    let new_model = model.with_state(loaded.clone()).unwrap();
    let bindings = HashMap::from([("x".into(), input.clone())]);
    assert_eq!(
        old.run(&bindings).unwrap()[0].to_vec::<f32>().unwrap(),
        [2., 12.]
    );
    assert_eq!(
        new_model.forward(&input).unwrap().to_vec::<f32>().unwrap(),
        [18., 48.]
    );
    old.rebind_state(&capture.bind_state(&loaded).unwrap())
        .unwrap();
    assert_eq!(
        old.run(&bindings).unwrap()[0].to_vec::<f32>().unwrap(),
        [18., 48.]
    );
    assert_eq!(
        model.forward(&input).unwrap().to_vec::<f32>().unwrap(),
        [2., 12.]
    );
    assert_eq!(
        capture
            .program()
            .state_schema()
            .find(|binding| binding.name() == first.name())
            .unwrap()
            .id(),
        capture
            .program()
            .state_schema()
            .find(|binding| binding.name() == second.name())
            .unwrap()
            .id()
    );
}

#[test]
fn test_replacement_and_graph_rebinding_preserve_old_cpu_snapshots() {
    exercise_replacement(Device::Cpu);
}

#[test]
fn test_invalid_names_foreign_keys_and_conflicting_state_fail_before_publication() {
    let mut builder = StateBuilder::new();
    let key = builder.parameter("a", tensor(&[1], vec![1.])).unwrap();
    assert!(builder.parameter("a", tensor(&[1], vec![1.])).is_err());
    assert!(builder.parameter("a..b", tensor(&[1], vec![1.])).is_err());
    assert!(StateBuilder::new().tie("b", &key).is_err());
    let (model, _, _) = tied_model();
    assert!(model.state().tensor(&key).is_err());
    let before = model.state().snapshot().revision();
    assert!(
        model
            .state()
            .replace(&HashMap::from([
                ("first.weight".into(), tensor(&[2, 2], vec![1.; 4])),
                ("second.weight".into(), tensor(&[2, 2], vec![2.; 4])),
            ]))
            .is_err()
    );
    assert_eq!(model.state().snapshot().revision(), before);
    let empty = StateBuilder::new()
        .finish()
        .unwrap()
        .safetensors_bytes()
        .unwrap();
    assert!(
        matches!(model.state().load_safetensors_bytes(&empty), Err(Error::Checkpoint(CheckpointError::Keys { missing, unexpected })) if missing.len() == 4 && unexpected.is_empty())
    );
    assert!(model.state().load_safetensors_bytes(b"broken").is_err());
}

#[test]
fn test_checkpoint_conflicts_metadata_errors_and_io_are_checked_before_replacement() {
    let (model, _, _) = tied_model();
    let original = model.state();
    let before = original.snapshot().revision();
    for (shape, dtype, conflict) in [
        (vec![2, 2], DataType::F32, true),
        (vec![4], DataType::F32, false),
        (vec![2, 2], DataType::F64, false),
    ] {
        let mut builder = StateBuilder::new();
        for name in ["first.weight", "second.weight"] {
            let value = if conflict && name == "second.weight" {
                2.0_f32
            } else {
                1.0
            };
            let tensor = Tensor::full(Device::Cpu, Shape::create(shape.clone()).unwrap(), value)
                .unwrap()
                .cast(dtype)
                .unwrap();
            builder.parameter(name, tensor).unwrap();
        }
        builder
            .buffer(
                "steps",
                Tensor::from_vec(Device::Cpu, Shape::scalar(), vec![7_i64]).unwrap(),
            )
            .unwrap();
        builder
            .buffer(
                "enabled",
                Tensor::from_vec(Device::Cpu, Shape::scalar(), vec![true]).unwrap(),
            )
            .unwrap();
        let bad = builder.finish().unwrap().safetensors_bytes().unwrap();
        assert!(original.load_safetensors_bytes(&bad).is_err());
        assert_eq!(original.snapshot().revision(), before);
    }
    let mut builder = StateBuilder::new();
    builder
        .parameter("unexpected", tensor(&[1], vec![1.]))
        .unwrap();
    let bad = builder.finish().unwrap().safetensors_bytes().unwrap();
    assert!(
        matches!(original.load_safetensors_bytes(&bad), Err(Error::Checkpoint(CheckpointError::Keys { missing, unexpected })) if missing.len() == 4 && unexpected == ["unexpected"])
    );
    // A flashpack file stores each tie once, and its scalars and booleans as they are.
    for extension in ["safetensors", "flashpack"] {
        let path = env::temp_dir().join(format!(
            "singe-module-checkpoint-{}.{extension}",
            process::id()
        ));
        if extension == "flashpack" {
            original.save_flashpack(&path).unwrap();
        } else {
            original.save_safetensors(&path).unwrap();
        }
        let loaded = original.load_file(&path).unwrap();
        fs::remove_file(path).unwrap();
        assert_eq!(
            loaded
                .named_tensor("steps")
                .unwrap()
                .to_vec::<i64>()
                .unwrap(),
            [7]
        );
        assert_eq!(
            loaded
                .named_tensor("enabled")
                .unwrap()
                .to_vec::<bool>()
                .unwrap(),
            [true]
        );
        assert_eq!(
            loaded.named_tensor("first.weight").unwrap().id(),
            loaded.named_tensor("second.weight").unwrap().id()
        );
    }
}

fn exercise_embedding(device: Device) {
    use singe::nn::Embedding;
    let source = tensor(&[3, 4], (0..12).map(|value| value as f32).collect())
        .to_device(device)
        .unwrap();
    let mut builder = StateBuilder::new();
    let weight = builder
        .parameter("embedding.weight", source.transpose(0, 1).unwrap())
        .unwrap();
    let model = Model::new(Embedding::new(weight), builder.finish().unwrap());
    let indices =
        |values: Vec<i64>| Tensor::from_vec(device, Shape::create([3]).unwrap(), values).unwrap();
    let capture = model.capture("tokens", indices(vec![0, 3, 1])).unwrap();
    let mut executor = capture.prepare(GraphPrepareOptions::new(device)).unwrap();
    let analysis = executor.analysis().clone();
    for (tokens, expected) in [
        (vec![0, 3, 1], vec![0., 4., 8., 3., 7., 11., 1., 5., 9.]),
        (vec![2, 1, 2], vec![2., 6., 10., 1., 5., 9., 2., 6., 10.]),
    ] {
        let input = indices(tokens);
        assert_eq!(
            model.forward(&input).unwrap().to_vec::<f32>().unwrap(),
            expected
        );
        assert_eq!(
            executor
                .run(&HashMap::from([("tokens".into(), input)]))
                .unwrap()[0]
                .to_vec::<f32>()
                .unwrap(),
            expected
        );
    }
    for tokens in [vec![-1, 0, 1], vec![0, 4, 1]] {
        let input = indices(tokens);
        assert!(matches!(
            model.forward(&input),
            Err(Error::IndexOutOfBounds { .. })
        ));
        assert!(matches!(
            executor.run(&HashMap::from([("tokens".into(), input)])),
            Err(Error::IndexOutOfBounds { .. })
        ));
    }
    let input = indices(vec![0, 1, 2]);
    assert_eq!(
        executor
            .run(&HashMap::from([("tokens".into(), input)]))
            .unwrap()[0]
            .to_vec::<f32>()
            .unwrap(),
        [0., 4., 8., 1., 5., 9., 2., 6., 10.]
    );
    assert!(ptr::eq(analysis.ir(), executor.analysis().ir()));
}

#[test]
fn test_cpu_embedding_capture_reuses_analysis_checks_bounds_and_accepts_strided_weights() {
    exercise_embedding(Device::Cpu);
}

#[test]
fn test_allocation_sharing_does_not_declare_ties_and_checkpoint_may_store_one_name_per_tie() {
    let tensor = tensor(&[2, 2], vec![1., 0., 0., 2.]);
    let mut builder = StateBuilder::new();
    let first = builder.parameter("first.weight", tensor.clone()).unwrap();
    let second = builder.parameter("second.weight", tensor).unwrap();
    let state = builder.finish().unwrap();
    assert_ne!(
        state.tensor(&first).unwrap().id(),
        state.tensor(&second).unwrap().id()
    );
    let changed = state
        .replace(&HashMap::from([(
            "first.weight".into(),
            Tensor::zeroed_on(Device::Cpu, DataType::F32, Shape::create([2, 2]).unwrap()).unwrap(),
        )]))
        .unwrap();
    assert_eq!(
        changed.tensor(&second).unwrap().to_vec::<f32>().unwrap(),
        [1., 0., 0., 2.]
    );
    let (model, first, second) = tied_model();
    let mut builder = StateBuilder::new();
    builder
        .parameter(
            "first.weight",
            model.state().tensor(&first).unwrap().clone(),
        )
        .unwrap();
    builder
        .buffer(
            "steps",
            model.state().named_tensor("steps").unwrap().clone(),
        )
        .unwrap();
    builder
        .buffer(
            "enabled",
            model.state().named_tensor("enabled").unwrap().clone(),
        )
        .unwrap();
    let bytes = builder.finish().unwrap().safetensors_bytes().unwrap();
    let loaded = model.state().load_safetensors_bytes(&bytes).unwrap();
    assert_eq!(
        loaded.tensor(&first).unwrap().id(),
        loaded.tensor(&second).unwrap().id()
    );
}

#[cfg(feature = "cuda")]
#[test]
fn test_optimized_native_normalization_is_available_without_cudnn() {
    use singe::{
        backend::cuda::{CudaGraphProvider, LoweringPolicy},
        graph::{CaptureBindings, Graph, GraphCudaOptions},
        nn::{Layer, RmsNorm},
    };
    let device = Device::Cuda(0);
    let mut builder = StateBuilder::new();
    let weight = builder
        .parameter(
            "weight",
            tensor(&[2], vec![1., 2.]).to_device(device).unwrap(),
        )
        .unwrap();
    let bias = builder
        .parameter(
            "bias",
            tensor(&[2], vec![0.5, -0.25]).to_device(device).unwrap(),
        )
        .unwrap();
    let mean = builder
        .buffer(
            "mean",
            tensor(&[2], vec![0.25, -0.5]).to_device(device).unwrap(),
        )
        .unwrap();
    let variance = builder
        .buffer(
            "variance",
            tensor(&[2], vec![0.5, 2.]).to_device(device).unwrap(),
        )
        .unwrap();
    let state = builder.finish().unwrap();
    let layers: Vec<Box<dyn Layer>> = vec![
        Box::new(LayerNorm::new(
            vec![2],
            Some(weight.clone()),
            Some(bias.clone()),
            1e-5,
        )),
        Box::new(RmsNorm::new(vec![2], Some(weight.clone()), 1e-5)),
        Box::new(BatchNorm::new(
            Some(weight),
            Some(bias),
            mean,
            variance,
            1e-5,
        )),
    ];
    let input = tensor(&[2, 2], vec![1., 2., 3., 5.])
        .to_device(device)
        .unwrap();
    let config = GraphPrepareOptions::new(device).with_cuda_options(
        GraphCudaOptions::new()
            .with_lowering_policy(LoweringPolicy::new(vec![CudaGraphProvider::Kernel])),
    );
    for layer in layers {
        let expected = layer
            .forward(&input, &state)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let (capture, _) = Graph::capture_with_bindings(
            [("x", input.clone())],
            CaptureBindings::new()
                .with_parameter("weight", state.named_tensor("weight").unwrap().clone())
                .with_parameter("bias", state.named_tensor("bias").unwrap().clone())
                .with_buffer("mean", state.named_tensor("mean").unwrap().clone())
                .with_buffer("variance", state.named_tensor("variance").unwrap().clone()),
            |inputs| Ok(vec![layer.forward(&inputs[0], &state)?]),
        )
        .unwrap();
        let mut executor = capture.prepare(config.clone()).unwrap();
        let actual = executor
            .run(&HashMap::from([("x".into(), input.clone())]))
            .unwrap()[0]
            .to_vec::<f32>()
            .unwrap();
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5);
        }
        assert!(executor.report().unwrap().contains("Kernel"));
    }
}

fn exercise_empty_embedding(device: Device) {
    use singe::nn::Embedding;
    for (rows, width, tokens) in [(4, 3, vec![]), (4, 0, vec![0_i64, 3]), (0, 3, vec![])] {
        let mut builder = StateBuilder::new();
        let weight = builder
            .parameter(
                "weight",
                Tensor::zeroed_on(device, DataType::F32, Shape::create([rows, width]).unwrap())
                    .unwrap(),
            )
            .unwrap();
        let model = Model::new(Embedding::new(weight), builder.finish().unwrap());
        let count = i64::try_from(tokens.len()).unwrap();
        let input = Tensor::from_vec(device, Shape::create([count]).unwrap(), tokens).unwrap();
        let eager = model.forward(&input).unwrap();
        assert_eq!(eager.shape().dimensions(), [count, width]);
        assert_eq!(eager.to_vec::<f32>().unwrap(), [] as [f32; 0]);
        let mut executor = model
            .prepare("tokens", input.clone(), GraphPrepareOptions::new(device))
            .unwrap();
        let output = executor
            .run(&HashMap::from([("tokens".into(), input)]))
            .unwrap();
        assert_eq!(output[0].shape().dimensions(), [count, width]);
        assert_eq!(output[0].to_vec::<f32>().unwrap(), [] as [f32; 0]);
    }
}

#[test]
fn test_empty_embedding_shapes_are_valid_on_cpu() {
    exercise_empty_embedding(Device::Cpu);
}

#[test]
fn test_workers_share_one_module_description_and_immutable_checkpoint_state() {
    use std::thread;
    let references: Value = reference!("layers").resolved();
    let reference = &references["models"]["mlp"];
    let model = fixture_model("mlp", reference);
    let revision = model.state().snapshot().revision();
    thread::scope(|scope| {
        let workers = (0..4)
            .map(|index| {
                let model = &model;
                let case = &reference["cases"][index % 2];
                scope.spawn(move || {
                    for _ in 0..3 {
                        let input = fixture_tensor(&case["input"], Device::Cpu);
                        assert_close(&model.forward(&input).unwrap(), &case["output"]);
                    }
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }
    });
    assert_eq!(model.state().snapshot().revision(), revision);
}

#[test]
fn test_replacing_independent_parameters_with_one_tensor_keeps_capture_unambiguous() {
    let mut builder = StateBuilder::new();
    let first = builder
        .parameter("first", tensor(&[2, 2], vec![1., 0., 0., 2.]))
        .unwrap();
    let second = builder
        .parameter("second", tensor(&[2, 2], vec![1., 0., 0., 2.]))
        .unwrap();
    let model = Model::new(
        Sequential::new()
            .with(Linear::new(first, None))
            .with(Linear::new(second, None)),
        builder.finish().unwrap(),
    );
    let value = tensor(&[2, 2], vec![2., 0., 0., 3.]);
    let state = model
        .state()
        .replace(&HashMap::from([
            ("first".into(), value.clone()),
            ("second".into(), value),
        ]))
        .unwrap();
    let updated = model.with_state(state).unwrap();
    let input = tensor(&[1, 2], vec![1., 1.]);
    let mut executor = updated
        .prepare("x", input.clone(), GraphPrepareOptions::new(Device::Cpu))
        .unwrap();
    assert_eq!(
        executor
            .run(&HashMap::from([("x".into(), input.clone())]))
            .unwrap()[0]
            .to_vec::<f32>()
            .unwrap(),
        [4., 9.]
    );
    let changed = updated
        .with_state(
            updated
                .state()
                .replace(&HashMap::from([(
                    "first".into(),
                    tensor(&[2, 2], vec![5., 0., 0., 6.]),
                )]))
                .unwrap(),
        )
        .unwrap();
    assert_eq!(
        changed.forward(&input).unwrap().to_vec::<f32>().unwrap(),
        [10., 18.]
    );
    assert_eq!(
        updated.forward(&input).unwrap().to_vec::<f32>().unwrap(),
        [4., 9.]
    );
    let shared = updated.state().named_tensor("second").unwrap().clone();
    let single = updated
        .with_state(
            updated
                .state()
                .replace(&HashMap::from([("first".into(), shared)]))
                .unwrap(),
        )
        .unwrap();
    single
        .prepare("x", input, GraphPrepareOptions::new(Device::Cpu))
        .unwrap();
}

fn checkpoint_directory(name: &str) -> PathBuf {
    let directory = env::temp_dir().join(format!("singe-checkpoint-{}-{name}", process::id()));
    let _ = fs::remove_dir_all(&directory);
    fs::create_dir_all(&directory).unwrap();
    directory
}

// A tied schema on metadata, whose weights a checkpoint in another type supplies.
fn checkpoint_schema(data_type: DataType) -> ModelState {
    let mut builder = StateBuilder::new();
    let meta = |dimensions: &[i64]| {
        Tensor::zeroed_on(
            Device::Meta,
            data_type,
            Shape::create(dimensions.to_vec()).unwrap(),
        )
        .unwrap()
    };
    let first = builder.parameter("first", meta(&[2])).unwrap();
    builder.tie("alias", &first).unwrap();
    builder.parameter("second", meta(&[1])).unwrap();
    builder.finish().unwrap()
}

fn write_shard(path: &Path, tensors: &[(&str, Vec<f32>)]) {
    let mut builder = StateBuilder::new();
    for (name, values) in tensors {
        let length = i64::try_from(values.len()).unwrap();
        builder
            .parameter(*name, tensor(&[length], values.clone()))
            .unwrap();
    }
    builder.finish().unwrap().save_safetensors(path).unwrap();
}

#[test]
fn test_checkpoints_read_shards_cast_to_the_schema_and_map_names() {
    let directory = checkpoint_directory("shards");
    write_shard(
        &directory.join("a.safetensors"),
        &[("first", vec![1., 2.]), ("stale", vec![9.])],
    );
    write_shard(&directory.join("b.safetensors"), &[("renamed", vec![3.])]);
    let checkpoint = Checkpoint::open([
        directory.join("a.safetensors"),
        directory.join("b.safetensors"),
    ])
    .unwrap();
    // The plan leaves `stale` unread.
    let mut plan = LoadPlan::new();
    plan.insert("first", TensorSource::Stored("first".to_owned()))
        .unwrap();
    plan.insert("second", TensorSource::Stored("renamed".to_owned()))
        .unwrap();
    let options = CheckpointOptions {
        device: Some(Device::Cpu),
        convert: true,
    };
    let state = checkpoint_schema(DataType::F64)
        .load_checkpoint(&checkpoint, &plan, options)
        .unwrap();
    assert_eq!(state.device().unwrap(), Device::Cpu);
    assert_eq!(state.floating_type().unwrap(), DataType::F64);
    assert_eq!(
        state
            .named_tensor("alias")
            .unwrap()
            .to_vec::<f64>()
            .unwrap(),
        [1., 2.]
    );
    assert_eq!(
        state
            .named_tensor("second")
            .unwrap()
            .to_vec::<f64>()
            .unwrap(),
        [3.]
    );
    // The checkpoint's own names are unexpected and the schema's are missing.
    let identity = LoadPlan::identity(&checkpoint);
    assert!(matches!(
        checkpoint_schema(DataType::F64).load_checkpoint(&checkpoint, &identity, options),
        Err(Error::Checkpoint(CheckpointError::Keys { .. }))
    ));
    // A plan may not read what the checkpoint lacks.
    let mut absent = plan.clone();
    absent
        .insert("alias", TensorSource::Stored("gone".to_owned()))
        .unwrap();
    assert!(matches!(
        checkpoint_schema(DataType::F64).load_checkpoint(&checkpoint, &absent, options),
        Err(Error::Checkpoint(CheckpointError::Absent { .. }))
    ));
    // Without conversion a checkpoint must already be in the schema's type.
    assert!(
        checkpoint_schema(DataType::F64)
            .load_checkpoint(
                &checkpoint,
                &plan,
                CheckpointOptions {
                    convert: false,
                    ..options
                }
            )
            .is_err()
    );
    fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn test_checkpoints_refuse_ties_that_differ_across_shards() {
    let directory = checkpoint_directory("ties");
    write_shard(
        &directory.join("a.safetensors"),
        &[("first", vec![1., 2.]), ("second", vec![3.])],
    );
    write_shard(&directory.join("b.safetensors"), &[("alias", vec![1., 5.])]);
    let checkpoint = Checkpoint::open([
        directory.join("a.safetensors"),
        directory.join("b.safetensors"),
    ])
    .unwrap();
    let plan = LoadPlan::identity(&checkpoint);
    let options = CheckpointOptions {
        device: Some(Device::Cpu),
        convert: false,
    };
    assert!(
        checkpoint_schema(DataType::F32)
            .load_checkpoint(&checkpoint, &plan, options)
            .is_err()
    );
    fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn test_replacing_state_refuses_another_builders() {
    let (model, _, _) = tied_model();
    let (other, _, _) = tied_model();
    assert!(model.with_state(other.state().clone()).is_err());
    assert!(model.with_state(model.state().clone()).is_ok());
}

// A checkpoint may hold a state tensor in pieces, as DeepSeek-V3's holds each expert, which must cover it once.
#[test]
fn test_checkpoints_assemble_a_tensor_from_pieces_and_refuse_a_bad_cover() {
    let values = |shape: &[i64], first: f32| {
        let count = shape.iter().product::<i64>();
        let values = (0..count)
            .map(|index| first + f32::from(i16::try_from(index).unwrap()))
            .collect::<Vec<_>>();
        Tensor::from_vec(Device::Cpu, Shape::create(shape.to_vec()).unwrap(), values).unwrap()
    };
    let mut builder = StateBuilder::new();
    let packed = builder
        .parameter("packed", values(&[2, 2, 3], 0.0))
        .unwrap();
    let state = builder.finish().unwrap();
    let origin = |name: &str| match name {
        "a" => [0, 0, 0],
        "b" | "again" => [0, 1, 0],
        "c" => [1, 0, 0],
        _ => [1, 2, 0],
    };
    let load = |tensors: Vec<(&str, Tensor)>| {
        let mut builder = StateBuilder::new();
        let mut plan = LoadPlan::new();
        for (name, tensor) in tensors {
            builder.parameter(name, tensor).unwrap();
            plan.insert(
                "packed",
                TensorSource::Pieces(vec![StatePiece {
                    name: name.to_owned(),
                    origin: origin(name).to_vec(),
                }]),
            )
            .unwrap();
        }
        let bytes = builder.finish().unwrap().safetensors_bytes().unwrap();
        let checkpoint = Checkpoint::from_bytes(&bytes).unwrap();
        state.load_checkpoint(&checkpoint, &plan, CheckpointOptions::default())
    };
    let loaded = load(vec![
        // A piece may leave out leading axes of extent one.
        ("a", values(&[1, 3], 100.0)),
        ("b", values(&[1, 1, 3], 200.0)),
        ("c", values(&[1, 2, 3], 300.0)),
    ])
    .unwrap();
    assert_eq!(
        loaded.tensor(&packed).unwrap().to_vec::<f32>().unwrap(),
        [
            100.0, 101.0, 102.0, 200.0, 201.0, 202.0, 300.0, 301.0, 302.0, 303.0, 304.0, 305.0
        ]
    );
    // A row left unset, a row set twice, and a row past the end.
    let partial = vec![
        ("a", values(&[1, 1, 3], 0.0)),
        ("c", values(&[1, 2, 3], 0.0)),
    ];
    assert!(load(partial).is_err());
    let twice = vec![
        ("a", values(&[1, 1, 3], 0.0)),
        ("b", values(&[1, 1, 3], 0.0)),
        ("again", values(&[1, 1, 3], 0.0)),
        ("c", values(&[1, 2, 3], 0.0)),
    ];
    assert!(load(twice).is_err());
    let outside = vec![("outside", values(&[1, 1, 3], 0.0))];
    assert!(load(outside).is_err());
}
