#![cfg(feature = "onnx")]

use std::collections::HashMap;

use safetensors::{Dtype, SafeTensors};
use serde_json::Value;
use singe::{
    DataType, Device, OnnxModel, OnnxOptions, Result, Shape, Tensor,
    backend::registry::{Operation, OperationAttributes, ReductionAttributes},
    dispatch::call_one,
    graph::{Determinism, GraphExecutionMode, GraphInputSpec, GraphPrepareOptions},
    nn::{
        Embedding, EncoderBlock, Layer, LayerNorm, Linear, Model, ModelState, SelfAttention,
        Sequential, StateBuilder,
    },
    onnx,
    value::ValueSpec,
};

#[derive(Debug)]
struct Pool;

impl Layer for Pool {
    fn forward(&self, input: &Tensor, _state: &ModelState) -> Result<Tensor> {
        call_one(
            Operation::Mean,
            OperationAttributes::Reduction(ReductionAttributes {
                dimensions: vec![1],
                keepdim: false,
            }),
            &[input],
        )
    }
}

#[derive(Debug)]
struct FloatLogits;

impl Layer for FloatLogits {
    fn forward(&self, input: &Tensor, _state: &ModelState) -> Result<Tensor> {
        call_one(
            Operation::Cast,
            OperationAttributes::Cast(DataType::F32),
            &[input],
        )
    }
}

fn model(name: &str) -> Model<Sequential> {
    let checkpoint = match name {
        "f32" => include_bytes!("../references/set_encoder/f32.safetensors").as_slice(),
        "f16" => include_bytes!("../references/set_encoder/f16.safetensors").as_slice(),
        "bf16" => include_bytes!("../references/set_encoder/bf16.safetensors").as_slice(),
        _ => unreachable!(),
    };
    let source = SafeTensors::deserialize(checkpoint).unwrap();
    let mut builder = StateBuilder::new();
    let mut keys = HashMap::new();
    for (name, value) in source.tensors() {
        let shape =
            Shape::create(value.shape().iter().map(|&dim| i64::try_from(dim).unwrap())).unwrap();
        let dtype = match value.dtype() {
            Dtype::F32 => DataType::F32,
            Dtype::F16 => DataType::F16,
            Dtype::BF16 => DataType::Bf16,
            dtype => panic!("unexpected checkpoint dtype {dtype:?}"),
        };
        let value = Tensor::from_bytes(Device::Cpu, dtype, shape, value.data()).unwrap();
        keys.insert(name.clone(), builder.parameter(name, value).unwrap());
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
    let module = Sequential::new()
        .with(Embedding::new(key("embedding.weight")))
        .with(EncoderBlock::new(
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
        ))
        .with(Pool)
        .with(linear("head"))
        .with(FloatLogits);
    Model::new(module, builder.finish().unwrap())
}

fn input(case: &Value, device: Device) -> Tensor {
    Tensor::from_vec(
        device,
        Shape::create(
            case["input"]["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|dim| dim.as_i64().unwrap()),
        )
        .unwrap(),
        case["input"]["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_i64().unwrap())
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

fn compare(actual: &Tensor, expected: &Value, atol: f32, rtol: f32) {
    assert_eq!(
        actual.shape().dimensions(),
        expected["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|dim| dim.as_i64().unwrap())
            .collect::<Vec<_>>()
    );
    for (actual, expected) in actual
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .zip(expected["values"].as_array().unwrap())
    {
        let expected = reference_f32(expected);
        assert!(
            (actual - expected).abs() <= rtol.mul_add(expected.abs(), atol),
            "actual {actual}, expected {expected}"
        );
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "references are float32 values stored as f64 JSON numbers"
)]
fn reference_f32(value: &Value) -> f32 {
    value.as_f64().unwrap() as f32
}

fn exercise(name: &str, device: Device) {
    let references: Value = singe_testing::reference!("set_encoder").resolved();
    let reference = &references["models"][name];
    let cases = &reference["cases"];
    let atol = reference_f32(&reference["atol"]);
    let rtol = reference_f32(&reference["rtol"]);
    let native = model(name)
        .build(
            "x",
            GraphInputSpec::new(ValueSpec::new(
                DataType::I64,
                Shape::create([2, 3]).unwrap(),
            ))
            .with_dimension(0, "batch", Some(1), Some(4))
            .unwrap()
            .with_dimension(1, "sequence", Some(1), Some(8))
            .unwrap(),
        )
        .unwrap();
    let imported = OnnxModel::from_reader(
        match name {
            "f32" => include_bytes!("../references/set_encoder/f32.onnx").as_slice(),
            "f16" => include_bytes!("../references/set_encoder/f16.onnx").as_slice(),
            "bf16" => include_bytes!("../references/set_encoder/bf16.onnx").as_slice(),
            _ => unreachable!(),
        },
        OnnxOptions::new()
            .with_symbolic_dimension_range("batch", 2, Some(1), Some(4))
            .with_symbolic_dimension_range("sequence", 3, Some(1), Some(8)),
    )
    .unwrap();
    let exported_bytes = onnx::export(native.program(), native.state()).unwrap();
    let exported = OnnxModel::from_reader(exported_bytes.as_slice(), OnnxOptions::new()).unwrap();
    if device == Device::Cpu {
        record_export(name, &exported_bytes, reference);
    }
    for mode in [GraphExecutionMode::Eager, GraphExecutionMode::Compiled] {
        let config = GraphPrepareOptions::new(device)
            .with_execution_mode(mode)
            .with_determinism(Determinism::Require)
            .with_cache_capacity(2);
        let mut native = native.prepare(config.clone()).unwrap();
        let mut imported = imported.prepare(config.clone()).unwrap();
        let mut exported = exported.prepare(config).unwrap();
        let mut retained = Vec::new();
        let mut repeated: [Option<Vec<u8>>; 3] = [None, None, None];
        for index in [0, 0, 1, 2, 3, 4] {
            let case = &cases[index];
            let inputs = HashMap::from([("x".into(), input(case, device))]);
            for (worker, executor) in [&mut native, &mut imported, &mut exported]
                .into_iter()
                .enumerate()
            {
                let output = executor.run(&inputs).unwrap().remove(0);
                if index == 0 {
                    let bytes = output.to_bytes().unwrap();
                    if let Some(previous) = &repeated[worker] {
                        assert_eq!(&bytes, previous);
                    }
                    repeated[worker] = Some(bytes);
                }
                compare(&output, &case["output"], atol, rtol);
                compare(&output, &case["fp32_accumulation_reference"], atol, rtol);
                retained.push((output, case));
            }
        }
        for executor in [&native, &imported, &exported] {
            assert_eq!(executor.cache_stats().analysis_builds(), 5);
            assert_eq!(executor.cache_stats().evictions(), 3);
        }
        for (output, case) in retained {
            compare(&output, &case["output"], atol, rtol);
            compare(&output, &case["fp32_accumulation_reference"], atol, rtol);
        }
    }
}

fn record_export(name: &str, bytes: &[u8], reference: &Value) {
    use std::{env, fs, path::PathBuf};
    if let Some(directory) = env::var_os("SINGE_ONNX_EXPORT_DIR") {
        let directory = PathBuf::from(directory);
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join(format!("set_encoder_{name}_cpu"));
        fs::write(path.with_extension("onnx"), bytes).unwrap();
        let reference = serde_json::json!({"cases": reference["cases"], "input_dtype": "int64", "atol": reference["atol"], "rtol": reference["rtol"], "reference_evaluator": name != "bf16"});
        fs::write(
            path.with_extension("json"),
            serde_json::to_vec_pretty(&reference).unwrap(),
        )
        .unwrap();
    }
}

#[test]
fn test_categorical_set_encoder_matches_pytorch_and_onnx_on_cpu() {
    exercise("f32", Device::Cpu);
}

#[test]
fn test_reduced_precision_models_build_fingerprint_and_prepare_on_the_cpu() {
    #[cfg(feature = "testing")]
    use singe::testing::TensorActivity;
    // Reading the reference makes tensors, so it comes before the count of storage requests.
    let reference: Value = singe_testing::reference!("set_encoder").resolved();
    for name in ["f16", "bf16"] {
        let model = model(name);
        #[cfg(feature = "testing")]
        let before = TensorActivity::current().storage_requests;
        let program = model
            .build(
                "x",
                GraphInputSpec::new(ValueSpec::new(
                    DataType::I64,
                    Shape::create([2, 3]).unwrap(),
                ))
                .with_dimension(0, "batch", Some(1), Some(4))
                .unwrap()
                .with_dimension(1, "sequence", Some(1), Some(8))
                .unwrap(),
            )
            .unwrap();
        program.program().fingerprint().unwrap();
        let bytes = onnx::export(program.program(), program.state()).unwrap();
        record_export(name, &bytes, &reference["models"][name]);
        #[cfg(feature = "testing")]
        assert_eq!(TensorActivity::current().storage_requests, before);
        // The CPU computes half-precision neural operations in `F32`.
        assert!(
            program
                .prepare(GraphPrepareOptions::new(Device::Cpu))
                .is_ok()
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_categorical_set_encoder_matches_pytorch_and_onnx_on_cuda() {
    exercise("f32", Device::Cuda(0));
}

#[cfg(feature = "cuda")]
#[test]
fn test_complete_fp16_set_encoder_matches_quantized_and_fp32_references_on_cuda() {
    exercise("f16", Device::Cuda(0));
}

#[cfg(feature = "cuda")]
#[test]
fn test_complete_bf16_set_encoder_matches_quantized_and_fp32_references_on_cuda() {
    exercise("bf16", Device::Cuda(0));
}
