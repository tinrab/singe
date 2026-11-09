#![cfg(feature = "onnx")]

use std::{collections::HashMap, env, fs, path::Path};

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Device, OnnxModel, OnnxOptions, Shape, Tensor,
    error::Error,
    graph::{GraphExecutionMode, GraphPrepareOptions},
    onnx::Error as OnnxError,
};

#[derive(Deserialize)]
struct EncodedTensor {
    dtype: String,
    shape: Vec<i64>,
    bytes: String,
}

#[derive(Deserialize)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "each flag is an independent field of the conformance fixture format"
)]
struct Case {
    name: String,
    model: String,
    inputs: HashMap<String, EncodedTensor>,
    outputs: Vec<EncodedTensor>,
    #[serde(default)]
    check_zero_sign: bool,
    #[serde(default)]
    required: bool,
    #[serde(default)]
    required_cuda: bool,
    #[serde(default)]
    exact_storage: bool,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
struct Support {
    name: String,
    importable: bool,
    eager: bool,
    compiled: bool,
}

fn decode(hex: &str) -> Vec<u8> {
    assert_eq!(hex.len() % 2, 0);
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect()
}

fn unsupported(error: &Error) -> bool {
    match error {
        Error::Unsupported { .. } => true,
        Error::GraphSource { source, .. } => unsupported(source),
        _ => false,
    }
}

fn record_rejection(case: &Case, device: Device, mode: &str, stage: &str, error: &impl ToString) {
    eprintln!(
        "{}",
        serde_json::json!({
            "case": case.name, "device": device.to_string(), "mode": mode,
            "stage": stage, "reason": error.to_string(),
        })
    );
}

impl EncodedTensor {
    fn data_type(&self) -> DataType {
        match self.dtype.as_str() {
            "float32" => DataType::F32,
            "float16" => DataType::F16,
            "bfloat16" => DataType::Bf16,
            "float64" => DataType::F64,
            "int32" => DataType::I32,
            "int64" => DataType::I64,
            "uint8" => DataType::U8,
            "int8" => DataType::I8,
            "bool" => DataType::Bool,
            dtype => panic!("unhandled conformance dtype {dtype}"),
        }
    }

    fn tensor(&self, device: Device) -> Tensor {
        Tensor::from_bytes(
            device,
            self.data_type(),
            Shape::create(self.shape.clone()).unwrap(),
            &decode(&self.bytes),
        )
        .unwrap()
    }

    fn assert_matches(
        &self,
        actual: &Tensor,
        label: &str,
        check_zero_sign: bool,
        exact_storage: bool,
    ) {
        assert_eq!(actual.data_type(), self.data_type(), "{label}");
        assert_eq!(actual.shape().dimensions(), self.shape, "{label}");
        if !exact_storage
            && matches!(
                self.data_type(),
                DataType::F32 | DataType::F16 | DataType::Bf16
            )
        {
            let expected = self
                .tensor(Device::Cpu)
                .cast(DataType::F32)
                .unwrap()
                .to_vec::<f32>()
                .unwrap();
            let actual = actual
                .to_device(Device::Cpu)
                .unwrap()
                .cast(DataType::F32)
                .unwrap()
                .to_vec::<f32>()
                .unwrap();
            let tolerance = match self.data_type() {
                DataType::F16 => 2e-3,
                DataType::Bf16 => 2e-2,
                _ => 3e-4,
            };
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected) {
                if check_zero_sign && expected == 0. {
                    assert_eq!(
                        actual.is_sign_negative(),
                        expected.is_sign_negative(),
                        "{label}: zero sign"
                    );
                }
                assert!(
                    (actual - expected).abs() <= tolerance * expected.abs().max(1.)
                        || *actual == expected
                        || actual.is_nan() && expected.is_nan(),
                    "{label}: actual {actual}, expected {expected}"
                );
            }
        } else {
            assert_eq!(actual.to_bytes().unwrap(), decode(&self.bytes), "{label}");
        }
    }
}

fn check_case(case: &Case, device: Device) -> Support {
    let mut result = Support {
        name: case.name.clone(),
        importable: false,
        eager: false,
        compiled: false,
    };
    let model = match OnnxModel::from_reader(decode(&case.model).as_slice(), OnnxOptions::new()) {
        Ok(model) => model,
        Err(error) => {
            assert!(
                matches!(
                    error,
                    OnnxError::UnsupportedOperator { .. }
                        | OnnxError::UnsupportedType { .. }
                        | OnnxError::UnsupportedShape { .. }
                ),
                "{} unexpected import error: {error}",
                case.name
            );
            record_rejection(case, device, "all", "import", &error);
            return result;
        }
    };
    result.importable = true;
    let inputs = case
        .inputs
        .iter()
        .map(|(name, value)| (name.clone(), value.tensor(device)))
        .collect();
    for (mode, supported) in [
        (GraphExecutionMode::Eager, &mut result.eager),
        (GraphExecutionMode::Compiled, &mut result.compiled),
    ] {
        let mut executor =
            match model.prepare(GraphPrepareOptions::new(device).with_execution_mode(mode)) {
                Ok(executor) => executor,
                Err(error) if unsupported(&error) => {
                    record_rejection(case, device, &format!("{mode:?}"), "prepare", &error);
                    continue;
                }
                Err(error) => panic!(
                    "{} {device} {mode:?} preparation failed: {error}",
                    case.name
                ),
            };
        let outputs = executor.run(&inputs).unwrap_or_else(|error| {
            panic!("{} {device} {mode:?} execution failed: {error}", case.name)
        });
        assert_eq!(
            outputs.len(),
            case.outputs.len(),
            "{} {device} {mode:?}",
            case.name
        );
        for (actual, expected) in outputs.iter().zip(&case.outputs) {
            expected.assert_matches(
                actual,
                &format!("{} {device} {mode:?}", case.name),
                case.check_zero_sign,
                case.exact_storage,
            );
        }
        *supported = true;
    }
    result
}

fn check_conformance(device: Device, baseline: &str) {
    let results: Vec<_> = include_str!("../references/onnx/conformance.jsonl")
        .lines()
        .map(|line| {
            let case: Case = serde_json::from_str(line).unwrap();
            let result = check_case(&case, device);
            if case.required || (case.required_cuda && device.is_cuda()) {
                assert!(
                    result.importable && result.eager && result.compiled,
                    "{} {device}: required conformance failed: {result:?}",
                    case.name
                );
            }
            result
        })
        .collect();
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/baselines/onnx")
        .join(baseline);
    if env::var_os("SINGE_UPDATE_ONNX_CONFORMANCE").is_some() {
        fs::write(path, serde_json::to_string_pretty(&results).unwrap() + "\n").unwrap();
    } else {
        let expected: Vec<Support> =
            serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(
            results, expected,
            "ONNX support changed; inspect failures before updating the conformance baseline"
        );
    }
}

#[test]
fn test_operator_conformance_on_cpu() {
    check_conformance(Device::Cpu, "cpu.json");
}

#[cfg(feature = "cuda")]
#[test]
fn test_operator_conformance_on_cuda() {
    check_conformance(Device::Cuda(0), "cuda.json");
}
