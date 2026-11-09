#![cfg(feature = "onnx")]

use std::{
    collections::HashMap,
    env, fs,
    path::PathBuf,
    process,
    sync::atomic::{AtomicU64, Ordering},
};

#[cfg(unix)]
use std::os::unix::fs::symlink;

use prost::Message;
use singe::{DataType, OnnxModel, OnnxOptions, onnx::Error};
use singe_onnx::proto::onnx::{
    GraphProto, ModelProto, NodeProto, OperatorSetIdProto, StringStringEntryProto, TensorProto,
    TensorShapeProto, TypeProto, ValueInfoProto, tensor_proto, tensor_shape_proto, type_proto,
};

static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

struct Directory(PathBuf);

impl Directory {
    fn create() -> Self {
        let path = env::temp_dir().join(format!(
            "singe-onnx-{}-{}",
            process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn load(&self, model: &ModelProto) -> Result<OnnxModel, Error> {
        let path = self.0.join("model.onnx");
        fs::write(&path, model.encode_to_vec()).unwrap();
        OnnxModel::create(path, OnnxOptions::new())
    }
}

impl Drop for Directory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn constant_model(tensor: TensorProto) -> ModelProto {
    let shape = tensor.dims.clone();
    let dtype = tensor.data_type;
    ModelProto {
        ir_version: 10,
        graph: Some(GraphProto {
            name: "external_data".into(),
            node: vec![NodeProto {
                op_type: "Identity".into(),
                input: vec!["weight".into()],
                output: vec!["y".into()],
                ..Default::default()
            }],
            initializer: vec![tensor],
            output: vec![ValueInfoProto {
                name: "y".into(),
                r#type: Some(TypeProto {
                    value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                        elem_type: dtype,
                        shape: Some(TensorShapeProto {
                            dim: shape
                                .into_iter()
                                .map(|size| tensor_shape_proto::Dimension {
                                    value: Some(tensor_shape_proto::dimension::Value::DimValue(
                                        size,
                                    )),
                                    ..Default::default()
                                })
                                .collect(),
                        }),
                    })),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }),
        opset_import: vec![OperatorSetIdProto {
            version: 13,
            ..Default::default()
        }],
        ..Default::default()
    }
}

fn external_tensor(location: &str, offset: &str, length: Option<&str>) -> TensorProto {
    let mut props = vec![("location", location), ("offset", offset)];
    if let Some(length) = length {
        props.push(("length", length));
    }
    TensorProto {
        name: "weight".into(),
        dims: vec![2],
        data_type: tensor_proto::DataType::Float as i32,
        data_location: tensor_proto::DataLocation::External as i32,
        external_data: props
            .into_iter()
            .map(|(key, value)| StringStringEntryProto {
                key: key.into(),
                value: value.into(),
            })
            .collect(),
        ..Default::default()
    }
}

#[test]
fn test_external_data_offsets_lengths_truncation_and_reader_context_are_checked() {
    let directory = Directory::create();
    let payload: Vec<_> = [1.5_f32, -2.25]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let mut file = vec![0_u8; 3];
    file.extend(&payload);
    file.extend([0; 2]);
    fs::write(directory.0.join("weights.bin"), file).unwrap();
    let source = constant_model(external_tensor("weights.bin", "3", Some("8")));
    let model = directory.load(&source).unwrap();
    assert_eq!(
        model.run_ordered(&HashMap::new()).unwrap()[0]
            .to_vec::<f32>()
            .unwrap(),
        [1.5, -2.25]
    );
    assert!(OnnxModel::from_reader(source.encode_to_vec().as_slice(), OnnxOptions::new()).is_err());
    for (offset, length) in [
        ("-1", "8"),
        ("invalid", "8"),
        ("3", "9"),
        ("3", "7"),
        ("3", "100"),
        ("18446744073709551615", "2"),
    ] {
        assert!(
            directory
                .load(&constant_model(external_tensor(
                    "weights.bin",
                    offset,
                    Some(length)
                )))
                .is_err(),
            "{offset}, {length}"
        );
    }
    fs::write(directory.0.join("weights.bin"), &payload[..4]).unwrap();
    assert!(
        directory
            .load(&constant_model(external_tensor("weights.bin", "0", None)))
            .is_err()
    );
    assert!(
        directory
            .load(&constant_model(external_tensor(
                "missing.bin",
                "0",
                Some("8")
            )))
            .is_err()
    );
}

#[test]
fn test_external_paths_cannot_escape_the_model_directory() {
    let directory = Directory::create();
    let outside = Directory::create();
    fs::write(outside.0.join("weights.bin"), [0_u8; 8]).unwrap();
    for location in [
        outside.0.join("weights.bin").to_str().unwrap(),
        "../weights.bin",
    ] {
        assert!(matches!(
            directory.load(&constant_model(external_tensor(location, "0", Some("8")))),
            Err(Error::InvalidExternalData { .. })
        ));
    }
    #[cfg(unix)]
    {
        symlink(
            outside.0.join("weights.bin"),
            directory.0.join("escape.bin"),
        )
        .unwrap();
        assert!(matches!(
            directory.load(&constant_model(external_tensor(
                "escape.bin",
                "0",
                Some("8")
            ))),
            Err(Error::InvalidExternalData { .. })
        ));
    }
}

#[test]
fn test_tensor_encodings_and_initializer_collisions_are_validated() {
    let import = |tensor| {
        OnnxModel::from_reader(
            constant_model(tensor).encode_to_vec().as_slice(),
            OnnxOptions::new(),
        )
    };
    for (dtype, bits, expected_type, expected_bytes) in [
        (
            tensor_proto::DataType::Float16,
            vec![0x3c00, 0xc000],
            DataType::F16,
            vec![0, 0x3c, 0, 0xc0],
        ),
        (
            tensor_proto::DataType::Bfloat16,
            vec![0x3f80, 0xc000],
            DataType::Bf16,
            vec![0x80, 0x3f, 0, 0xc0],
        ),
        (
            tensor_proto::DataType::Bool,
            vec![0, 1],
            DataType::Bool,
            vec![0, 1],
        ),
    ] {
        let tensor = TensorProto {
            name: "weight".into(),
            dims: vec![2],
            data_type: dtype as i32,
            int32_data: bits,
            ..Default::default()
        };
        let output = import(tensor)
            .unwrap()
            .run_ordered(&HashMap::new())
            .unwrap();
        assert_eq!(output[0].data_type(), expected_type);
        assert_eq!(output[0].to_bytes().unwrap(), expected_bytes);
    }
    for (dtype, bits) in [
        (tensor_proto::DataType::Float16, vec![-1, 0]),
        (tensor_proto::DataType::Bfloat16, vec![65536, 0]),
        (tensor_proto::DataType::Bool, vec![0, 2]),
        (tensor_proto::DataType::Uint8, vec![0, 256]),
    ] {
        assert!(
            import(TensorProto {
                name: "weight".into(),
                dims: vec![2],
                data_type: dtype as i32,
                int32_data: bits,
                ..Default::default()
            })
            .is_err()
        );
    }
    let tensor = TensorProto {
        name: "weight".into(),
        dims: vec![2],
        data_type: tensor_proto::DataType::Float as i32,
        float_data: vec![1., 2.],
        ..Default::default()
    };
    let mut source = constant_model(tensor.clone());
    source
        .graph
        .as_mut()
        .unwrap()
        .initializer
        .push(tensor.clone());
    assert!(matches!(
        OnnxModel::from_reader(source.encode_to_vec().as_slice(), OnnxOptions::new()),
        Err(Error::DuplicateValue(_))
    ));
    let mut missing = tensor.clone();
    missing.float_data.clear();
    assert!(import(missing).is_err());
    let mut truncated = tensor.clone();
    truncated.float_data.pop();
    assert!(import(truncated).is_err());
    let mut ambiguous = tensor;
    ambiguous.raw_data = vec![0; 8];
    assert!(import(ambiguous).is_err());
}
