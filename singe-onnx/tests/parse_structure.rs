use std::io::Cursor;

use bomboni_request::parse::RequestParseInto;
use prost::Message;
use singe_onnx::proto::onnx::{
    GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto, TensorShapeProto,
    TypeProto, ValueInfoProto, tensor_proto, tensor_shape_proto, type_proto,
};

#[test]
fn decode_model_structure() {
    let weight_data: Vec<_> = (0..=255).cycle().take(4096).collect();
    let model = ModelProto {
        ir_version: 13,
        producer_name: "singe-onnx-test".into(),
        graph: Some(GraphProto {
            name: "structure_graph".into(),
            node: vec![NodeProto {
                input: vec!["input".into(), "linear.weight".into()],
                output: vec!["output".into()],
                name: "linear".into(),
                op_type: "MatMul".into(),
                ..Default::default()
            }],
            initializer: vec![TensorProto {
                dims: vec![4, 4],
                data_type: tensor_proto::DataType::Float as i32,
                name: "linear.weight".into(),
                raw_data: weight_data.clone(),
                ..Default::default()
            }],
            input: vec![value_info("input", &[1, 4])],
            output: vec![value_info("output", &[1, 4])],
            ..Default::default()
        }),
        opset_import: vec![OperatorSetIdProto {
            version: 23,
            ..Default::default()
        }],
        ..Default::default()
    };

    let full_bytes = model.encode_to_vec();
    let full_model = singe_onnx::decode_proto(&full_bytes).expect("decode full ONNX model");
    let full_initializer = full_model
        .graph
        .as_ref()
        .expect("full graph")
        .initializer
        .first()
        .expect("full initializer");
    assert_eq!(full_initializer.raw_data, weight_data);

    let structure =
        singe_onnx::decode_structure_proto(&full_bytes).expect("decode structure-only ONNX model");
    let graph = structure.graph.as_ref().expect("structure graph");
    assert_eq!(graph.name, "structure_graph");
    assert_eq!(graph.node.len(), 1);
    assert_eq!(graph.node[0].op_type, "MatMul");
    assert_eq!(
        graph.node[0].input,
        ["input".to_string(), "linear.weight".to_string()]
    );
    assert_eq!(graph.input.len(), 1);
    assert_eq!(graph.output.len(), 1);

    let initializer = graph.initializer.first().expect("structure initializer");
    assert_eq!(initializer.name, "linear.weight");
    assert_eq!(initializer.dims, [4, 4]);
    assert_eq!(initializer.data_type, tensor_proto::DataType::Float as i32);

    let structure_bytes = structure.encode_to_vec();
    assert!(
        structure_bytes.len() < full_bytes.len() - full_initializer.raw_data.len(),
        "structure-only encoding should not retain skipped inline tensor payloads"
    );
    assert!(
        !structure_bytes
            .windows(full_initializer.raw_data.len())
            .any(|window| window == full_initializer.raw_data),
        "structure-only encoding unexpectedly retained raw_data"
    );

    let parsed_structure =
        singe_onnx::decode_structure(&full_bytes).expect("parse structure-only ONNX model");
    assert_eq!(parsed_structure.producer_name, "singe-onnx-test");
    assert_eq!(parsed_structure.graph.name, "structure_graph");
    assert_eq!(parsed_structure.graph.nodes.len(), 1);
    assert_eq!(parsed_structure.graph.nodes[0].operator_type, "MatMul");
    assert_eq!(parsed_structure.graph.inputs.len(), 1);
    assert_eq!(parsed_structure.graph.outputs.len(), 1);

    let parsed_initializer = parsed_structure
        .graph
        .initializers
        .first()
        .expect("parsed structure initializer");
    assert_eq!(parsed_initializer.name, "linear.weight");
    assert_eq!(parsed_initializer.dimensions, [4, 4]);
    assert_eq!(
        parsed_initializer.data_type,
        singe_onnx::TensorDataType::Float
    );
    assert_eq!(parsed_initializer.data, None);

    let read_structure =
        singe_onnx::read_structure(Cursor::new(&full_bytes)).expect("read structure ONNX model");
    assert_eq!(read_structure, parsed_structure);

    let parsed_from_proto: singe_onnx::Model = structure
        .parse_into()
        .expect("parse structure-only ONNX proto into typed model");
    assert_eq!(parsed_from_proto, parsed_structure);
}

fn value_info(name: &str, dimensions: &[i64]) -> ValueInfoProto {
    ValueInfoProto {
        name: name.into(),
        r#type: Some(TypeProto {
            value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                elem_type: tensor_proto::DataType::Float as i32,
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
