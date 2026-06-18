use std::path::Path;

use singe_onnx::{AttributeValue, TensorData, TensorDataType};

#[test]
fn parses_generated_image_classifier_export() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/image_classifier_model.onnx");
    if !path.exists() {
        eprintln!(
            "warning: skipping ONNX export parser test because {} is missing; run `just prepare` to generate it",
            path.display()
        );
        return;
    }

    let model = singe_onnx::read_path(path).expect("parse generated ONNX model");

    assert_eq!(model.producer_name, "pytorch");
    assert_eq!(model.graph.name, "main_graph");
    assert_eq!(model.graph.inputs.len(), 1);
    assert_eq!(model.graph.outputs.len(), 1);
    assert_eq!(model.graph.nodes.len(), 12);
    assert_eq!(model.graph.initializers.len(), 11);

    let operators: Vec<_> = model
        .graph
        .nodes
        .iter()
        .map(|node| node.operator_type.as_str())
        .collect();
    assert_eq!(
        operators,
        [
            "Conv", "Relu", "MaxPool", "Conv", "Relu", "MaxPool", "Reshape", "Gemm", "Relu",
            "Gemm", "Relu", "Gemm"
        ]
    );

    let conv_weight = model
        .graph
        .initializers
        .iter()
        .find(|tensor| tensor.name == "conv1.weight")
        .expect("conv1.weight initializer");
    assert_eq!(conv_weight.data_type, TensorDataType::Float);
    assert_eq!(conv_weight.dimensions, [6, 1, 5, 5]);
    assert!(matches!(conv_weight.data, Some(TensorData::Raw(_))));

    let conv = model
        .graph
        .nodes
        .iter()
        .find(|node| node.operator_type == "Conv")
        .expect("conv node");
    let strides = conv
        .attributes
        .iter()
        .find(|attribute| attribute.name == "strides")
        .expect("conv strides attribute");
    assert_eq!(strides.value, AttributeValue::Ints(vec![1, 1]));
}
