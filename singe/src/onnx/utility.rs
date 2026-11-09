use singe_onnx::Node;

use crate::onnx::error::{Error, Result};

pub fn pair(values: &[i64], attribute: &str, node: &Node) -> Result<(i64, i64)> {
    match values {
        [a, b] => Ok((*a, *b)),
        _ => Err(Error::invalid_attribute(
            node_label(node),
            attribute,
            "expected two values",
        )),
    }
}

pub fn node_name(node: &Node) -> Option<String> {
    (!node.name.is_empty()).then(|| node.name.clone())
}

pub fn node_label(node: &Node) -> String {
    if node.name.is_empty() {
        node.operator_type.clone()
    } else {
        node.name.clone()
    }
}
