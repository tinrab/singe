use singe_onnx::Node;

use crate::{
    onnx::error::{Error, Result},
    shape::spatial::{Padding, Padding2d, Spatial2dI64},
};

// The node metadata naming the operation an `Identity` of an exported decomposition stands for, and each of its inputs, `singe.input.<index>`.
pub const FUSED_OPERATION: &str = "singe.operation";
pub const FUSED_INPUT: &str = "singe.input";

pub fn pair(values: &[i64], attribute: &str, node: &Node) -> Result<Spatial2dI64> {
    match *values {
        [height, width] => Ok(Spatial2dI64::new(height, width)),
        _ => Err(Error::invalid_attribute(
            node_label(node),
            attribute,
            "expected two values",
        )),
    }
}

// ONNX `pads` of two axes, which lists every axis's start before any axis's end.
pub fn planar_pads(values: &[i64], node: &Node) -> Result<Padding2d> {
    match *values {
        [top, left, bottom, right] => Ok(Padding2d::new(
            Padding::new(top, bottom),
            Padding::new(left, right),
        )),
        _ => Err(Error::invalid_attribute(
            node_label(node),
            "pads",
            "expected four values",
        )),
    }
}

// `padding` as ONNX `pads`.
#[must_use]
pub fn onnx_planar_pads(padding: Padding2d) -> Vec<i64> {
    vec![
        padding.height.before,
        padding.width.before,
        padding.height.after,
        padding.width.after,
    ]
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
