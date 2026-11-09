mod concat;
mod construction;
mod pad;
mod slice;
mod view;

use singe_onnx::Node;

pub use self::{
    concat::lower_concat,
    construction::{lower_expand, lower_reshape, lower_shape},
    pad::lower_pad,
    slice::lower_slice,
    view::{
        lower_flatten, lower_gather, lower_scatter_elements, lower_squeeze, lower_transpose,
        lower_unsqueeze,
    },
};
use crate::onnx::{
    error::{Error, Result},
    utility::node_label,
};

pub fn onnx_dimension_to_usize(node: &Node, name: &'static str, value: i64) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::UnsupportedShape {
        name: node_label(node),
        reason: format!("{name} is out of range"),
    })
}

pub fn onnx_len_to_i64(node: &Node, name: &'static str, value: usize) -> Result<i64> {
    i64::try_from(value).map_err(|_| Error::UnsupportedShape {
        name: node_label(node),
        reason: format!("{name} is too large"),
    })
}
