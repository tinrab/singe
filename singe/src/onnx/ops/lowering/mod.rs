mod graph;
mod quantization;
mod shape;
mod validation;

use singe_onnx::Node;

pub use self::{
    graph::{
        add_binary_intermediate, add_scale, add_transpose, add_unary_intermediate,
        register_node_outputs,
    },
    quantization::{
        optional_quantization_zero_point, optional_static_f32_scalar, quantization_scale,
        static_f32_scalar, zero_point_initializer,
    },
    shape::{
        axes_from_attribute_or_input, expand_output_shape, normalize_shape_bound,
        normalize_slice_index, normalized_axes, normalized_unsqueeze_axes, onnx_padding,
        padded_shape, shape_from_static_dimensions, shape_input_values, validate_permutation,
    },
    validation::{expect_input_count, expect_input_range, expect_outputs, optional_input},
};
use crate::onnx::{error::Error, importer::Importer};

fn unsupported_output_count(importer: &Importer, node: &Node, details: impl Into<String>) -> Error {
    Error::unsupported_operator(
        node.domain.as_deref(),
        &node.operator_type,
        importer.opset_version(node.domain.as_deref()),
        details,
    )
}
