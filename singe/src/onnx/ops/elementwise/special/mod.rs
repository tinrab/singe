mod activation;
mod selection;

use singe_onnx::Node;

pub use self::{
    activation::{lower_leaky_relu, lower_pow, lower_softmax},
    selection::{lower_clip, lower_where},
};
use crate::{
    data_type::DataType,
    onnx::{
        error::{Error, Result},
        importer::Importer,
        ops::{expect_input_count, expect_input_range, expect_outputs, register_node_outputs},
    },
};

pub fn lower_identity(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_count(importer, node, 1)?;
    expect_outputs(importer, node, 1)?;
    let input = importer.required_input(node, 0)?;
    register_node_outputs(importer, node, vec![input])
}

pub fn lower_dropout(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_range(importer, node, 1, 3)?;
    if node.outputs.is_empty() || node.outputs.len() > 2 {
        return Err(Error::unsupported_operator(
            node.domain.as_deref(),
            "Dropout",
            importer.opset_version(node.domain.as_deref()),
            format!("expected 1..=2 outputs, got {}", node.outputs.len()),
        ));
    }

    if node.outputs.get(1).is_some_and(|name| !name.is_empty()) {
        return Err(Error::unsupported_operator(
            node.domain.as_deref(),
            "Dropout",
            importer.opset_version(node.domain.as_deref()),
            "training mask output is not supported for inference import",
        ));
    }

    if let Some((_, training)) = importer.optional_static_input_tensor(node, 2)? {
        if training.shape().rank() != 0
            || training.data_type() != DataType::Bool
            || training.to_vec::<bool>()? != [false]
        {
            return Err(Error::unsupported_operator(
                node.domain.as_deref(),
                "Dropout",
                importer.opset_version(node.domain.as_deref()),
                "only inference dropout is supported",
            ));
        }
    } else if node.inputs.get(2).is_some_and(|name| !name.is_empty()) {
        return Err(Error::unsupported_operator(
            node.domain.as_deref(),
            "Dropout",
            importer.opset_version(node.domain.as_deref()),
            "runtime training mode is not supported",
        ));
    }

    let input = importer.required_input(node, 0)?;
    importer.register_value(&node.outputs[0], input)
}
