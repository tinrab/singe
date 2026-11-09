use crate::onnx::{
    error::{Error, Result},
    importer::{ImportedValue, Importer},
    ops::lowering::{Node, unsupported_output_count},
};

pub fn optional_input(
    importer: &Importer,
    node: &Node,
    index: usize,
) -> Result<Option<ImportedValue>> {
    importer.optional_input(node, index)
}

pub fn expect_input_count(importer: &Importer, node: &Node, count: usize) -> Result<()> {
    expect_input_range(importer, node, count, count)
}

pub fn expect_input_range(importer: &Importer, node: &Node, min: usize, max: usize) -> Result<()> {
    if node.inputs.len() < min || node.inputs.len() > max {
        return Err(Error::unsupported_operator(
            node.domain.as_deref(),
            &node.operator_type,
            importer.opset_version(node.domain.as_deref()),
            format!("expected {min}..={max} inputs, got {}", node.inputs.len()),
        ));
    }
    Ok(())
}

pub fn expect_outputs(importer: &Importer, node: &Node, count: usize) -> Result<()> {
    if node.outputs.len() != count {
        return Err(unsupported_output_count(
            importer,
            node,
            format!("expected {count} outputs, got {}", node.outputs.len()),
        ));
    }
    Ok(())
}
