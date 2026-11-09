use singe_onnx::Node;

use crate::{
    backend::registry::Operation,
    data_type::DataType,
    graph::{GraphOperationOutput, GraphOperationRequest},
    onnx::{
        attributes::Attributes,
        error::{Error, Result},
        importer::{ImportedValue, Importer},
        ops::{expect_input_count, expect_outputs, register_node_outputs, shape_input_values},
        utility::{node_label, node_name},
    },
    shape::{Shape, normalize_dim_usize},
};

pub fn lower_topk(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_count(importer, node, 2)?;
    expect_outputs(importer, node, 2)?;
    let prepared = topk_input(importer, node)?;
    let base_name = node_name(node);
    let device = importer.options().device;
    let values = importer
        .graph()
        .add_operation_request(GraphOperationRequest::topk(
            base_name.as_ref().map(|name| format!("{name}/values")),
            Operation::TopKValues,
            prepared.input.node,
            prepared.k,
            prepared.dim,
            prepared.largest,
            prepared.sorted,
            GraphOperationOutput::new(prepared.input.data_type, prepared.shape.clone(), device),
        ))?;
    let indices = importer
        .graph()
        .add_operation_request(GraphOperationRequest::topk(
            base_name.as_ref().map(|name| format!("{name}/indices")),
            Operation::TopKIndices,
            prepared.input.node,
            prepared.k,
            prepared.dim,
            prepared.largest,
            prepared.sorted,
            GraphOperationOutput::new(DataType::I64, prepared.shape.clone(), device),
        ))?;
    register_node_outputs(
        importer,
        node,
        vec![
            ImportedValue::new(values, prepared.input.data_type, prepared.shape.clone()),
            ImportedValue::new(indices, DataType::I64, prepared.shape),
        ],
    )
}

pub struct TopKInput {
    pub(crate) input: ImportedValue,
    pub(crate) k: i64,
    pub(crate) dim: i64,
    pub(crate) largest: bool,
    pub(crate) sorted: bool,
    pub(crate) shape: Shape,
}

pub fn topk_input(importer: &Importer, node: &Node) -> Result<TopKInput> {
    let attrs = Attributes::create(node)?;
    let input = importer.required_input(node, 0)?;
    validate_topk_input(importer, node, &input)?;
    let k = topk_k(importer, node)?;
    let dim = attrs.int("axis", -1)?;
    let shape = topk_shape(node, &input, dim, k)?;
    Ok(TopKInput {
        input,
        k,
        dim,
        largest: attrs.int("largest", 1)? != 0,
        sorted: attrs.int("sorted", 1)? != 0,
        shape,
    })
}

pub fn validate_topk_input(importer: &Importer, node: &Node, input: &ImportedValue) -> Result<()> {
    if input.data_type == DataType::F32 {
        return Ok(());
    }
    Err(Error::unsupported_operator(
        node.domain.as_deref(),
        &node.operator_type,
        importer.opset_version(node.domain.as_deref()),
        "input must be a float32 tensor",
    ))
}

pub fn topk_k(importer: &Importer, node: &Node) -> Result<i64> {
    let k_name = &node.inputs[1];
    let k_values = shape_input_values(importer, node, 1, "k")?;
    match k_values.as_slice() {
        [k] if *k > 0 => Ok(*k),
        [_] => Err(Error::invalid_initializer(k_name, "k must be positive")),
        _ => Err(Error::invalid_initializer(
            k_name,
            "k must contain exactly one element",
        )),
    }
}

pub fn topk_shape(node: &Node, input: &ImportedValue, dim: i64, k: i64) -> Result<Shape> {
    let dim_index = normalize_dim_usize(dim, input.shape.rank())?;
    let mut output_dimensions = input.shape.dimensions().to_vec();
    if k > output_dimensions[dim_index] {
        return Err(Error::UnsupportedShape {
            name: node_label(node),
            reason: "topk k exceeds selected dimension".into(),
        });
    }

    output_dimensions[dim_index] = k;
    Shape::create(output_dimensions).map_err(Into::into)
}
