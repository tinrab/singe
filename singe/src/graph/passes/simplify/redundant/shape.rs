use crate::{
    backend::registry::Operation,
    data_type::DataType,
    error::{Error, Result},
    graph::{
        Graph,
        ir::ValueId,
        node::NodeRef,
        passes::{
            fold::constant_input,
            shape::{
                concat_output_shape_for_inputs, pad_output_shape_for_input,
                repeat_kv_output_shape_for_input,
            },
        },
    },
    shape::normalize_dim_usize,
    utility::to_usize,
};

pub fn identity_concat_input(
    graph: &Graph,
    node: &NodeRef<'_>,
    inputs: &[ValueId],
    dim: i64,
) -> Result<Option<ValueId>> {
    validate_concat_metadata(graph, node, inputs, dim)?;
    if inputs.len() == 1 {
        let input = inputs[0];
        concat_axis_dimension(graph, input, dim)?;
        let input_node = &graph.required_node(input)?;
        return Ok(input_node.has_same_value_metadata(node).then_some(input));
    }

    let mut replacement = None;
    let mut zero_replacement = None;
    for input in inputs {
        if concat_axis_dimension(graph, *input, dim)? == 0 {
            zero_replacement.get_or_insert(*input);
            continue;
        }
        if replacement.replace(*input).is_some() {
            return Ok(None);
        }
    }

    let Some(replacement) = replacement.or(zero_replacement) else {
        return Ok(None);
    };
    let input_node = &graph.required_node(replacement)?;
    Ok(input_node
        .has_same_value_metadata(node)
        .then_some(replacement))
}

fn validate_concat_metadata(
    graph: &Graph,
    node: &NodeRef<'_>,
    inputs: &[ValueId],
    dim: i64,
) -> Result<()> {
    let output_shape = concat_output_shape_for_inputs(graph, inputs, dim)?;
    if node.output_shape() != &output_shape {
        return Err(Error::shape_mismatch(
            "redundant concat output shape",
            output_shape.dimensions().to_vec(),
            node.output_shape().dimensions().to_vec(),
        ));
    }
    Ok(())
}

pub fn identity_pad_input(
    graph: &Graph,
    node: &NodeRef<'_>,
    input: ValueId,
    padding: &[(i64, i64)],
) -> Result<Option<ValueId>> {
    validate_pad_metadata(graph, node, input, padding)?;
    let padding = graph.node_normalized_padding(input, padding)?;
    if !padding.is_empty() {
        return Ok(None);
    }

    let input_node = &graph.required_node(input)?;
    Ok(input_node.has_same_value_metadata(node).then_some(input))
}

fn validate_pad_metadata(
    graph: &Graph,
    node: &NodeRef<'_>,
    input: ValueId,
    padding: &[(i64, i64)],
) -> Result<()> {
    let output_shape = pad_output_shape_for_input(graph, input, padding)?;
    if node.output_shape() != &output_shape {
        return Err(Error::shape_mismatch(
            "redundant pad output shape",
            output_shape.dimensions().to_vec(),
            node.output_shape().dimensions().to_vec(),
        ));
    }
    Ok(())
}

pub fn identity_repeat_kv_input(
    graph: &Graph,
    node: &NodeRef<'_>,
    input: ValueId,
    repeats: i64,
) -> Result<Option<ValueId>> {
    let output_shape = repeat_kv_output_shape_for_input(graph, input, repeats)?;
    if node.output_shape() != &output_shape {
        return Ok(None);
    }

    let input_node = &graph.required_node(input)?;
    Ok(input_node.has_same_value_metadata(node).then_some(input))
}

pub fn identity_index_select_input(
    graph: &Graph,
    node: &NodeRef<'_>,
    input: ValueId,
    dim: i64,
    indices: ValueId,
) -> Result<Option<ValueId>> {
    let input_node = &graph.required_node(input)?;
    if !input_node.has_same_value_metadata(node) {
        return Ok(None);
    }

    let axis = normalize_dim_usize(dim, input_node.output_shape().rank())?;
    let dimension = input_node.output_shape().dimensions()[axis];
    if dimension < 0 {
        return Err(Error::out_of_range("identity index_select dimension"));
    }

    let Some(indices) = constant_input(graph, indices)? else {
        return Ok(None);
    };
    if indices.data_type() != DataType::I64 || indices.shape().rank() != 1 {
        return Ok(None);
    }

    let expected_len = to_usize(dimension, "identity index_select dimension")?;
    if indices.shape().dimensions()[0] != dimension {
        return Ok(None);
    }

    let indices = indices.to_vec::<i64>()?;
    let expected = (0..dimension).collect::<Vec<_>>();
    Ok((indices.len() == expected_len && indices == expected).then_some(input))
}

pub fn identity_slice_input(
    graph: &Graph,
    node: &NodeRef<'_>,
    input: ValueId,
    dim: i64,
    start: i64,
    length: i64,
) -> Result<Option<ValueId>> {
    if start != 0 {
        return Ok(None);
    }

    let input_node = &graph.required_node(input)?;
    let axis = normalize_dim_usize(dim, input_node.output_shape().rank())?;
    if input_node.output_shape().dimensions()[axis] != length {
        return Ok(None);
    }
    Ok(input_node.has_same_value_metadata(node).then_some(input))
}

fn concat_axis_dimension(graph: &Graph, input: ValueId, dim: i64) -> Result<i64> {
    let shape = graph.required_value_shape(input)?;
    let axis = normalize_dim_usize(dim, shape.rank())?;
    Ok(shape.dimensions()[axis])
}

// A reduction over dimensions of extent one that keeps them, or a pool whose window is one point with no padding, returns its input unchanged when the result keeps the input's data type.
pub fn identity_window_input(graph: &Graph, node: &NodeRef<'_>) -> Result<Option<ValueId>> {
    let Some(&input) = node.inputs().first() else {
        return Ok(None);
    };
    let input_node = &graph.required_node(input)?;
    if !input_node.has_same_value_metadata(node) {
        return Ok(None);
    }

    let attributes = node.attributes();
    let identity = match node.operation() {
        Some(
            Operation::Sum
            | Operation::Mean
            | Operation::Max
            | Operation::Min
            | Operation::Product
            | Operation::LogSumExp,
        ) => attributes.reduction().is_some_and(|reduction| {
            let dimensions = input_node.output_shape().dimensions();
            reduction.keepdim
                && !reduction.dimensions.is_empty()
                && reduction.dimensions.iter().all(|&dim| {
                    normalize_dim_usize(dim, input_node.output_shape().rank())
                        .is_ok_and(|axis| dimensions[axis] == 1)
                })
        }),
        // Pools accept floating values only, so an integer pool stays for preparation to reject.
        Some(Operation::MaxPool2d | Operation::AvgPool2d) if node.output_data_type().is_float() => {
            attributes.pool2d().is_some_and(|pool| {
                pool.kernel_size == (1, 1) && pool.stride == (1, 1) && pool.padding == (0, 0, 0, 0)
            })
        }
        _ => false,
    };
    Ok(identity.then_some(input))
}
