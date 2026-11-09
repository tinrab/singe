use crate::{
    device::Device,
    error::{Error, Result},
    graph::{Graph, ir::ValueId, node::NodeRef},
    shape::normalize_dim_usize,
    utility::to_i64,
};

pub fn redundant_metadata_or_transfer_input(
    graph: &Graph,
    node: &NodeRef<'_>,
) -> Result<Option<ValueId>> {
    if node.is_contiguous_operation() {
        let input = node
            .materialization_input()
            .ok_or_else(|| Error::missing_node("redundant metadata or transfer input"))?;
        let input_node = &graph.required_node(input)?;
        return Ok((input_node.output_layout().is_contiguous()
            && input_node.has_same_value_metadata(node))
        .then_some(input));
    }

    if let Some((input, data_type)) = node.cast() {
        let input_node = &graph.required_node(input)?;
        return Ok((input_node.output_data_type() == data_type
            && input_node.has_same_value_metadata(node))
        .then_some(input));
    }

    if let Some((input, device)) = node.copy_to() {
        let input_node = &graph.required_node(input)?;
        if input_node.output_device() == device && input_node.has_same_value_metadata(node) {
            return Ok(Some(input));
        }
        return copy_to_chain_redundant_input(graph, node, input, device);
    }
    Ok(None)
}

pub fn copy_to_chain_redundant_input(
    graph: &Graph,
    node: &NodeRef<'_>,
    input: ValueId,
    device: Device,
) -> Result<Option<ValueId>> {
    let input_node = &graph.required_node(input)?;
    let Some((grandparent, _)) = input_node.copy_to() else {
        return Ok(None);
    };
    let grandparent_node = &graph.required_node(grandparent)?;
    Ok((grandparent_node.output_device() == device
        && grandparent_node.has_same_value_metadata(node))
    .then_some(grandparent))
}

pub fn redundant_alias_operation_input(
    graph: &Graph,
    node: &NodeRef<'_>,
) -> Result<Option<ValueId>> {
    let Some(input) = node.alias_input() else {
        return Ok(None);
    };
    let input_node = &graph.required_node(input)?;
    if node.view_input().is_some() {
        node.view_output_shape(input_node.output_shape())?;
        node.view_output_offset(input_node.output_layout())?;
    }

    if input_node.has_same_value_metadata(node) {
        return Ok(Some(input));
    }

    let Some(parent) = input_node.alias_input() else {
        return Ok(None);
    };
    let parent_node = &graph.required_node(parent)?;
    Ok(parent_node.has_same_value_metadata(node).then_some(parent))
}

pub fn composed_permutation_redundant_input(
    graph: &Graph,
    node: &NodeRef<'_>,
) -> Result<Option<ValueId>> {
    let Some(input) = node.alias_input() else {
        return Ok(None);
    };
    let input_node = &graph.required_node(input)?;
    let Some(grandparent) = input_node.alias_input() else {
        return Ok(None);
    };
    let grandparent_node = &graph.required_node(grandparent)?;
    let Some(parent_order) =
        input_node.permutation_order(grandparent_node.output_shape().rank())?
    else {
        return Ok(None);
    };
    let Some(child_order) = node.permutation_order(input_node.output_shape().rank())? else {
        return Ok(None);
    };
    let composed = compose_permutation(&parent_order, &child_order)?;
    let identity = (0..grandparent_node.output_shape().rank()).collect::<Vec<_>>();
    Ok(
        (composed == identity && grandparent_node.has_same_value_metadata(node))
            .then_some(grandparent),
    )
}

fn compose_permutation(parent_order: &[i64], child_order: &[i64]) -> Result<Vec<i64>> {
    if parent_order.len() != child_order.len() {
        return Err(Error::length_mismatch(
            "graph simplify permutation rank",
            parent_order.len(),
            child_order.len(),
        ));
    }

    let rank = to_i64(parent_order.len(), "graph simplify permutation rank")?;
    child_order
        .iter()
        .map(|axis| {
            let axis = normalize_dim_usize(*axis, rank)?;
            Ok(parent_order[axis])
        })
        .collect()
}
