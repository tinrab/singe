use crate::{
    error::Result,
    graph::{
        Graph,
        ir::ValueId,
        node::NodeRef,
        passes::simplify::{
            redundant::{
                alias::{composed_permutation_redundant_input, redundant_alias_operation_input},
                shape::{
                    identity_concat_input, identity_index_select_input, identity_pad_input,
                    identity_repeat_kv_input, identity_slice_input, identity_window_input,
                },
            },
            topk::identity_topk_values_input,
        },
    },
};

pub fn alias_redundant_input(graph: &Graph, node: &NodeRef<'_>) -> Result<Option<ValueId>> {
    if node
        .permutation_order(node.output_shape().rank())?
        .is_some()
    {
        if let Some(input) = composed_permutation_redundant_input(graph, node)? {
            return Ok(Some(input));
        }
        return redundant_alias_operation_input(graph, node);
    }

    if node.shape_only_view_input().is_some() || node.expand_input().is_some() {
        return redundant_alias_operation_input(graph, node);
    }
    Ok(None)
}

pub fn shape_redundant_input(graph: &Graph, node: &NodeRef<'_>) -> Result<Option<ValueId>> {
    if let Some((inputs, dim)) = node.concat() {
        return identity_concat_input(graph, node, &inputs, dim);
    }

    if let Some((input, dim, indices)) = node.index_select() {
        return identity_index_select_input(graph, node, input, dim, indices);
    }

    if let Some((input, repeats)) = node.repeat_kv() {
        return identity_repeat_kv_input(graph, node, input, repeats);
    }

    if let Some((input, padding, _)) = node.pad() {
        return identity_pad_input(graph, node, input, &padding);
    }

    if let Some((input, k, dim, sorted)) = node.topk_values_identity() {
        return identity_topk_values_input(graph, node, input, k, dim, sorted);
    }

    if let Some((input, dim, start, length)) = node.slice() {
        return identity_slice_input(graph, node, input, dim, start, length);
    }
    identity_window_input(graph, node)
}
