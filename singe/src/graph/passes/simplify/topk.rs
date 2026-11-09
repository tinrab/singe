use crate::{
    error::Result,
    graph::{Graph, ir::ValueId, node::NodeRef},
    shape::normalize_dim_usize,
};

pub fn identity_topk_values_input(
    graph: &Graph,
    node: &NodeRef<'_>,
    input: ValueId,
    k: i64,
    dim: i64,
    sorted: bool,
) -> Result<Option<ValueId>> {
    let input_node = &graph.required_node(input)?;
    let axis = normalize_dim_usize(dim, input_node.output_shape().rank())?;
    let dimension = input_node.output_shape().dimensions()[axis];
    if dimension != k || (sorted && dimension > 1) {
        return Ok(None);
    }
    Ok(input_node.has_same_value_metadata(node).then_some(input))
}
