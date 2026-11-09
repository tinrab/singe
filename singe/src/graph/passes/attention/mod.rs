//! Rewrites attention so backends run it as matmuls and a softmax, and folds grouped-query attention's repeated keys and values into the rows of those matmuls.
//!
//! [`decompose_scaled_dot_product_attention`] replaces each `scaled_dot_product_attention` no fused kernel runs with its decomposition, `softmax(q kᵀ · scale + mask) v` with the causal mask a constant.
//!
//! `repeat_kv(x, g)` copies each of a cache's KV heads `g` times so query head `h` reads KV head `h / g`.
//! A matmul of queries `[B, H, S, D]` with repeated keys or values instead runs on queries viewed as `[B, H / g, g · S, D]` against the cache itself, so the heads that share a KV head become rows of one matmul and the repeated copy is never made.
//! The product, `[B, H / g, g · S, N]`, has the memory order of `[B, H, S, N]`, which a reshape restores.

use std::collections::HashMap;

use crate::{
    backend::{
        registry::{Operation, OperationAttributes},
        spec,
    },
    error::{Error, Result},
    graph::{
        Graph, GraphOperationInputs,
        ir::ValueId,
        node::{Node, NodeKind, NodeRef},
        passes::{decompose::splice, remap::remapped_node},
    },
    shape::Shape,
};

// Returns `graph` with all attention as matmuls around a softmax, for a consumer with no fused kernel to run it, such as ONNX export.
#[cfg(feature = "onnx")]
pub fn decompose_all_attention(graph: &Graph) -> Result<Graph> {
    let mut id_map = HashMap::with_capacity(graph.node_count());
    let mut nodes = Vec::with_capacity(graph.node_count());
    let mut decomposed = false;
    for index in 0..graph.node_count() {
        let id = ValueId::new(index);
        let node = graph.required_node(id)?;
        let new_id = if matches!(
            node.attributes(),
            OperationAttributes::ScaledDotProductAttention(_)
        ) && node.operation() == Some(Operation::ScaledDotProductAttention)
        {
            decomposed = true;
            splice(graph, &node, &id_map, &mut nodes)?
        } else {
            push(&mut nodes, remapped_node(&node, &id_map)?)
        };
        id_map.insert(id, new_id);
    }

    if !decomposed {
        return Ok(graph.clone());
    }

    let mut decomposed = Graph::from_retained_nodes(graph, &id_map, nodes)?;
    // Compiling resolves shapes before it rewrites a graph, and export reads them afterwards, so they must follow their values.
    let mapped = |id: ValueId| {
        id_map
            .get(&id)
            .copied()
            .ok_or_else(|| Error::missing_node("decompose all attention"))
    };
    decomposed.shape_program = graph.shape_program.remap(mapped, &|index| {
        mapped(ValueId::new(index)).map(ValueId::index)
    })?;
    Ok(decomposed)
}

/// Returns `graph` with every matmul of queries by repeated keys or values folded into a matmul by the unrepeated ones, and the map from old values to new ones, which is empty when nothing folds.
///
/// The `repeat_kv` nodes stay, and the dead-node pruning after the passes removes those nothing else reads.
///
/// # Errors
///
/// Returns an error if a node's inputs cannot be remapped.
pub fn fold_grouped_query_attention(graph: &Graph) -> Result<(Graph, Vec<(ValueId, ValueId)>)> {
    let mut id_map = HashMap::with_capacity(graph.node_count());
    let mut nodes = Vec::with_capacity(graph.node_count());
    let mut folded = false;
    for index in 0..graph.node_count() {
        let id = ValueId::new(index);
        let node = graph.required_node(id)?;
        let new_id = match grouped_matmul(graph, &node)? {
            Some(grouped) => {
                folded = true;
                grouped.push(&node, &id_map, &mut nodes)?
            }
            None => push(&mut nodes, remapped_node(&node, &id_map)?),
        };
        id_map.insert(id, new_id);
    }

    if !folded {
        return Ok((graph.clone(), Vec::new()));
    }

    let remapped = graph.remapped_node_pairs(&id_map)?;
    Ok((Graph::from_retained_nodes(graph, &id_map, nodes)?, remapped))
}

/// A matmul whose right operand repeats a cache's KV heads.
struct GroupedMatmul {
    queries: ValueId,
    /// The unrepeated cache, `[B, H / g, T, D]`.
    cache: ValueId,
    /// Whether the matmul reads the cache transposed, as attention scores read keys.
    transposed: bool,
    groups: i64,
    /// The queries' shape, `[B, H, S, D]`.
    query_shape: Shape,
    /// The cache's shape.
    cache_shape: Shape,
}

/// Returns the folding of `node`, when it is a matmul of `[B, H, S, D]` queries by `repeat_kv` of a `[B, H / g, T, D]` cache, or by its transpose of the last two dimensions.
fn grouped_matmul(graph: &Graph, node: &NodeRef<'_>) -> Result<Option<GroupedMatmul>> {
    let &[queries, right] = node.inputs() else {
        return Ok(None);
    };
    if node.operation() != Some(Operation::MatMul) {
        return Ok(None);
    }

    let right_node = graph.required_node(right)?;
    let (repeated, transposed) = match right_node.transpose() {
        Some((input, dim0, dim1)) if last_two(dim0, dim1, right_node.output_shape().rank()) => {
            (input, true)
        }
        Some(_) => return Ok(None),
        None => (right, false),
    };
    let Some((cache, groups)) = graph.required_node(repeated)?.repeat_kv() else {
        return Ok(None);
    };
    let query_shape = graph.required_node(queries)?.output_shape().clone();
    let cache_shape = graph.required_node(cache)?.output_shape().clone();
    let (query, kv) = (query_shape.dimensions(), cache_shape.dimensions());
    // Both operands are four-dimensional with the same batch, and the query heads are the repeated ones.
    let folds = groups > 1
        && query.len() == 4
        && kv.len() == 4
        && query[0] == kv[0]
        && query[1] == kv[1] * groups;
    Ok(folds.then_some(GroupedMatmul {
        queries,
        cache,
        transposed,
        groups,
        query_shape,
        cache_shape,
    }))
}

impl GroupedMatmul {
    /// Pushes the folded matmul in place of `node` and returns the id of its result.
    fn push(
        &self,
        node: &NodeRef<'_>,
        id_map: &HashMap<ValueId, ValueId>,
        nodes: &mut Vec<Node>,
    ) -> Result<ValueId> {
        let remap = |id: ValueId| {
            id_map
                .get(&id)
                .copied()
                .ok_or_else(|| Error::missing_node("push"))
        };
        let (data_type, device) = (node.output_data_type(), node.output_device());
        let query = self.query_shape.dimensions();
        let kv = self.cache_shape.dimensions();
        let rows = Shape::create([query[0], kv[1], self.groups * query[2], query[3]])?;
        let queries = push(
            nodes,
            Node::create_with_shape(
                NodeKind::reshape(remap(self.queries)?, rows.clone()),
                None,
                data_type,
                rows,
                device,
            ),
        );
        let cache = remap(self.cache)?;
        let right = if self.transposed {
            let shape = Shape::create([kv[0], kv[1], kv[3], kv[2]])?;
            push(
                nodes,
                Node::create_with_shape(
                    NodeKind::transpose(cache, 2, 3),
                    None,
                    data_type,
                    shape,
                    device,
                ),
            )
        } else {
            cache
        };
        let output = node.output_shape().dimensions();
        let product_shape = Shape::create([query[0], kv[1], self.groups * query[2], output[3]])?;
        let product = push(
            nodes,
            Node::create_with_shape(
                NodeKind::Operation(GraphOperationInputs::new(
                    Operation::MatMul,
                    node.attributes(),
                    vec![queries, right],
                )),
                None,
                data_type,
                product_shape,
                device,
            ),
        );
        // The result keeps the matmul's name, source location, and output metadata.
        Ok(push(
            nodes,
            node.with_kind(NodeKind::reshape(product, node.output_shape().clone())),
        ))
    }
}

/// Returns whether `dim0` and `dim1` name the last two of `rank` dimensions, in either order.
fn last_two(dim0: i64, dim1: i64, rank: i64) -> bool {
    let normalized = |dim: i64| if dim < 0 { dim + rank } else { dim };
    let mut axes = [normalized(dim0), normalized(dim1)];
    axes.sort_unstable();
    axes == [rank - 2, rank - 1]
}

fn push(nodes: &mut Vec<Node>, node: Node) -> ValueId {
    nodes.push(node);
    ValueId::new(nodes.len() - 1)
}

/// Returns `graph` with every `scaled_dot_product_attention` written as its matmuls around a scaled, optionally causally masked softmax, and the map from old values to new ones, which is empty when the graph has none.
///
/// # Errors
///
/// Returns an error if a node's inputs cannot be remapped or a causal mask cannot be made on the attention's device.
pub fn decompose_scaled_dot_product_attention(
    graph: &Graph,
) -> Result<(Graph, Vec<(ValueId, ValueId)>)> {
    let mut id_map = HashMap::with_capacity(graph.node_count());
    let mut nodes = Vec::with_capacity(graph.node_count());
    let mut decomposed = false;
    for index in 0..graph.node_count() {
        let id = ValueId::new(index);
        let node = graph.required_node(id)?;
        let new_id = match attention_rewrite(graph, &node)? {
            Some(AttentionRewrite::Decomposed) => {
                decomposed = true;
                splice(graph, &node, &id_map, &mut nodes)?
            }
            Some(AttentionRewrite::Unrepeated {
                query,
                key,
                value,
                rest,
            }) => {
                decomposed = true;
                let remap = |id: ValueId| {
                    id_map.get(&id).copied().ok_or_else(|| {
                        Error::missing_node("decompose scaled dot product attention")
                    })
                };
                let kind = NodeKind::operation_invocation(
                    Operation::ScaledDotProductAttention,
                    node.attributes().clone(),
                    [query, key, value]
                        .into_iter()
                        .chain(rest)
                        .map(remap)
                        .collect::<Result<Vec<_>>>()?,
                );
                push(&mut nodes, node.with_kind(kind))
            }
            None => push(&mut nodes, remapped_node(&node, &id_map)?),
        };
        id_map.insert(id, new_id);
    }

    if !decomposed {
        return Ok((graph.clone(), Vec::new()));
    }

    let remapped = graph.remapped_node_pairs(&id_map)?;
    Ok((Graph::from_retained_nodes(graph, &id_map, nodes)?, remapped))
}

// What becomes of one attention node.
enum AttentionRewrite {
    // Written as matmuls around a softmax.
    Decomposed,
    // Kept for the fused kernel, reading the cache its `repeat_kv` keys and values repeat, whose heads the kernel shares in place.
    Unrepeated {
        query: ValueId,
        key: ValueId,
        value: ValueId,
        // The mask and sinks, which follow the value unchanged.
        rest: Vec<ValueId>,
    },
}

fn attention_rewrite(graph: &Graph, node: &NodeRef<'_>) -> Result<Option<AttentionRewrite>> {
    let (&[query, key, value, ref rest @ ..], OperationAttributes::ScaledDotProductAttention(_)) =
        (node.inputs(), node.attributes())
    else {
        return Ok(None);
    };
    let shape =
        |id: ValueId| -> Result<Shape> { Ok(graph.required_node(id)?.output_shape().clone()) };
    let data_type = node.output_data_type();
    if let (Some((cache_key, groups)), Some((cache_value, value_groups))) = (
        graph.required_node(key)?.repeat_kv(),
        graph.required_node(value)?.repeat_kv(),
    ) && groups == value_groups
        && spec::fuses_attention(
            data_type,
            &shape(query)?,
            &shape(cache_key)?,
            &shape(cache_value)?,
        )
    {
        return Ok(Some(AttentionRewrite::Unrepeated {
            query,
            key: cache_key,
            value: cache_value,
            rest: rest.to_vec(),
        }));
    }
    Ok(decomposes(graph, node)?.then_some(AttentionRewrite::Decomposed))
}

// Whether `node` is attention no fused kernel runs, which backends run as matmuls around a softmax.
fn decomposes(graph: &Graph, node: &NodeRef<'_>) -> Result<bool> {
    let &[query, key, value, ..] = node.inputs() else {
        return Ok(false);
    };
    let shape =
        |id: ValueId| -> Result<Shape> { Ok(graph.required_node(id)?.output_shape().clone()) };
    let (query, key, value) = (shape(query)?, shape(key)?, shape(value)?);
    Ok(query.rank() == 4
        && key.rank() == 4
        && !spec::fuses_attention(node.output_data_type(), &query, &key, &value))
}

#[cfg(test)]
mod tests;
