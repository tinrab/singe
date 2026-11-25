// Several matmuls of one left side by different weights can run as one matmul by their weights joined, as vLLM's `qkv` and `gate_up` do, which shares the load of that left side.
//
// Weights that are already adjacent rows of one tensor, as `ModelState::join_projection_rows` stores them, merge always: the joined matmul reads them in place, as eager layers do.
// Other weights merge only when a record asks for it, since that merge copies them and can break a fusion worth more, which only timing a whole plan shows.
// The copies are built where the graph is compiled, from the values it is compiled with, so preparing with other state rebuilds them, and they are never stale.

use std::collections::HashMap;

use crate::{
    backend::registry::Operation,
    data_type::DataType,
    error::{Error, Result},
    graph::{
        Graph,
        ir::ValueId,
        node::{Node, NodeRef},
        passes::{
            decompose::{Placement, push, splice_traced_on},
            remap::{prune_dead_nodes_with_map, remapped_node},
        },
    },
    layout::Layout,
    nn::joined_rows,
    tensor::Tensor,
};

// A matmul that may join others: its id, its weight, and how many outputs that weight gives it.
struct Sibling {
    id: ValueId,
    weight: Tensor,
    outputs: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProjectionMerge {
    Joined,
    All,
}

// Matmuls that may run as one.
struct SiblingGroup {
    left: ValueId,
    // Whether the right side is the weight transposed, as a linear layer reads it.
    transposed: bool,
    siblings: Vec<Sibling>,
    joined: Option<Tensor>,
}

impl SiblingGroup {
    // The weights as one tensor in place, when they are adjacent rows of one storage; only weights read transposed join by rows.
    fn join(&mut self) {
        if !self.transposed {
            return;
        }

        self.joined = joined_rows(
            &self
                .siblings
                .iter()
                .map(|sibling| &sibling.weight)
                .collect::<Vec<_>>(),
        );
    }

    fn merged_weight(&self) -> Result<Tensor> {
        if let Some(joined) = &self.joined {
            return Ok(joined.clone());
        }

        let weights = self
            .siblings
            .iter()
            .map(|sibling| sibling.weight.clone())
            .collect::<Vec<_>>();
        Tensor::concat(&weights, i64::from(!self.transposed))
    }

    // The bytes merging copies, which are zero for weights already joined.
    #[cfg(feature = "cuda")]
    fn bytes(&self) -> u64 {
        if self.joined.is_some() {
            return 0;
        }

        self.siblings
            .iter()
            .map(|sibling| {
                u64::try_from(
                    sibling.weight.element_count() * sibling.weight.data_type().size_bytes(),
                )
                .unwrap_or(u64::MAX)
            })
            .sum()
    }
}

// The value of a graph's constant or of state only preparing can change.
// Other state stays apart, since a run may replace it or write it, and a joined copy would not see that.
fn known_value(graph: &Graph, node: &NodeRef<'_>) -> Option<Tensor> {
    if let Some(tensor) = node.constant_tensor(0) {
        return Some(tensor.clone());
    }

    let (name, ..) = node.input_metadata()?;
    if !graph.is_static_initializer(name)
        || graph.input_writes.iter().any(|written| written == name)
    {
        return None;
    }

    graph.initializers.get(name).cloned()
}

// The weight of a matmul's right side, and whether it is read transposed.
fn weight_of(graph: &Graph, right: ValueId) -> Result<Option<(Tensor, bool)>> {
    let node = graph.required_node(right)?;
    if let Some(weight) = known_value(graph, &node) {
        return Ok((weight.shape().rank() == 2).then_some((weight, false)));
    }

    if node.operation() != Some(Operation::Transpose) {
        return Ok(None);
    }

    let [input] = node.inputs() else {
        return Ok(None);
    };
    let input = graph.required_node(*input)?;
    Ok(known_value(graph, &input)
        .filter(|weight| weight.shape().rank() == 2)
        .map(|weight| (weight, true)))
}

// The groups to merge: those whose weights are joined already, and when `copies`, the rest too.
fn sibling_groups(graph: &Graph, merge: ProjectionMerge) -> Result<Vec<SiblingGroup>> {
    let mut groups = HashMap::<(ValueId, DataType, bool, i64), SiblingGroup>::new();
    for id in graph.node_ids() {
        let node = graph.required_node(id)?;
        if node.operation() != Some(Operation::MatMul) {
            continue;
        }

        let &[left, right] = node.inputs() else {
            continue;
        };
        let Some((weight, transposed)) = weight_of(graph, right)? else {
            continue;
        };
        let left_node = graph.required_node(left)?;
        let &[rows, columns] = weight.shape().dimensions() else {
            continue;
        };
        let (depth, outputs) = if transposed {
            (columns, rows)
        } else {
            (rows, columns)
        };
        // A left side of one dimension gives a result with no axis of outputs to slice.
        if left_node.output_layout().shape().rank() < 2
            || left_node.output_data_type() != weight.data_type()
            || left_node.output_device() != weight.device()
        {
            continue;
        }

        groups
            .entry((left, weight.data_type(), transposed, depth))
            .or_insert_with(|| SiblingGroup {
                left,
                transposed,
                siblings: Vec::new(),
                joined: None,
            })
            .siblings
            .push(Sibling {
                id,
                weight,
                outputs,
            });
    }

    let mut groups = groups
        .into_values()
        .filter(|group| group.siblings.len() > 1)
        .map(|mut group| {
            group.join();
            group
        })
        .filter(|group| merge == ProjectionMerge::All || group.joined.is_some())
        .collect::<Vec<_>>();
    // Which matmul comes first decides the order of what this adds, which must not depend on a map's.
    groups.sort_by_key(|group| group.siblings[0].id);
    Ok(groups)
}

// The bytes of weights a record's merge would copy, which are zero if it has nothing to merge beyond the joined weights every compile merges.
//
// # Errors
//
// Returns an error if the graph is malformed.
#[cfg(feature = "cuda")]
pub fn merged_projection_bytes(graph: &Graph) -> Result<u64> {
    Ok(sibling_groups(graph, ProjectionMerge::All)?
        .iter()
        .map(SiblingGroup::bytes)
        .sum())
}

// Returns `graph` with each group of sibling matmuls written as one matmul, and its results as views of that: groups whose weights are joined already, and when `copies`, the rest too.
pub fn merge_sibling_projections(
    graph: &Graph,
    merge: ProjectionMerge,
) -> Result<(Graph, Vec<(ValueId, ValueId)>)> {
    let groups = sibling_groups(graph, merge)?;
    if groups.is_empty() {
        return Ok((graph.clone(), Vec::new()));
    }

    let mut member = HashMap::new();
    for (index, group) in groups.iter().enumerate() {
        let mut first = 0;
        for sibling in &group.siblings {
            member.insert(sibling.id, (index, first, sibling.outputs));
            first += sibling.outputs;
        }
    }

    let mut merged = vec![None; groups.len()];
    let mut id_map = HashMap::with_capacity(graph.node_count());
    let mut nodes = Vec::with_capacity(graph.node_count());
    for id in graph.node_ids() {
        let node = graph.required_node(id)?;
        let new_id = if let Some(&(index, first, outputs)) = member.get(&id) {
            let group = &groups[index];
            let product = if let Some(product) = merged[index] {
                product
            } else {
                let product = merged_matmul(graph, group, &node, &id_map, &mut nodes)?;
                merged[index] = Some(product);
                product
            };
            let (data_type, layout) = node_metadata(&nodes, product)?;
            let axis = layout.shape().rank() - 1;
            let placement = Placement {
                device: node.output_device(),
                source_location: node.source_location(),
                contiguous: node.output_layout().is_contiguous(),
                follows_axes: false,
            };
            splice_traced_on(
                &[(data_type, layout, product)],
                &placement,
                &mut nodes,
                |inputs| Ok(vec![inputs[0].narrow(axis, first, outputs)?]),
            )?
            .0
        } else {
            push(&mut nodes, remapped_node(&node, &id_map)?)
        };
        id_map.insert(id, new_id);
    }

    let merged = Graph::from_retained_nodes(graph, &id_map, nodes)?;
    let mut remapped = graph.remapped_node_pairs(&id_map)?;
    let (pruned, moved) = prune_dead_nodes_with_map(&merged)?;
    let moved = moved.into_iter().collect::<HashMap<_, _>>();
    for (_, current) in &mut remapped {
        if let Some(&next) = moved.get(current) {
            *current = next;
        }
    }

    Ok((pruned, remapped))
}

fn node_metadata(nodes: &[Node], id: ValueId) -> Result<(DataType, Layout)> {
    let node = nodes
        .get(id.index())
        .ok_or_else(|| Error::missing_node("merged projection"))?
        .view();
    Ok((node.output_data_type(), node.output_layout().clone()))
}

fn merged_matmul(
    graph: &Graph,
    group: &SiblingGroup,
    first: &NodeRef<'_>,
    id_map: &HashMap<ValueId, ValueId>,
    nodes: &mut Vec<Node>,
) -> Result<ValueId> {
    let weight = group.merged_weight()?;
    let (data_type, layout) = (weight.data_type(), Layout::from_tensor(&weight));
    let merged = push(nodes, Node::constant_node(weight));
    let left = graph.required_node(group.left)?;
    let transposed = group.transposed;
    let placement = Placement {
        device: first.output_device(),
        source_location: first.source_location(),
        contiguous: false,
        follows_axes: false,
    };
    let (product, _) = splice_traced_on(
        &[
            (
                left.output_data_type(),
                left.output_layout().clone(),
                Graph::remapped_node(group.left, id_map)?,
            ),
            (data_type, layout, merged),
        ],
        &placement,
        nodes,
        |inputs| {
            let right = if transposed {
                inputs[1].transpose(0, 1)?
            } else {
                inputs[1].clone()
            };
            Ok(vec![inputs[0].matmul(&right)?])
        },
    )?;
    Ok(product)
}
