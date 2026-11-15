use std::{cmp::Reverse, collections::HashMap};

use singe_cuda::graph::{Graph as NativeCudaGraph, GraphNode};

use crate::error::{Error, Result};

pub fn drop_implied_dependencies(graph: &mut NativeCudaGraph) -> Result<()> {
    // A graph of more than about 4,096 edges never finishes on the device, with nothing running, so a large step keeps only the edges its order needs: none that another edge implies, as a node waiting on a writer and on that writer's own source does.
    let nodes = graph.nodes()?;
    let index = nodes
        .iter()
        .enumerate()
        .map(|(position, node)| (node.as_raw() as usize, position))
        .collect::<HashMap<_, _>>();
    let position = |node: &GraphNode| {
        index
            .get(&(node.as_raw() as usize))
            .copied()
            .ok_or_else(|| Error::missing_input("a cuda graph edge's node"))
    };
    let mut incoming = vec![Vec::new(); nodes.len()];
    for edge in graph.edges()? {
        let (from, to) = (position(&edge.from)?, position(&edge.to)?);
        incoming[to].push(from);
    }

    // Kahn's order, so a node's sources come before it.
    let mut remaining = incoming.iter().map(Vec::len).collect::<Vec<_>>();
    let mut dependents = vec![Vec::new(); nodes.len()];
    for (to, sources) in incoming.iter().enumerate() {
        for &from in sources {
            dependents[from].push(to);
        }
    }

    let mut order = (0..nodes.len())
        .filter(|&node| remaining[node] == 0)
        .collect::<Vec<_>>();
    let mut next = 0;
    while next < order.len() {
        let node = order[next];
        next += 1;
        for &dependent in &dependents[node] {
            remaining[dependent] -= 1;
            if remaining[dependent] == 0 {
                order.push(dependent);
            }
        }
    }

    if order.len() != nodes.len() {
        return Err(Error::invalid_argument("a cuda graph with a cycle"));
    }

    let mut rank = vec![0; nodes.len()];
    for (place, &node) in order.iter().enumerate() {
        rank[node] = place;
    }

    // Each node's ancestors, as bits; a source already among the ancestors of a later source is implied by it.
    let words = nodes.len().div_ceil(64);
    let mut ancestors = vec![vec![0_u64; words]; nodes.len()];
    let (mut from, mut to) = (Vec::new(), Vec::new());
    for &node in &order {
        let mut sources = incoming[node].clone();
        sources.sort_unstable_by_key(|&source| Reverse(rank[source]));
        let mut reached = vec![0_u64; words];
        for source in sources {
            if reached[source / 64] & (1 << (source % 64)) != 0 {
                from.push(nodes[source].clone());
                to.push(nodes[node].clone());
                continue;
            }

            reached[source / 64] |= 1 << (source % 64);
            for (word, bits) in reached.iter_mut().zip(&ancestors[source]) {
                *word |= bits;
            }
        }

        ancestors[node] = reached;
    }

    graph.remove_dependencies(&from, &to)?;
    let edges = graph.edges()?.len();
    if edges > LARGEST_EDGES {
        return Err(Error::unsupported_operation_with_details(
            "cuda graph",
            format!("{edges} edges remain, past the {LARGEST_EDGES} a launch finishes with"),
        ));
    }
    Ok(())
}

const LARGEST_EDGES: usize = 4096;
