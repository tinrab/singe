use std::collections::{BTreeMap, BTreeSet, HashSet, btree_map::Entry};

use crate::{
    backend::{execution::tensor_ops::ReshapeMode, tensor::TensorId},
    frontend::{
        graph::{Graph, TensorRecord, ValidatedGraph, subgraphs::ScoreSubgraph},
        operation::{
            ConcatInPlaceMode, Operation, ReductionOperation, TensorOperation, TensorUses,
        },
    },
};

#[derive(Clone, Copy)]
enum Producer {
    Overwrite,
    PartialOrAccumulating,
    Multiple,
}

impl Graph {
    /// Separates entry-state reads from reads of a value produced completely within the graph.
    /// Aliases keep their own identities because independent runtime layouts can address different elements.
    pub(crate) fn annotate_initial_reads(
        &self,
        tensors: &mut BTreeMap<TensorId, TensorRecord>,
        subgraphs: &BTreeMap<usize, ScoreSubgraph<ValidatedGraph>>,
    ) {
        let mut reads = BTreeSet::new();
        let mut producers = BTreeMap::new();
        collect_uses(self, subgraphs, &mut reads, &mut producers);
        for (&id, record) in tensors {
            record.reads_initial_state = !record.tensor.is_virtual()
                && reads.contains(&id)
                && !matches!(producers.get(&id), Some(Producer::Overwrite));
        }
    }
}

fn collect_uses(
    graph: &Graph,
    subgraphs: &BTreeMap<usize, ScoreSubgraph<ValidatedGraph>>,
    reads: &mut BTreeSet<TensorId>,
    producers: &mut BTreeMap<TensorId, Producer>,
) {
    for operation in &graph.operations {
        let uses = match operation {
            Operation::Sdpa(operation) => operation.native_tensor_uses(),
            operation => operation.tensor_uses(graph.config.data_type_policy),
        };
        reads.extend(uses.reads.iter().copied());
        let mut pending = uses.reads.union(&uses.writes).copied().collect::<Vec<_>>();
        let mut visited = HashSet::new();
        while let Some(id) = pending.pop() {
            if !visited.insert(id) {
                continue;
            }
            if let Some(offset) = graph
                .tensors
                .get(&id)
                .and_then(|tensor| tensor.ragged_offset)
            {
                reads.insert(offset);
                pending.push(offset);
            }
        }
        let full_write = fully_writes_dense_outputs(operation, graph, &uses);
        for id in &uses.writes {
            let producer = if full_write && !uses.reads.contains(id) {
                Producer::Overwrite
            } else {
                Producer::PartialOrAccumulating
            };
            match producers.entry(*id) {
                Entry::Vacant(entry) => {
                    entry.insert(producer);
                }
                Entry::Occupied(mut entry) => {
                    entry.insert(Producer::Multiple);
                }
            }
        }
    }
    // Native score graphs share identities with their parent. Custom reducer slots do not.
    for child in subgraphs.values() {
        collect_uses(
            &child.graph.expanded,
            &child.graph.subgraphs,
            reads,
            producers,
        );
    }
}

fn fully_writes_dense_outputs(operation: &Operation, graph: &Graph, uses: &TensorUses) -> bool {
    let full_write = match operation {
        Operation::Matmul(operation) => {
            operation.config.m_override().is_none()
                && operation.config.n_override().is_none()
                && operation.config.k_override().is_none()
        }
        Operation::Tensor(TensorOperation::Concat { config, .. }) => {
            config.in_place == ConcatInPlaceMode::None
        }
        Operation::Pointwise(_)
        | Operation::Convolution(_)
        | Operation::Normalization(_)
        | Operation::RandomNumberGenerator(_)
        | Operation::GenStats(_)
        | Operation::Rope(_)
        | Operation::Resample(_)
        | Operation::Reduction(ReductionOperation::Reduce {
            group_offset: None, ..
        })
        | Operation::Tensor(TensorOperation::Slice { .. } | TensorOperation::Transpose { .. }) => {
            true
        }
        Operation::Tensor(TensorOperation::Reshape { config, .. }) => {
            config.mode() == ReshapeMode::Logical
        }
        // Variable-length attention, indexed routing, segmented reductions, view aliases and
        // custom reducers can preserve destination elements or depend on caller-established state.
        _ => false,
    };
    full_write
        && uses.reads.union(&uses.writes).all(|id| {
            graph
                .tensors
                .get(id)
                .is_some_and(|tensor| tensor.ragged_offset.is_none())
        })
}
