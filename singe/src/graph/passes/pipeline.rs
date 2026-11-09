use crate::{
    backend::registry::{CudaGraphProviderSet, Operation},
    error::Result,
    graph::{
        Graph,
        ir::ValueId,
        passes::{
            decompose_operations, decompose_scaled_dot_product_attention, elide_contiguous_copies,
            fold_grouped_query_attention, fold_small_constants, infer_value_metadata,
            materialize_contiguous_operands, merge_sibling_projections, prune_dead_nodes_with_map,
            recognize_row_operations, remove_redundant_ops_with_map,
            validation::{
                validate_initializers, validate_node_contracts, validate_outputs,
                validate_value_metadata,
            },
        },
    },
};

/// Ordered backend-neutral graph pass pipeline.
#[derive(Debug, Clone)]
pub struct GraphPassPipeline {
    passes: Vec<GraphPassKind>,
    cuda_providers: Option<CudaGraphProviderSet>,
}

impl GraphPassPipeline {
    /// Creates a graph pass pipeline from pass identifiers.
    pub fn new(passes: Vec<GraphPassKind>) -> Self {
        Self {
            passes,
            cuda_providers: None,
        }
    }

    // Compiles for CUDA with only `providers`, wherever the graph is placed.
    pub(crate) const fn with_cuda_providers(
        mut self,
        providers: Option<CudaGraphProviderSet>,
    ) -> Self {
        self.cuda_providers = providers;
        self
    }

    // Returns a pipeline that also runs sibling matmuls as one, before anything fuses them.
    pub(crate) fn with_merged_projections(self, merged: bool) -> Self {
        if !merged {
            return self;
        }

        let cuda_providers = self.cuda_providers;
        let mut passes = Vec::with_capacity(self.passes.len() + 1);
        for pass in self.passes {
            if pass == GraphPassKind::DecomposeOperations {
                passes.push(GraphPassKind::MergeSiblingProjections);
            }

            passes.push(pass);
        }

        Self::new(passes).with_cuda_providers(cuda_providers)
    }

    /// Returns a pipeline that prunes dead nodes before and after rewrite passes when requested.
    pub fn with_dead_node_pruning(self, prune_dead_nodes: bool) -> Self {
        if !prune_dead_nodes || self.passes.contains(&GraphPassKind::PruneDeadNodes) {
            return self;
        }

        let cuda_providers = self.cuda_providers;
        // Pruning first keeps dead operations, whose metadata may be invalid, out of validation; compilation prunes what the rewrites leave dead on the IR.
        let mut passes = Vec::with_capacity(self.passes.len() + 1);
        passes.push(GraphPassKind::PruneDeadNodes);
        passes.extend_from_slice(&self.passes);
        Self::new(passes).with_cuda_providers(cuda_providers)
    }

    /// Applies rewrite passes and validates the resulting graph.
    ///
    /// # Errors
    ///
    /// Returns an error if a rewrite pass cannot normalize the graph or if any validation pass finds invalid graph metadata.
    pub fn normalize(&self, graph: &Graph) -> Result<(Graph, Vec<GraphPassResult>)> {
        let mut graph = graph.clone();
        let mut results = Vec::with_capacity(self.passes.len());
        for &pass in &self.passes {
            match pass {
                GraphPassKind::PruneDeadNodes => {
                    let (pruned, remapped_nodes) = prune_dead_nodes_with_map(&graph)?;
                    graph = pruned;
                    results.push(GraphPassResult::new(
                        pass,
                        graph.node_count(),
                        remapped_nodes,
                    ));
                }
                GraphPassKind::MergeSiblingProjections => {
                    let checked_nodes = graph.node_count();
                    let (merged, remapped_nodes) = merge_sibling_projections(&graph)?;
                    graph = merged;
                    results.push(GraphPassResult::new(pass, checked_nodes, remapped_nodes));
                }
                GraphPassKind::FoldSmallConstants => {
                    let checked_nodes = graph.node_count();
                    graph = fold_small_constants(&graph)?;
                    results.push(GraphPassResult::new(pass, checked_nodes, Vec::new()));
                }
                GraphPassKind::RemoveRedundantOps => {
                    let checked_nodes = graph.node_count();
                    let (simplified, remapped_nodes) = remove_redundant_ops_with_map(&graph)?;
                    graph = simplified;
                    results.push(GraphPassResult::new(pass, checked_nodes, remapped_nodes));
                }
                GraphPassKind::RecognizeRowOperations => {
                    let checked_nodes = graph.node_count();
                    let (recognized, remapped_nodes) = recognize_row_operations(&graph)?;
                    graph = recognized;
                    results.push(GraphPassResult::new(pass, checked_nodes, remapped_nodes));
                }
                GraphPassKind::DecomposeAttention => {
                    let checked_nodes = graph.node_count();
                    let (decomposed, remapped_nodes) =
                        decompose_scaled_dot_product_attention(&graph)?;
                    graph = decomposed;
                    results.push(GraphPassResult::new(pass, checked_nodes, remapped_nodes));
                }
                GraphPassKind::FoldGroupedQueryAttention => {
                    let checked_nodes = graph.node_count();
                    let (folded, remapped_nodes) = fold_grouped_query_attention(&graph)?;
                    graph = folded;
                    results.push(GraphPassResult::new(pass, checked_nodes, remapped_nodes));
                }
                GraphPassKind::DecomposeOperations => {
                    let checked_nodes = graph.node_count();
                    let (decomposed, remapped_nodes, operations) =
                        decompose_operations(&graph, self.cuda_providers)?;
                    graph = decomposed;
                    let mut result = GraphPassResult::new(pass, checked_nodes, remapped_nodes);
                    result.decomposed = operations;
                    results.push(result);
                }
                GraphPassKind::ElideContiguousCopies => {
                    let checked_nodes = graph.node_count();
                    let (elided, remapped_nodes) =
                        elide_contiguous_copies(&graph, self.cuda_providers.is_some())?;
                    graph = elided;
                    results.push(GraphPassResult::new(pass, checked_nodes, remapped_nodes));
                }
                GraphPassKind::MaterializeContiguousOperands => {
                    let checked_nodes = graph.node_count();
                    let (copied, remapped_nodes) = materialize_contiguous_operands(&graph)?;
                    graph = copied;
                    results.push(GraphPassResult::new(pass, checked_nodes, remapped_nodes));
                }
                GraphPassKind::InferValueMetadata => {
                    let (inferred, remapped_nodes) = infer_value_metadata(&graph)?;
                    graph = inferred;
                    results.push(GraphPassResult::new(
                        pass,
                        graph.node_count(),
                        remapped_nodes,
                    ));
                }
                _ => results.push(pass.run(&graph)?),
            }
        }
        Ok((graph, results))
    }
}

impl Default for GraphPassPipeline {
    fn default() -> Self {
        Self::new(vec![
            GraphPassKind::ValidateInitializers,
            GraphPassKind::ValidateValueMetadata,
            GraphPassKind::ValidateNodeContracts,
            GraphPassKind::ValidateOutputs,
            GraphPassKind::DecomposeOperations,
            GraphPassKind::FoldSmallConstants,
            GraphPassKind::RemoveRedundantOps,
            GraphPassKind::RecognizeRowOperations,
            GraphPassKind::DecomposeAttention,
            GraphPassKind::FoldGroupedQueryAttention,
            GraphPassKind::ElideContiguousCopies,
            GraphPassKind::MaterializeContiguousOperands,
        ])
    }
}

/// Backend-neutral graph pass identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphPassKind {
    /// Rewrites the graph to contain only nodes reachable from graph outputs.
    PruneDeadNodes,
    /// Folds small CPU constant-only arithmetic subgraphs.
    FoldSmallConstants,
    /// Removes metadata-preserving no-op operations.
    RemoveRedundantOps,
    /// Rewrites primitive spellings of RMS norm, layer norm, softmax, and log-softmax as those operations.
    RecognizeRowOperations,
    /// Writes `scaled_dot_product_attention` as its matmuls around a scaled, optionally masked softmax.
    DecomposeAttention,
    /// Folds grouped-query attention's repeated keys and values into the rows of its matmuls.
    FoldGroupedQueryAttention,
    /// Writes the operations compiled graphs have no kernel for as their decompositions.
    DecomposeOperations,
    // Writes matmuls of one left side by different weights as one matmul by their weights joined.
    MergeSiblingProjections,
    /// Removes explicit contiguous copies whose readers read strided operands with the same generated kernels.
    ElideContiguousCopies,
    /// Copies the strided operands of operations whose kernels read contiguous operands.
    MaterializeContiguousOperands,
    /// Infers value metadata and inserts copies for incompatible reshape views.
    InferValueMetadata,
    /// Validates graph input initializers against declared input metadata.
    ValidateInitializers,
    /// Validates node output shape/layout metadata and alias/view consistency.
    ValidateValueMetadata,
    /// Validates node-kind-declared dtype, device, shape, and dependency metadata.
    ValidateNodeContracts,
    /// Validates graph output node ids.
    ValidateOutputs,
}

/// Result returned by one graph pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphPassResult {
    /// Pass name.
    pub pass_name: &'static str,
    /// Number of nodes inspected by the pass.
    pub checked_nodes: usize,
    /// Node id remapping produced by the pass.
    pub remapped_nodes: Vec<(ValueId, ValueId)>,
    // The operations the pass split only because an excluded provider runs them whole.
    pub decomposed: Vec<Operation>,
}

impl GraphPassResult {
    pub(crate) fn new(
        pass: GraphPassKind,
        checked_nodes: usize,
        remapped_nodes: Vec<(ValueId, ValueId)>,
    ) -> Self {
        Self {
            pass_name: pass.name(),
            checked_nodes,
            remapped_nodes,
            decomposed: Vec::new(),
        }
    }
}

impl GraphPassKind {
    /// Returns a stable pass name.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::PruneDeadNodes => "prune_dead_nodes",
            Self::FoldSmallConstants => "fold_small_constants",
            Self::RemoveRedundantOps => "remove_redundant_ops",
            Self::RecognizeRowOperations => "recognize_row_operations",
            Self::DecomposeAttention => "decompose_scaled_dot_product_attention",
            Self::FoldGroupedQueryAttention => "fold_grouped_query_attention",
            Self::DecomposeOperations => "decompose_operations",
            Self::MergeSiblingProjections => "merge_sibling_projections",
            Self::ElideContiguousCopies => "elide_contiguous_copies",
            Self::MaterializeContiguousOperands => "materialize_contiguous_operands",
            Self::InferValueMetadata => "infer_value_metadata",
            Self::ValidateInitializers => "validate_initializers",
            Self::ValidateValueMetadata => "validate_value_metadata",
            Self::ValidateNodeContracts => "validate_node_contracts",
            Self::ValidateOutputs => "validate_outputs",
        }
    }
}

impl GraphPassKind {
    /// Runs this pass on `graph`.
    ///
    /// # Errors
    ///
    /// Returns an error if the pass finds invalid graph metadata.
    pub(crate) fn run(self, graph: &Graph) -> Result<GraphPassResult> {
        let (checked_nodes, remapped_nodes) = match self {
            Self::PruneDeadNodes => {
                let (graph, remapped_nodes) = prune_dead_nodes_with_map(graph)?;
                (graph.node_count(), remapped_nodes)
            }
            Self::MergeSiblingProjections => {
                let checked_nodes = graph.node_count();
                let (_, remapped_nodes) = merge_sibling_projections(graph)?;
                (checked_nodes, remapped_nodes)
            }
            Self::FoldSmallConstants => (fold_small_constants(graph)?.node_count(), Vec::new()),
            Self::RemoveRedundantOps => {
                let checked_nodes = graph.node_count();
                let (_, remapped_nodes) = remove_redundant_ops_with_map(graph)?;
                (checked_nodes, remapped_nodes)
            }
            Self::RecognizeRowOperations => {
                let checked_nodes = graph.node_count();
                let (_, remapped_nodes) = recognize_row_operations(graph)?;
                (checked_nodes, remapped_nodes)
            }
            Self::DecomposeAttention => {
                let checked_nodes = graph.node_count();
                let (_, remapped_nodes) = decompose_scaled_dot_product_attention(graph)?;
                (checked_nodes, remapped_nodes)
            }
            Self::FoldGroupedQueryAttention => {
                let checked_nodes = graph.node_count();
                let (_, remapped_nodes) = fold_grouped_query_attention(graph)?;
                (checked_nodes, remapped_nodes)
            }
            Self::DecomposeOperations => {
                let checked_nodes = graph.node_count();
                let (_, remapped_nodes, _) = decompose_operations(graph, None)?;
                (checked_nodes, remapped_nodes)
            }
            Self::ElideContiguousCopies => {
                let checked_nodes = graph.node_count();
                let (_, remapped_nodes) = elide_contiguous_copies(graph, false)?;
                (checked_nodes, remapped_nodes)
            }
            Self::MaterializeContiguousOperands => {
                let checked_nodes = graph.node_count();
                let (_, remapped_nodes) = materialize_contiguous_operands(graph)?;
                (checked_nodes, remapped_nodes)
            }
            Self::InferValueMetadata => {
                let (graph, remapped_nodes) = infer_value_metadata(graph)?;
                (graph.node_count(), remapped_nodes)
            }
            Self::ValidateInitializers => (validate_initializers(graph)?, Vec::new()),
            Self::ValidateValueMetadata => (validate_value_metadata(graph)?, Vec::new()),
            Self::ValidateNodeContracts => (validate_node_contracts(graph)?, Vec::new()),
            Self::ValidateOutputs => (validate_outputs(graph)?, Vec::new()),
        };
        Ok(GraphPassResult::new(self, checked_nodes, remapped_nodes))
    }
}
