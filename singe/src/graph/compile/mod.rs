mod specialization;
mod topology;

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use crate::{
    backend::registry::{CheckedOperationRequest, CudaGraphProviderSet, Operation},
    error::{Error, Result},
    graph::{
        Graph, GraphSignature, StateFreezePolicy,
        interpreter::value_releases,
        ir::{OperationId, ValueId},
        passes::GraphPassPipeline,
        run::operation_order,
        symbolic::SymbolicShapeEnv,
    },
};

/// Options used when compiling a graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphCompileOptions {
    /// Whether compilation should drop nodes unreachable from graph outputs.
    prune_dead_nodes: bool,
    // The providers a graph meant for CUDA may use, wherever it is placed, so its passes write what they cannot run as decompositions.
    cuda_providers: Option<CudaGraphProviderSet>,
    // Whether sibling matmuls run as one.
    merged_projections: bool,
}

/// Graph plus a validated execution order.
#[derive(Debug, Clone)]
pub struct CompiledGraph {
    pub(crate) analysis: Arc<GraphAnalysis>,
}

/// Immutable results of normalization and graph analysis, shared by instances.
#[derive(Debug)]
pub struct GraphAnalysis {
    pub(crate) compile_options: GraphCompileOptions,
    pub(crate) ir: Arc<Graph>,
    /// The primary result of each operation in execution order, the order backends lower values in.
    pub(crate) value_order: Vec<ValueId>,
    pub(crate) operation_order: Vec<OperationId>,
    pub(crate) operation_requests: HashMap<OperationId, CheckedOperationRequest>,
    pub(crate) release_after: Vec<Vec<ValueId>>,
    pub(crate) signature: GraphSignature,
    pub(crate) symbolic_shapes: SymbolicShapeEnv,
    // The operations that run as decompositions because an excluded provider runs them whole.
    #[cfg_attr(
        not(feature = "cuda"),
        expect(dead_code, reason = "program export reads it")
    )]
    pub(crate) decomposed: Vec<Operation>,
}

impl GraphCompileOptions {
    /// Creates the default graph compile configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            prune_dead_nodes: true,
            cuda_providers: None,
            merged_projections: false,
        }
    }

    // Runs sibling matmuls as one, when `merged`.
    #[must_use]
    pub const fn with_merged_projections(mut self, merged: bool) -> Self {
        self.merged_projections = merged;
        self
    }

    /// Returns whether compilation should drop nodes unreachable from graph outputs.
    pub(crate) fn prune_dead_nodes(self) -> bool {
        self.prune_dead_nodes
    }

    // Compiles for CUDA with only `providers`.
    #[must_use]
    pub const fn with_cuda_providers(mut self, providers: CudaGraphProviderSet) -> Self {
        self.cuda_providers = Some(providers);
        self
    }

    /// Sets whether compilation prunes nodes unreachable from graph outputs.
    #[cfg(test)]
    pub(crate) fn with_dead_node_pruning(mut self, prune_dead_nodes: bool) -> Self {
        self.prune_dead_nodes = prune_dead_nodes;
        self
    }
}

impl Default for GraphCompileOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl CompiledGraph {
    /// Returns the compile configuration retained by this prepared graph.
    #[must_use]
    pub fn compile_options(&self) -> GraphCompileOptions {
        self.analysis.compile_options
    }

    /// Returns an owned copy of the retained normalized IR.
    #[must_use]
    pub fn to_ir(&self) -> Graph {
        self.ir().clone()
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn shared_ir(&self) -> Arc<Graph> {
        Arc::clone(&self.analysis.ir)
    }

    /// Borrows the IR built during preparation.
    /// Runs never rebuild this IR.
    #[must_use]
    pub fn ir(&self) -> &Graph {
        &self.analysis.ir
    }

    /// Returns all declared inputs, including inputs removed by optimization.
    #[must_use]
    pub fn signature(&self) -> &GraphSignature {
        &self.analysis.signature
    }

    /// Returns the primary result of each operation in execution order, the order backends lower values in.
    #[must_use]
    pub fn value_order(&self) -> &[ValueId] {
        &self.analysis.value_order
    }
}

impl Graph {
    /// Compiles the graph using the default compile configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if input names are invalid or topological sorting fails.
    pub fn compile(&self) -> Result<CompiledGraph> {
        self.compile_with_options(GraphCompileOptions::new())
    }

    /// Compiles the graph using `options`.
    ///
    /// # Errors
    ///
    /// Returns an error if input names are invalid or topological sorting fails.
    pub fn compile_with_options(&self, options: GraphCompileOptions) -> Result<CompiledGraph> {
        self.compile_with_pipeline(options, &GraphPassPipeline::default())
    }

    pub(crate) fn compile_with_freeze(
        &self,
        options: GraphCompileOptions,
        freeze: StateFreezePolicy,
    ) -> Result<CompiledGraph> {
        self.compile_with_pipeline_and_freeze(options, &GraphPassPipeline::default(), freeze)
    }

    /// Compiles the graph using `options` after running `passes`.
    ///
    /// # Errors
    ///
    /// Returns an error if input names are invalid, a graph pass fails, or topological sorting fails.
    #[cfg(test)]
    pub(crate) fn compile_with_options_and_passes(
        &self,
        options: GraphCompileOptions,
        passes: &GraphPassPipeline,
    ) -> Result<CompiledGraph> {
        self.compile_with_pipeline(options, passes)
    }

    fn compile_with_pipeline(
        &self,
        options: GraphCompileOptions,
        passes: &GraphPassPipeline,
    ) -> Result<CompiledGraph> {
        self.compile_with_pipeline_and_freeze(options, passes, StateFreezePolicy::None)
    }

    fn compile_with_pipeline_and_freeze(
        &self,
        options: GraphCompileOptions,
        passes: &GraphPassPipeline,
        freeze: StateFreezePolicy,
    ) -> Result<CompiledGraph> {
        self.validate_unique_input_names()?;
        let mut signature = GraphSignature::create(self)?;
        if !self.shape_program.is_empty() {
            // Shape values and reshape attributes below become constants in this analysis.
            // Other shapes require a new logical specialization.
            signature.specialize_dimensions();
        }

        let mut source = self.resolve_shape_program()?;
        let frozen = source
            .state_kinds
            .iter()
            .filter(|(_, kind)| freeze.freezes(**kind))
            .map(|(name, _)| name.clone())
            .collect::<HashSet<_>>();
        source.rewrite_nodes(|graph, mut node| {
            if let Some((name, _, _, _)) = node.view().input_metadata()
                && frozen.contains(name)
            {
                let tensor = graph
                    .initializers
                    .get(name)
                    .cloned()
                    .ok_or_else(|| Error::missing_input("compile with pipeline and freeze"))?;
                node.replace_with_constant(tensor);
            }
            Ok(node)
        })?;
        for name in frozen {
            source.initializers.remove(&name);
            source.static_initializers.remove(&name);
            source.state_kinds.remove(&name);
        }

        source.refresh_input_bindings();
        let pipeline = passes
            .clone()
            .with_cuda_providers(options.cuda_providers)
            .with_merged_projections(options.merged_projections)
            .with_dead_node_pruning(options.prune_dead_nodes());
        let (graph, results) = pipeline.normalize(&source)?;
        let decomposed = results
            .iter()
            .flat_map(|result| result.decomposed.iter().copied())
            .collect();
        let mut ir = graph.to_ir()?;
        ir.infer_value_metadata()?;
        if options.prune_dead_nodes() {
            ir.prune_dead_operations()?;
        }

        ir.validate()?;
        let value_order = ir.primary_result_order()?;
        let operation_order = operation_order(&ir, &value_order)?;
        let operation_requests = ir.checked_requests(&operation_order)?;
        let release_after = value_releases(&ir, &operation_order)?;
        let symbolic_shapes = SymbolicShapeEnv::create_from_ir(&ir)?;

        Ok(CompiledGraph {
            analysis: Arc::new(GraphAnalysis {
                compile_options: options,
                ir: Arc::new(ir),
                value_order,
                operation_order,
                operation_requests,
                release_after,
                signature,
                symbolic_shapes,
                decomposed,
            }),
        })
    }

    fn validate_unique_input_names(&self) -> Result<HashSet<String>> {
        let mut input_names = HashSet::new();
        for (_, name, _) in self.input_entries() {
            if !input_names.insert(name.to_string()) {
                return Err(Error::DuplicateInput);
            }
        }
        Ok(input_names)
    }
}

#[cfg(test)]
mod tests;
