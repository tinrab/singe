use std::collections::{HashMap, HashSet};

use crate::{
    error::Result,
    graph::{
        Graph, GraphSignature,
        ir::ValueId,
        passes::{GraphPassKind, GraphPassPipeline, remapped_node},
    },
    tensor::Tensor,
};

impl Graph {
    /// Creates one concrete metadata specialization after checking the complete declared signature.
    /// This does not read tensor contents or execute kernels.
    ///
    /// # Errors
    ///
    /// Returns an error if input signature resolution fails, a guarded input node is missing, shape expressions cannot be evaluated, or graph metadata, initializers, node contracts, or outputs fail validation.
    pub fn specialize_runtime_input_metadata(
        &self,
        inputs: &HashMap<String, Tensor>,
    ) -> Result<Self> {
        if self.input_dimension_guards.is_empty() {
            return Ok(self.clone());
        }

        let resolved = GraphSignature::create(self)?.resolve(inputs)?;
        let guarded_inputs = self
            .input_dimension_guards
            .values()
            .map(|guard| guard.input().to_owned())
            .collect::<HashSet<_>>();
        let mut graph = self.clone();
        let input_ids = graph
            .input_entries()
            .filter(|(_, name, _)| guarded_inputs.contains(*name))
            .map(|(id, name, _)| (id, name.to_owned()))
            .collect::<Vec<_>>();
        for (id, name) in input_ids {
            if let Some(tensor) = resolved.get(&name) {
                graph.update_node(id, |node| node.update_input_metadata(tensor))?;
            }
        }

        let graph = graph.resolve_shape_program()?.retain_executable_nodes()?;
        GraphPassPipeline::new(vec![
            GraphPassKind::InferValueMetadata,
            GraphPassKind::ValidateInitializers,
            GraphPassKind::ValidateValueMetadata,
            GraphPassKind::ValidateNodeContracts,
            GraphPassKind::ValidateOutputs,
        ])
        .normalize(&graph)
        .map(|(graph, _)| graph)
    }

    /// Prunes unreachable operations while retaining every declared input.
    pub(crate) fn retain_executable_nodes(&self) -> Result<Self> {
        // Dead operations may have intentionally invalid metadata.
        // Keep the declared inputs while limiting inference to executable dependencies.
        let mut retained = self
            .topological_sort_reachable()?
            .into_iter()
            .collect::<HashSet<_>>();
        retained.extend(self.input_entries().map(|(id, _, _)| id));
        let order = self
            .topological_sort()?
            .into_iter()
            .filter(|id| retained.contains(id))
            .collect::<Vec<_>>();
        let map = order
            .iter()
            .enumerate()
            .map(|(index, old)| (*old, ValueId::new(index)))
            .collect::<HashMap<_, _>>();
        let nodes = order
            .into_iter()
            .map(|id| remapped_node(&self.required_node(id)?, &map))
            .collect::<Result<Vec<_>>>()?;
        Self::from_retained_nodes(self, &map, nodes)
    }
}
