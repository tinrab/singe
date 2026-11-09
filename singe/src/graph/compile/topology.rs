use crate::{
    error::Result,
    graph::{Graph, ir::ValueId, outputs::GraphOutput},
};

impl Graph {
    /// Returns reachable output dependencies in dependency order.
    ///
    /// # Errors
    ///
    /// Returns an error if an output references a missing node or if the graph contains a cycle.
    pub(crate) fn topological_sort_reachable(&self) -> Result<Vec<ValueId>> {
        let reachable = self.reachable_from_outputs()?;
        Ok(self
            .topological_sort()?
            .into_iter()
            .filter(|id| reachable[id.index()])
            .collect())
    }

    fn reachable_from_outputs(&self) -> Result<Vec<bool>> {
        let mut reachable = vec![false; self.node_count()];
        let mut stack = self
            .output_projections()
            .iter()
            .map(GraphOutput::node)
            .collect::<Vec<_>>();
        while let Some(id) = stack.pop() {
            self.ensure_node(id)?;
            if reachable[id.index()] {
                continue;
            }

            reachable[id.index()] = true;
            for &dependency in self.required_node(id)?.inputs() {
                self.ensure_node(dependency)?;
                stack.push(dependency);
            }
        }

        Ok(reachable)
    }
}
