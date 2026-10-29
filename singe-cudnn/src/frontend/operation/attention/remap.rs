use std::collections::BTreeSet;

use crate::{
    backend::tensor::TensorId,
    error::Result,
    frontend::{graph::TensorCaptureTarget, operation::attention::SdpaScoreSubgraph},
};

impl SdpaScoreSubgraph {
    pub(crate) fn captured_tensor_ids(&self) -> BTreeSet<TensorId> {
        let mut captures = self.graph.captured_tensor_ids();
        captures.remove(&self.input);
        captures
    }

    pub(crate) fn try_map_captures(
        &mut self,
        remap: &dyn Fn(TensorId) -> Result<Option<TensorCaptureTarget>>,
    ) -> Result<()> {
        let (graph, ids) = self.graph.try_map_captures(self.input, remap)?;
        self.input = ids.get(&self.input).copied().unwrap_or(self.input);
        self.output = ids.get(&self.output).copied().unwrap_or(self.output);
        self.graph = graph;
        Ok(())
    }
}
