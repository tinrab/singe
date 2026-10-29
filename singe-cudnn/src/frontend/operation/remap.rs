use crate::{
    backend::tensor::TensorId,
    error::Result,
    frontend::{
        graph::TensorCaptureTarget,
        operation::{Operation, SdpaScoreSubgraph},
    },
};

impl Operation {
    pub(crate) fn try_map_tensor_ids(
        &self,
        remap: &dyn Fn(TensorId) -> Result<TensorId>,
        captures: &dyn Fn(TensorId) -> Result<Option<TensorCaptureTarget>>,
    ) -> Result<Self> {
        let mut operation = self.clone();
        operation.visit_tensor_ids(&mut |id, _| {
            *id = remap(*id)?;
            Ok(())
        })?;
        for subgraph in operation.score_subgraphs_mut() {
            subgraph.try_map_captures(captures)?;
        }
        Ok(operation)
    }
}

impl Operation {
    pub(crate) fn score_subgraphs(&self) -> impl Iterator<Item = &SdpaScoreSubgraph> {
        let graphs = match self {
            Self::Sdpa(operation) => [operation.score_subgraph(), None],
            Self::AttentionForward(request) => [request.config.score_subgraph(), None],
            Self::AttentionBackward(request) => [
                request.config().score_subgraph(),
                request.config().score_subgraph_bprop(),
            ],
            _ => [None, None],
        };
        graphs.into_iter().flatten()
    }
}
