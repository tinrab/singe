use crate::{
    backend::tensor::TensorId,
    context::Context,
    error::{Error, Result},
    frontend::{
        graph::Graph,
        operation::{
            AttentionBackwardConfig, AttentionBackwardRequest, AttentionBackwardWorkspaceLimit,
            Operation, ReductionOperation, SdpaOperation, SdpaScoreSubgraph,
        },
    },
    utility::to_usize,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatingBackwardAccumulation {
    Probability,
    Query,
    OrderedQuery,
}

impl Graph {
    pub(crate) fn floating_backward_accumulation(
        &self,
        query: TensorId,
        key: TensorId,
        has_bias_gradient: bool,
        config: &AttentionBackwardConfig,
        sm: i32,
    ) -> Result<FloatingBackwardAccumulation> {
        if has_bias_gradient {
            return Ok(FloatingBackwardAccumulation::Probability);
        }
        if !config.deterministic_algorithm() {
            return Ok(FloatingBackwardAccumulation::Query);
        }
        if sm / 10 != 9 {
            return Ok(FloatingBackwardAccumulation::Probability);
        }
        let limit = match config.probability_gradient_workspace_limit() {
            Some(limit) => Some(limit),
            None => AttentionBackwardWorkspaceLimit::from_environment()?,
        };
        let Some(AttentionBackwardWorkspaceLimit::Bytes(cap)) = limit else {
            return Ok(FloatingBackwardAccumulation::Probability);
        };
        let query = self.tensor_spec(query)?.shape.dimensions();
        let key = self.tensor_spec(key)?.shape.dimensions();
        let padded_query = pad_sequence(query[2], 64)?;
        let padded_key = pad_sequence(key[2], 128)?;
        let required = [
            to_usize(query[0], "attention batch")?,
            to_usize(query[1], "attention heads")?,
            padded_query,
            padded_key,
            2,
        ]
        .into_iter()
        .try_fold(1_usize, |bytes, factor| {
            bytes
                .checked_mul(factor)
                .ok_or_else(|| Error::out_of_range("sdpa probability gradient workspace bytes"))
        })?;
        Ok(if cap < required {
            FloatingBackwardAccumulation::OrderedQuery
        } else {
            FloatingBackwardAccumulation::Probability
        })
    }

    pub(crate) fn attention_workspace_environment(
        &self,
        context: Option<&Context>,
    ) -> Result<Option<AttentionBackwardWorkspaceLimit>> {
        let inherited = self.inherits_attention_workspace_limit();
        if !inherited
            || self
                .compilation_sm_version(context)?
                .is_some_and(|sm| sm / 10 != 9)
        {
            return Ok(None);
        }
        AttentionBackwardWorkspaceLimit::from_environment()
    }

    fn inherits_attention_workspace_limit(&self) -> bool {
        let inherited_subgraph =
            |subgraph: &SdpaScoreSubgraph| subgraph.graph().inherits_attention_workspace_limit();
        self.operations.iter().any(|operation| match operation {
            Operation::AttentionBackward(request) => {
                let inherited = matches!(request,
                    AttentionBackwardRequest::Floating { outputs, config, .. }
                        if config.deterministic_algorithm() && outputs.bias_gradient.is_none()
                            && config.probability_gradient_workspace_limit().is_none()
                );
                let config = request.config();
                inherited
                    || config.score_subgraph().is_some_and(inherited_subgraph)
                    || config
                        .score_subgraph_bprop()
                        .is_some_and(inherited_subgraph)
            }
            Operation::AttentionForward(request) => request
                .config
                .score_subgraph()
                .is_some_and(inherited_subgraph),
            Operation::Sdpa(SdpaOperation::Forward { score_subgraph, .. }) => {
                score_subgraph.as_deref().is_some_and(inherited_subgraph)
            }
            Operation::Sdpa(SdpaOperation::Backward { config, .. }) => {
                config.score_subgraph().is_some_and(inherited_subgraph)
            }
            Operation::Reduction(ReductionOperation::Custom { body, .. }) => {
                body.inherits_attention_workspace_limit()
            }
            _ => false,
        })
    }
}

fn pad_sequence(length: i64, multiple: usize) -> Result<usize> {
    let length = to_usize(length, "attention sequence length")?;
    (length / multiple)
        .checked_add(usize::from(length % multiple != 0))
        .and_then(|blocks| blocks.checked_mul(multiple))
        .ok_or_else(|| Error::out_of_range("sdpa probability gradient workspace bytes"))
}
