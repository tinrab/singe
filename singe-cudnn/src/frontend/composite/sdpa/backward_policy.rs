use crate::{
    error::{Error, Result},
    frontend::{
        graph::Graph,
        operation::{AttentionBackwardConfig, AttentionBackwardWorkspaceLimit},
    },
    tensor::TensorId,
    utility::to_usize,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FloatingBackwardAccumulation {
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

        let Some(AttentionBackwardWorkspaceLimit::Bytes(cap)) =
            config.probability_gradient_workspace_limit()
        else {
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
}

fn pad_sequence(length: i64, multiple: usize) -> Result<usize> {
    let length = to_usize(length, "attention sequence length")?;
    (length / multiple)
        .checked_add(usize::from(length % multiple != 0))
        .and_then(|blocks| blocks.checked_mul(multiple))
        .ok_or_else(|| Error::out_of_range("sdpa probability gradient workspace bytes"))
}
