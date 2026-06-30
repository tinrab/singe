mod forward;
mod support;

use std::collections::BTreeMap;

use crate::{frontend::graph::Graph, tensor::TensorId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SdpaInputs {
    pub query: TensorId,
    pub key: TensorId,
    pub value: TensorId,
    pub scale: TensorId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SdpaOutputTensors {
    pub output: TensorId,
    pub stats: Option<TensorId>,
    pub logit_max: Option<TensorId>,
    pub score_sum_exp: Option<TensorId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SdpaOutputs {
    output: TensorId,
    stats: Option<TensorId>,
}

impl SdpaOutputs {
    pub fn new(output: TensorId, stats: Option<TensorId>) -> Self {
        Self { output, stats }
    }

    pub fn output(self) -> TensorId {
        self.output
    }

    pub fn stats(self) -> Option<TensorId> {
        self.stats
    }
}

impl From<SdpaAuxOutputs> for SdpaOutputs {
    fn from(value: SdpaAuxOutputs) -> Self {
        Self {
            output: value.output,
            stats: value.stats,
        }
    }
}

impl SdpaOutputTensors {
    pub fn new(output: TensorId) -> Self {
        Self {
            output,
            stats: None,
            logit_max: None,
            score_sum_exp: None,
        }
    }

    pub fn with_stats(mut self, stats: TensorId) -> Self {
        self.stats = Some(stats);
        self
    }

    pub fn with_logit_max(mut self, logit_max: TensorId) -> Self {
        self.logit_max = Some(logit_max);
        self
    }

    pub fn with_score_sum_exp(mut self, score_sum_exp: TensorId) -> Self {
        self.score_sum_exp = Some(score_sum_exp);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SdpaBackwardInputs {
    pub query: TensorId,
    pub key: TensorId,
    pub value: TensorId,
    pub output: TensorId,
    pub output_gradient: TensorId,
    pub stats: TensorId,
    pub scale: TensorId,
}

impl SdpaBackwardInputs {
    pub fn new(
        query: TensorId,
        key: TensorId,
        value: TensorId,
        output: TensorId,
        output_gradient: TensorId,
        stats: TensorId,
        scale: TensorId,
    ) -> Self {
        Self {
            query,
            key,
            value,
            output,
            output_gradient,
            stats,
            scale,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SdpaBackwardGradientTensors {
    pub query_gradient: TensorId,
    pub key_gradient: TensorId,
    pub value_gradient: TensorId,
    pub bias_gradient: Option<TensorId>,
    pub rng_dump: Option<TensorId>,
    pub sink_token_gradient: Option<TensorId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SdpaBackwardOutputs {
    query_gradient: TensorId,
    key_gradient: TensorId,
    value_gradient: TensorId,
    bias_gradient: Option<TensorId>,
}

impl From<SdpaBackwardAuxOutputs> for SdpaBackwardOutputs {
    fn from(value: SdpaBackwardAuxOutputs) -> Self {
        Self {
            query_gradient: value.query_gradient,
            key_gradient: value.key_gradient,
            value_gradient: value.value_gradient,
            bias_gradient: value.bias_gradient,
        }
    }
}

impl SdpaBackwardOutputs {
    pub fn query_gradient(self) -> TensorId {
        self.query_gradient
    }

    pub fn key_gradient(self) -> TensorId {
        self.key_gradient
    }

    pub fn value_gradient(self) -> TensorId {
        self.value_gradient
    }

    pub fn bias_gradient(self) -> Option<TensorId> {
        self.bias_gradient
    }
}

impl SdpaBackwardGradientTensors {
    pub fn with_bias_gradient(mut self, bias_gradient: TensorId) -> Self {
        self.bias_gradient = Some(bias_gradient);
        self
    }

    pub fn with_rng_dump(mut self, rng_dump: TensorId) -> Self {
        self.rng_dump = Some(rng_dump);
        self
    }

    pub fn with_sink_token_gradient(mut self, sink_token_gradient: TensorId) -> Self {
        self.sink_token_gradient = Some(sink_token_gradient);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SdpaFp8Inputs {
    pub query: TensorId,
    pub key: TensorId,
    pub value: TensorId,
    pub descale_query: TensorId,
    pub descale_key: TensorId,
    pub descale_value: TensorId,
    pub descale_scores: TensorId,
    pub scale_scores: TensorId,
    pub scale_output: TensorId,
}

impl SdpaFp8Inputs {
    pub fn new(
        query: TensorId,
        key: TensorId,
        value: TensorId,
        descale_query: TensorId,
        descale_key: TensorId,
        descale_value: TensorId,
        descale_scores: TensorId,
        scale_scores: TensorId,
        scale_output: TensorId,
    ) -> Self {
        Self {
            query,
            key,
            value,
            descale_query,
            descale_key,
            descale_value,
            descale_scores,
            scale_scores,
            scale_output,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SdpaFp8OutputTensors {
    pub output: TensorId,
    pub stats: Option<TensorId>,
    pub absolute_max_scores: Option<TensorId>,
    pub absolute_max_output: Option<TensorId>,
}

impl SdpaFp8OutputTensors {
    pub fn new(output: TensorId) -> Self {
        Self {
            output,
            stats: None,
            absolute_max_scores: None,
            absolute_max_output: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SdpaMxfp8Inputs {
    pub query: TensorId,
    pub key: TensorId,
    pub value: TensorId,
    pub scale_query: TensorId,
    pub scale_key: TensorId,
    pub scale_value: TensorId,
    pub attention_scale: TensorId,
}

impl SdpaMxfp8Inputs {
    pub fn new(
        query: TensorId,
        key: TensorId,
        value: TensorId,
        scale_query: TensorId,
        scale_key: TensorId,
        scale_value: TensorId,
        attention_scale: TensorId,
    ) -> Self {
        Self {
            query,
            key,
            value,
            scale_query,
            scale_key,
            scale_value,
            attention_scale,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SdpaMxfp8OutputTensors {
    pub output: TensorId,
    pub stats: Option<TensorId>,
    pub absolute_max_output: Option<TensorId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SdpaMxfp8AuxOutputs {
    output: TensorId,
    stats: Option<TensorId>,
    absolute_max_output: Option<TensorId>,
}

impl SdpaMxfp8AuxOutputs {
    pub fn new(
        output: TensorId,
        stats: Option<TensorId>,
        absolute_max_output: Option<TensorId>,
    ) -> Self {
        Self {
            output,
            stats,
            absolute_max_output,
        }
    }

    pub fn output(self) -> TensorId {
        self.output
    }

    pub fn stats(self) -> Option<TensorId> {
        self.stats
    }

    pub fn absolute_max_output(self) -> Option<TensorId> {
        self.absolute_max_output
    }
}

impl SdpaMxfp8OutputTensors {
    pub fn new(output: TensorId) -> Self {
        Self {
            output,
            stats: None,
            absolute_max_output: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SdpaFp8BackwardInputs {
    pub query: TensorId,
    pub key: TensorId,
    pub value: TensorId,
    pub output: TensorId,
    pub output_gradient: TensorId,
    pub stats: TensorId,
    pub attention_scale: TensorId,
    pub descale_query: TensorId,
    pub descale_key: TensorId,
    pub descale_value: TensorId,
    pub descale_output: TensorId,
    pub descale_output_gradient: TensorId,
    pub descale_scores: TensorId,
    pub descale_probability_gradient: TensorId,
    pub scale_scores: TensorId,
    pub scale_query_gradient: TensorId,
    pub scale_key_gradient: TensorId,
    pub scale_value_gradient: TensorId,
    pub scale_probability_gradient: TensorId,
}

impl SdpaFp8BackwardInputs {
    pub fn new(
        query: TensorId,
        key: TensorId,
        value: TensorId,
        output: TensorId,
        output_gradient: TensorId,
        stats: TensorId,
        attention_scale: TensorId,
        descale_query: TensorId,
        descale_key: TensorId,
        descale_value: TensorId,
        descale_output: TensorId,
        descale_output_gradient: TensorId,
        descale_scores: TensorId,
        descale_probability_gradient: TensorId,
        scale_scores: TensorId,
        scale_query_gradient: TensorId,
        scale_key_gradient: TensorId,
        scale_value_gradient: TensorId,
        scale_probability_gradient: TensorId,
    ) -> Self {
        Self {
            query,
            key,
            value,
            output,
            output_gradient,
            stats,
            attention_scale,
            descale_query,
            descale_key,
            descale_value,
            descale_output,
            descale_output_gradient,
            descale_scores,
            descale_probability_gradient,
            scale_scores,
            scale_query_gradient,
            scale_key_gradient,
            scale_value_gradient,
            scale_probability_gradient,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SdpaFp8BackwardGradientTensors {
    pub query_gradient: TensorId,
    pub key_gradient: TensorId,
    pub value_gradient: TensorId,
    pub sink_token_gradient: Option<TensorId>,
    pub absolute_max_query_gradient: TensorId,
    pub absolute_max_key_gradient: TensorId,
    pub absolute_max_value_gradient: TensorId,
    pub absolute_max_probability_gradient: TensorId,
}

impl SdpaFp8BackwardGradientTensors {
    pub fn new(
        query_gradient: TensorId,
        key_gradient: TensorId,
        value_gradient: TensorId,
        absolute_max_query_gradient: TensorId,
        absolute_max_key_gradient: TensorId,
        absolute_max_value_gradient: TensorId,
        absolute_max_probability_gradient: TensorId,
    ) -> Self {
        Self {
            query_gradient,
            key_gradient,
            value_gradient,
            sink_token_gradient: None,
            absolute_max_query_gradient,
            absolute_max_key_gradient,
            absolute_max_value_gradient,
            absolute_max_probability_gradient,
        }
    }

    pub fn with_sink_token_gradient(mut self, sink_token_gradient: TensorId) -> Self {
        self.sink_token_gradient = Some(sink_token_gradient);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SdpaMxfp8BackwardInputs {
    pub query: TensorId,
    pub query_transposed: TensorId,
    pub key: TensorId,
    pub key_transposed: TensorId,
    pub value: TensorId,
    pub output: TensorId,
    pub output_gradient: TensorId,
    pub quantized_output_gradient: TensorId,
    pub output_gradient_transposed: TensorId,
    pub stats: TensorId,
    pub scale_query: TensorId,
    pub scale_query_transposed: TensorId,
    pub scale_key: TensorId,
    pub scale_key_transposed: TensorId,
    pub scale_value: TensorId,
    pub scale_output_gradient: TensorId,
    pub scale_output_gradient_transposed: TensorId,
    pub attention_scale: TensorId,
}

impl SdpaMxfp8BackwardInputs {
    pub fn new(
        query: TensorId,
        query_transposed: TensorId,
        key: TensorId,
        key_transposed: TensorId,
        value: TensorId,
        output: TensorId,
        output_gradient: TensorId,
        quantized_output_gradient: TensorId,
        output_gradient_transposed: TensorId,
        stats: TensorId,
        scale_query: TensorId,
        scale_query_transposed: TensorId,
        scale_key: TensorId,
        scale_key_transposed: TensorId,
        scale_value: TensorId,
        scale_output_gradient: TensorId,
        scale_output_gradient_transposed: TensorId,
        attention_scale: TensorId,
    ) -> Self {
        Self {
            query,
            query_transposed,
            key,
            key_transposed,
            value,
            output,
            output_gradient,
            quantized_output_gradient,
            output_gradient_transposed,
            stats,
            scale_query,
            scale_query_transposed,
            scale_key,
            scale_key_transposed,
            scale_value,
            scale_output_gradient,
            scale_output_gradient_transposed,
            attention_scale,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SdpaMxfp8BackwardGradientTensors {
    pub query_gradient: TensorId,
    pub key_gradient: TensorId,
    pub value_gradient: TensorId,
    pub absolute_max_query_gradient: TensorId,
    pub absolute_max_key_gradient: TensorId,
    pub absolute_max_value_gradient: TensorId,
}

impl SdpaMxfp8BackwardGradientTensors {
    pub fn new(
        query_gradient: TensorId,
        key_gradient: TensorId,
        value_gradient: TensorId,
        absolute_max_query_gradient: TensorId,
        absolute_max_key_gradient: TensorId,
        absolute_max_value_gradient: TensorId,
    ) -> Self {
        Self {
            query_gradient,
            key_gradient,
            value_gradient,
            absolute_max_query_gradient,
            absolute_max_key_gradient,
            absolute_max_value_gradient,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SdpaAuxOutputs {
    output: TensorId,
    stats: Option<TensorId>,
    logit_max: Option<TensorId>,
    score_sum_exp: Option<TensorId>,
    rng_dump: Option<TensorId>,
}

impl SdpaAuxOutputs {
    pub fn new(output: TensorId, stats: Option<TensorId>) -> Self {
        Self {
            output,
            stats,
            logit_max: None,
            score_sum_exp: None,
            rng_dump: None,
        }
    }

    pub fn with_aux(
        output: TensorId,
        stats: Option<TensorId>,
        logit_max: Option<TensorId>,
        score_sum_exp: Option<TensorId>,
        rng_dump: Option<TensorId>,
    ) -> Self {
        Self {
            output,
            stats,
            logit_max,
            score_sum_exp,
            rng_dump,
        }
    }

    pub fn output(self) -> TensorId {
        self.output
    }

    pub fn stats(self) -> Option<TensorId> {
        self.stats
    }

    pub fn logit_max(self) -> Option<TensorId> {
        self.logit_max
    }

    pub fn score_sum_exp(self) -> Option<TensorId> {
        self.score_sum_exp
    }

    pub fn rng_dump(self) -> Option<TensorId> {
        self.rng_dump
    }

    pub fn absolute_max_scores(self) -> Option<TensorId> {
        self.logit_max
    }

    pub fn absolute_max_output(self) -> Option<TensorId> {
        self.score_sum_exp
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SdpaBackwardAuxOutputs {
    query_gradient: TensorId,
    key_gradient: TensorId,
    value_gradient: TensorId,
    bias_gradient: Option<TensorId>,
    rng_dump: Option<TensorId>,
    sink_token_gradient: Option<TensorId>,
}

impl SdpaBackwardAuxOutputs {
    pub fn new(
        query_gradient: TensorId,
        key_gradient: TensorId,
        value_gradient: TensorId,
        bias_gradient: Option<TensorId>,
        rng_dump: Option<TensorId>,
        sink_token_gradient: Option<TensorId>,
    ) -> Self {
        Self {
            query_gradient,
            key_gradient,
            value_gradient,
            bias_gradient,
            rng_dump,
            sink_token_gradient,
        }
    }

    pub fn query_gradient(self) -> TensorId {
        self.query_gradient
    }

    pub fn key_gradient(self) -> TensorId {
        self.key_gradient
    }

    pub fn value_gradient(self) -> TensorId {
        self.value_gradient
    }

    pub fn bias_gradient(self) -> Option<TensorId> {
        self.bias_gradient
    }

    pub fn rng_dump(self) -> Option<TensorId> {
        self.rng_dump
    }

    pub fn sink_token_gradient(self) -> Option<TensorId> {
        self.sink_token_gradient
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SdpaFp8BackwardAuxOutputs {
    query_gradient: TensorId,
    key_gradient: TensorId,
    value_gradient: TensorId,
    sink_token_gradient: Option<TensorId>,
    absolute_max_query_gradient: TensorId,
    absolute_max_key_gradient: TensorId,
    absolute_max_value_gradient: TensorId,
    absolute_max_probability_gradient: TensorId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SdpaFp8BackwardOutputs {
    query_gradient: TensorId,
    key_gradient: TensorId,
    value_gradient: TensorId,
    absolute_max_query_gradient: TensorId,
    absolute_max_key_gradient: TensorId,
    absolute_max_value_gradient: TensorId,
    absolute_max_probability_gradient: TensorId,
}

impl From<SdpaFp8BackwardAuxOutputs> for SdpaFp8BackwardOutputs {
    fn from(value: SdpaFp8BackwardAuxOutputs) -> Self {
        Self {
            query_gradient: value.query_gradient,
            key_gradient: value.key_gradient,
            value_gradient: value.value_gradient,
            absolute_max_query_gradient: value.absolute_max_query_gradient,
            absolute_max_key_gradient: value.absolute_max_key_gradient,
            absolute_max_value_gradient: value.absolute_max_value_gradient,
            absolute_max_probability_gradient: value.absolute_max_probability_gradient,
        }
    }
}

impl SdpaFp8BackwardOutputs {
    pub fn query_gradient(self) -> TensorId {
        self.query_gradient
    }

    pub fn key_gradient(self) -> TensorId {
        self.key_gradient
    }

    pub fn value_gradient(self) -> TensorId {
        self.value_gradient
    }

    pub fn absolute_max_query_gradient(self) -> TensorId {
        self.absolute_max_query_gradient
    }

    pub fn absolute_max_key_gradient(self) -> TensorId {
        self.absolute_max_key_gradient
    }

    pub fn absolute_max_value_gradient(self) -> TensorId {
        self.absolute_max_value_gradient
    }

    pub fn absolute_max_probability_gradient(self) -> TensorId {
        self.absolute_max_probability_gradient
    }
}

impl SdpaFp8BackwardAuxOutputs {
    pub fn new(
        query_gradient: TensorId,
        key_gradient: TensorId,
        value_gradient: TensorId,
        sink_token_gradient: Option<TensorId>,
        absolute_max_query_gradient: TensorId,
        absolute_max_key_gradient: TensorId,
        absolute_max_value_gradient: TensorId,
        absolute_max_probability_gradient: TensorId,
    ) -> Self {
        Self {
            query_gradient,
            key_gradient,
            value_gradient,
            sink_token_gradient,
            absolute_max_query_gradient,
            absolute_max_key_gradient,
            absolute_max_value_gradient,
            absolute_max_probability_gradient,
        }
    }

    pub fn query_gradient(self) -> TensorId {
        self.query_gradient
    }

    pub fn key_gradient(self) -> TensorId {
        self.key_gradient
    }

    pub fn value_gradient(self) -> TensorId {
        self.value_gradient
    }

    pub fn sink_token_gradient(self) -> Option<TensorId> {
        self.sink_token_gradient
    }

    pub fn absolute_max_query_gradient(self) -> TensorId {
        self.absolute_max_query_gradient
    }

    pub fn absolute_max_key_gradient(self) -> TensorId {
        self.absolute_max_key_gradient
    }

    pub fn absolute_max_value_gradient(self) -> TensorId {
        self.absolute_max_value_gradient
    }

    pub fn absolute_max_probability_gradient(self) -> TensorId {
        self.absolute_max_probability_gradient
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SdpaMxfp8BackwardOutputs {
    query_gradient: TensorId,
    key_gradient: TensorId,
    value_gradient: TensorId,
    absolute_max_query_gradient: TensorId,
    absolute_max_key_gradient: TensorId,
    absolute_max_value_gradient: TensorId,
}

impl SdpaMxfp8BackwardOutputs {
    pub fn new(
        query_gradient: TensorId,
        key_gradient: TensorId,
        value_gradient: TensorId,
        absolute_max_query_gradient: TensorId,
        absolute_max_key_gradient: TensorId,
        absolute_max_value_gradient: TensorId,
    ) -> Self {
        Self {
            query_gradient,
            key_gradient,
            value_gradient,
            absolute_max_query_gradient,
            absolute_max_key_gradient,
            absolute_max_value_gradient,
        }
    }

    pub fn query_gradient(self) -> TensorId {
        self.query_gradient
    }

    pub fn key_gradient(self) -> TensorId {
        self.key_gradient
    }

    pub fn value_gradient(self) -> TensorId {
        self.value_gradient
    }

    pub fn absolute_max_query_gradient(self) -> TensorId {
        self.absolute_max_query_gradient
    }

    pub fn absolute_max_key_gradient(self) -> TensorId {
        self.absolute_max_key_gradient
    }

    pub fn absolute_max_value_gradient(self) -> TensorId {
        self.absolute_max_value_gradient
    }
}

pub(crate) struct UnifiedSdpaSubgraph {
    pub(crate) graph: Graph,
    pub(crate) input: TensorId,
    pub(crate) output: TensorId,
    pub(crate) cloned_tensors: BTreeMap<TensorId, TensorId>,
}
