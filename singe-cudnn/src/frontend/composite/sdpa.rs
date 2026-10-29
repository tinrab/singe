mod alibi;
mod backward;
mod backward_policy;
mod backward_request;
mod backward_softmax;
mod forward;
mod fp8_backward;
mod mxfp8_backward;
mod outputs;
mod scores;
mod support;
mod visit;

use std::{collections::BTreeMap, result::Result as StdResult};

use serde::{Deserialize, Serialize};

pub use crate::frontend::composite::sdpa::alibi::alibi_slopes;
pub(crate) use crate::frontend::composite::sdpa::scores::{
    attention_heads_compatible, infer_sdpa_output_shape, infer_sdpa_scores_shape,
};
use crate::{
    backend::tensor::{TensorId, TensorSpec},
    error::Result,
    frontend::{
        error::{Error as FrontendError, SdpaFeature},
        graph::{Graph, TensorOutputConfig},
        operation::{
            SdpaAuxOutputRequest, SdpaQuantizedAuxOutputRequest, SdpaSoftmaxAuxOutputRequest,
        },
    },
};

/// Attention operands with positive query, key and value head counts.
/// Query heads must be divisible by key heads and value heads independently.
/// Key and value head counts need not equal each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SdpaInputs {
    /// Logical `[batch, query_heads, query_sequence, query_hidden]` dimensions.
    pub query: TensorId,
    /// Dense K uses `[batch, key_heads, key_sequence, query_hidden]`, with storage order given by strides.
    /// Paged K follows the configured cache layout and is reconstructed in this logical order.
    pub key: TensorId,
    /// Dense V uses `[batch, value_heads, key_sequence, value_hidden]` dimensions.
    pub value: TensorId,
    /// Scalar tensor multiplying attention logits before softmax.
    pub scale: TensorId,
}

/// Floating forward results or independent inference policies for the result and softmax summaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SdpaOutputTensors<T = TensorId> {
    /// Attention result tensor or its output inference policy.
    pub output: T,
    /// Optional saved softmax log-sum-exp tensor or its inference policy.
    pub stats: Option<T>,
    /// Optional per-row maximum of the attention logits.
    pub logit_max: Option<T>,
    /// Optional per-row sum of exponentials after subtracting the logit maximum.
    pub score_sum_exp: Option<T>,
}

impl<T> SdpaOutputTensors<T> {
    /// Creates a floating attention output request with all softmax summaries disabled.
    pub fn new(output: T) -> Self {
        Self {
            output,
            stats: None,
            logit_max: None,
            score_sum_exp: None,
        }
    }

    /// Sets saved Stats or clears it with `None`.
    #[must_use]
    pub fn with_stats(mut self, stats: impl Into<Option<T>>) -> Self {
        self.stats = stats.into();
        self
    }

    /// Sets the logit maximum or clears it with `None`.
    #[must_use]
    pub fn with_logit_max(mut self, logit_max: impl Into<Option<T>>) -> Self {
        self.logit_max = logit_max.into();
        self
    }

    /// Sets the exponential sum or clears it with `None`.
    #[must_use]
    pub fn with_score_sum_exp(mut self, score_sum_exp: impl Into<Option<T>>) -> Self {
        self.score_sum_exp = score_sum_exp.into();
        self
    }
}

/// Floating backward operands use logical `[batch, heads, sequence, hidden]` Q/K/V shapes.
/// Preparation creates the matmul transposes and resolves architecture-dependent planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SdpaBackwardInputs {
    /// Query tensor in logical batch, head, sequence, hidden order.
    pub query: TensorId,
    /// Key tensor in logical batch, head, sequence, hidden order.
    pub key: TensorId,
    /// Value tensor in logical batch, head, sequence, hidden order.
    pub value: TensorId,
    /// Saved forward attention output.
    pub output: TensorId,
    /// Gradient with respect to the forward attention output.
    pub output_gradient: TensorId,
    /// Saved softmax log-sum-exp statistics from the forward operation.
    pub stats: TensorId,
    /// Scalar tensor multiplying attention logits before softmax.
    pub scale: TensorId,
}

/// Independent placement and encoding of inferred query, key and value gradients.
/// Auxiliary outputs select placement independently and retain operation-specific default encodings.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SdpaGradientOutputConfig {
    /// Placement and encoding request for the query gradient.
    pub query_gradient: TensorOutputConfig,
    /// Placement and encoding request for the key gradient.
    pub key_gradient: TensorOutputConfig,
    /// Placement and encoding request for the value gradient.
    pub value_gradient: TensorOutputConfig,
    /// Optional floating-attention bias gradient, defaulting to FP32 encoding.
    pub bias_gradient: Option<TensorOutputConfig>,
    /// Optional floating-attention RNG dump, retaining the native FP32 encoding contract.
    pub rng_dump: Option<TensorOutputConfig>,
    /// Optional sink gradient for floating, FP8 or MXFP8 attention.
    pub sink_token_gradient: Option<TensorOutputConfig>,
}

impl SdpaGradientOutputConfig {
    /// Applies one output configuration to all three gradients.
    /// Public fields allow each gradient to select a different configuration.
    /// Auxiliary outputs remain disabled until requested.
    #[must_use]
    pub fn new(output: TensorOutputConfig) -> Self {
        Self {
            query_gradient: output,
            key_gradient: output,
            value_gradient: output,
            ..Self::default()
        }
    }

    /// Requests a floating-attention bias gradient or clears it with `None`.
    #[must_use]
    pub fn with_bias_gradient(mut self, output: impl Into<Option<TensorOutputConfig>>) -> Self {
        self.bias_gradient = output.into();
        self
    }

    /// Requests a floating-attention RNG dump or clears it with `None`.
    #[must_use]
    pub fn with_rng_dump(mut self, output: impl Into<Option<TensorOutputConfig>>) -> Self {
        self.rng_dump = output.into();
        self
    }

    /// Requests a sink gradient or clears it with `None`.
    #[must_use]
    pub fn with_sink_token_gradient(
        mut self,
        output: impl Into<Option<TensorOutputConfig>>,
    ) -> Self {
        self.sink_token_gradient = output.into();
        self
    }

    pub(crate) fn validate_quantized(self, operation: &'static str) -> Result<()> {
        if self.bias_gradient.is_some() {
            return Err(FrontendError::SdpaUnsupported {
                feature: SdpaFeature::BiasGradient,
                operation: "quantized sdpa backward",
                sm_version: None,
            }
            .into());
        }
        if self.rng_dump.is_some() {
            return Err(FrontendError::SdpaUnsupported {
                operation,
                feature: SdpaFeature::RngDump,
                sm_version: None,
            }
            .into());
        }
        Ok(())
    }
}

/// Per-tensor FP8 gradient requests and all four required absolute-maximum outputs.
/// Amax outputs default to bindable FP32 tensors independently of graph precision policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SdpaFp8BackwardOutputConfig {
    /// Independent primary-gradient and auxiliary-output inference policies.
    pub gradients: SdpaGradientOutputConfig,
    /// Placement and encoding request for the FP32 query-gradient absolute maximum.
    pub absolute_max_query_gradient: TensorOutputConfig,
    /// Placement and encoding request for the FP32 key-gradient absolute maximum.
    pub absolute_max_key_gradient: TensorOutputConfig,
    /// Placement and encoding request for the FP32 value-gradient absolute maximum.
    pub absolute_max_value_gradient: TensorOutputConfig,
    /// Placement and encoding request for the FP32 intermediate-gradient absolute maximum.
    pub absolute_max_probability_gradient: TensorOutputConfig,
}

impl Default for SdpaFp8BackwardOutputConfig {
    fn default() -> Self {
        Self::new(SdpaGradientOutputConfig::default())
    }
}

impl SdpaFp8BackwardOutputConfig {
    /// Selects primary/sink-gradient policies and keeps bindable FP32 Amax defaults.
    #[must_use]
    pub fn new(gradients: SdpaGradientOutputConfig) -> Self {
        let amax = TensorOutputConfig::new().output_tensor();
        Self {
            gradients,
            absolute_max_query_gradient: amax,
            absolute_max_key_gradient: amax,
            absolute_max_value_gradient: amax,
            absolute_max_probability_gradient: amax,
        }
    }

    /// Applies one policy to all four required maxima, including the probability gradient.
    /// Public fields allow independent placement and supported encoding choices.
    #[must_use]
    pub fn with_absolute_max_outputs(mut self, output: TensorOutputConfig) -> Self {
        self.absolute_max_query_gradient = output;
        self.absolute_max_key_gradient = output;
        self.absolute_max_value_gradient = output;
        self.absolute_max_probability_gradient = output;
        self
    }
}

/// MXFP8 gradient requests with independently optional absolute-maximum outputs.
/// Amax outputs default to FP32 encoding and are omitted unless requested.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SdpaMxfp8BackwardOutputConfig {
    /// Independent primary-gradient and auxiliary-output inference policies.
    pub gradients: SdpaGradientOutputConfig,
    /// Optional placement and encoding request for the FP32 query-gradient absolute maximum.
    pub absolute_max_query_gradient: Option<TensorOutputConfig>,
    /// Optional placement and encoding request for the FP32 key-gradient absolute maximum.
    pub absolute_max_key_gradient: Option<TensorOutputConfig>,
    /// Optional placement and encoding request for the FP32 value-gradient absolute maximum.
    pub absolute_max_value_gradient: Option<TensorOutputConfig>,
}

impl SdpaMxfp8BackwardOutputConfig {
    /// Selects primary/sink-gradient policies without requesting Amax outputs.
    #[must_use]
    pub fn new(gradients: SdpaGradientOutputConfig) -> Self {
        Self {
            gradients,
            ..Self::default()
        }
    }

    /// Requests all three gradient maxima with one policy, or clears them with `None`.
    /// Public fields allow independent requests and placements.
    #[must_use]
    pub fn with_absolute_max_gradients(
        mut self,
        output: impl Into<Option<TensorOutputConfig>>,
    ) -> Self {
        let output = output.into();
        self.absolute_max_query_gradient = output;
        self.absolute_max_key_gradient = output;
        self.absolute_max_value_gradient = output;
        self
    }
}

/// Caller-declared floating gradients and independently optional auxiliary outputs.
/// Also returned by inference, preserving every requested output identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SdpaBackwardGradientTensors {
    /// Gradient with respect to the query tensor.
    pub query_gradient: TensorId,
    /// Gradient with respect to the key tensor.
    pub key_gradient: TensorId,
    /// Gradient with respect to the value tensor.
    pub value_gradient: TensorId,
    /// Optional gradient with respect to the attention bias.
    pub bias_gradient: Option<TensorId>,
    /// Optional dump of the random numbers generated for dropout.
    pub rng_dump: Option<TensorId>,
    /// Optional gradient with respect to the attention sink tensor.
    pub sink_token_gradient: Option<TensorId>,
}

impl SdpaBackwardGradientTensors {
    /// Records query, key, and value gradient identities without auxiliary outputs.
    #[must_use]
    pub fn new(query_gradient: TensorId, key_gradient: TensorId, value_gradient: TensorId) -> Self {
        Self {
            query_gradient,
            key_gradient,
            value_gradient,
            bias_gradient: None,
            rng_dump: None,
            sink_token_gradient: None,
        }
    }

    /// Sets the bias gradient or clears it with `None`.
    #[must_use]
    pub fn with_bias_gradient(mut self, bias_gradient: impl Into<Option<TensorId>>) -> Self {
        self.bias_gradient = bias_gradient.into();
        self
    }

    /// Sets the RNG dump or clears it with `None`.
    #[must_use]
    pub fn with_rng_dump(mut self, rng_dump: impl Into<Option<TensorId>>) -> Self {
        self.rng_dump = rng_dump.into();
        self
    }

    /// Sets the sink gradient or clears it with `None`.
    #[must_use]
    pub fn with_sink_token_gradient(
        mut self,
        sink_token_gradient: impl Into<Option<TensorId>>,
    ) -> Self {
        self.sink_token_gradient = sink_token_gradient.into();
        self
    }
}

/// FP8 attention operands and explicit quantization scales.
/// Composite scales broadcast against the score or output tensor at their point of use.
/// Native engines impose their own supported scale layouts.
/// Attention scaling is a separate scalar tensor, including runtime device or by-value inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SdpaFp8Inputs {
    /// Query tensor in logical batch, head, sequence, hidden order.
    pub query: TensorId,
    /// Dense K retains logical sequence/hidden axes, as in [`SdpaInputs`].
    /// The composite implementation transposes K for its score matmul.
    pub key: TensorId,
    /// Value tensor in logical batch, head, sequence, hidden order.
    pub value: TensorId,
    /// Scalar attention-logit multiplier, separate from quantization scales.
    pub attention_scale: TensorId,
    /// Dequantization multiplier for the query tensor.
    pub descale_query: TensorId,
    /// Dequantization multiplier for the key tensor.
    pub descale_key: TensorId,
    /// Dequantization multiplier for the value tensor.
    pub descale_value: TensorId,
    /// Dequantization multiplier for the quantized softmax-probability tensor S.
    pub descale_scores: TensorId,
    /// Quantization multiplier for the softmax-probability tensor S.
    pub scale_scores: TensorId,
    /// Quantization multiplier for the attention output.
    pub scale_output: TensorId,
}

impl SdpaFp8Inputs {
    /// Records FP8 operands, attention scaling, and explicit quantization and dequantization tensors.
    #[must_use]
    pub fn new(
        query: TensorId,
        key: TensorId,
        value: TensorId,
        attention_scale: TensorId,
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
            attention_scale,
            descale_query,
            descale_key,
            descale_value,
            descale_scores,
            scale_scores,
            scale_output,
        }
    }
}

/// Forward results or independent inference policies for floating, FP8 and MXFP8 attention.
/// Optional fields select outputs; `TensorOutputConfig` values select their inferred placement and encoding.
/// Tensor identities preserve caller-declared layouts and placement during explicit construction.
/// Max and Sum-exp retain their independently declared encodings, with native planning determining support.
/// Saved Stats and Amax outputs require FP32 storage.
/// MXFP8 does not support the softmax probability absolute-maximum output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SdpaForwardOutputTensors<T = TensorId> {
    /// Attention result tensor or its output inference policy.
    pub output: T,
    /// Optional saved softmax log-sum-exp tensor or its inference policy.
    pub stats: Option<T>,
    /// Optional per-row maximum of the attention logits.
    pub logit_max: Option<T>,
    /// Optional per-row sum of exponentials after subtracting the logit maximum.
    pub score_sum_exp: Option<T>,
    /// Optional dump of the random numbers generated for dropout.
    pub rng_dump: Option<T>,
    /// Optional maximum of softmax probabilities S, corresponding to the FP8 `Amax_S` output.
    pub absolute_max_probabilities: Option<T>,
    /// Optional absolute maximum of the attention output.
    pub absolute_max_output: Option<T>,
}

impl<T> SdpaForwardOutputTensors<T> {
    /// Creates a forward attention output request with all auxiliary outputs disabled.
    pub fn new(output: T) -> Self {
        Self {
            output,
            stats: None,
            logit_max: None,
            score_sum_exp: None,
            rng_dump: None,
            absolute_max_probabilities: None,
            absolute_max_output: None,
        }
    }

    /// Sets saved Stats or clears it with `None`.
    #[must_use]
    pub fn with_stats(mut self, stats: impl Into<Option<T>>) -> Self {
        self.stats = stats.into();
        self
    }

    /// Sets the logit maximum or clears it with `None`.
    #[must_use]
    pub fn with_logit_max(mut self, logit_max: impl Into<Option<T>>) -> Self {
        self.logit_max = logit_max.into();
        self
    }

    /// Sets the exponential sum or clears it with `None`.
    #[must_use]
    pub fn with_score_sum_exp(mut self, score_sum_exp: impl Into<Option<T>>) -> Self {
        self.score_sum_exp = score_sum_exp.into();
        self
    }

    /// Sets the RNG dump or clears it with `None`.
    #[must_use]
    pub fn with_rng_dump(mut self, rng_dump: impl Into<Option<T>>) -> Self {
        self.rng_dump = rng_dump.into();
        self
    }

    /// Sets the softmax-probability Amax or clears it with `None`.
    #[must_use]
    pub fn with_absolute_max_probabilities(mut self, output: impl Into<Option<T>>) -> Self {
        self.absolute_max_probabilities = output.into();
        self
    }

    /// Sets the output Amax or clears it with `None`.
    #[must_use]
    pub fn with_absolute_max_output(mut self, output: impl Into<Option<T>>) -> Self {
        self.absolute_max_output = output.into();
        self
    }

    /// Maps every declared output while preserving absent auxiliary outputs.
    pub fn map<U>(self, mut map_tensor: impl FnMut(T) -> U) -> SdpaForwardOutputTensors<U> {
        SdpaForwardOutputTensors {
            output: map_tensor(self.output),
            stats: self.stats.map(&mut map_tensor),
            logit_max: self.logit_max.map(&mut map_tensor),
            score_sum_exp: self.score_sum_exp.map(&mut map_tensor),
            rng_dump: self.rng_dump.map(&mut map_tensor),
            absolute_max_probabilities: self.absolute_max_probabilities.map(&mut map_tensor),
            absolute_max_output: self.absolute_max_output.map(&mut map_tensor),
        }
    }

    /// Maps every declared output, stopping at the first error.
    ///
    /// # Errors
    ///
    /// Returns the first error from the supplied output mapping; absent auxiliary outputs are not mapped.
    pub fn try_map<U, E>(
        self,
        mut map_tensor: impl FnMut(T) -> StdResult<U, E>,
    ) -> StdResult<SdpaForwardOutputTensors<U>, E> {
        Ok(SdpaForwardOutputTensors {
            output: map_tensor(self.output)?,
            stats: self.stats.map(&mut map_tensor).transpose()?,
            logit_max: self.logit_max.map(&mut map_tensor).transpose()?,
            score_sum_exp: self.score_sum_exp.map(&mut map_tensor).transpose()?,
            rng_dump: self.rng_dump.map(&mut map_tensor).transpose()?,
            absolute_max_probabilities: self
                .absolute_max_probabilities
                .map(&mut map_tensor)
                .transpose()?,
            absolute_max_output: self.absolute_max_output.map(&mut map_tensor).transpose()?,
        })
    }

    pub(crate) fn aux_outputs(&self) -> SdpaAuxOutputRequest {
        SdpaAuxOutputRequest::new(
            SdpaSoftmaxAuxOutputRequest {
                stats: self.stats.is_some(),
                logit_max: self.logit_max.is_some(),
                score_sum_exp: self.score_sum_exp.is_some(),
            },
            SdpaQuantizedAuxOutputRequest {
                absolute_max_probabilities: self.absolute_max_probabilities.is_some(),
                absolute_max_output: self.absolute_max_output.is_some(),
            },
            self.rng_dump.is_some(),
        )
    }
}

/// MXFP8 attention operands with separate E8M0 block-scale tensors and scalar attention scaling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SdpaMxfp8Inputs {
    /// Query tensor in logical batch, head, sequence, hidden order.
    pub query: TensorId,
    /// Logical `[batch, key_heads, key_sequence, query_hidden]` dimensions.
    /// K and its block scales are transposed internally for the composite score matmul.
    pub key: TensorId,
    /// Value tensor in logical batch, head, sequence, hidden order.
    pub value: TensorId,
    /// E8M0 block scales along the query hidden dimension.
    /// Unified execution requires `F8_128x4` reordering and a unit last stride.
    pub scale_query: TensorId,
    /// E8M0 block scales along the key hidden dimension, with the same native layout requirements as query scales.
    pub scale_key: TensorId,
    /// E8M0 block scales along the value sequence dimension.
    /// Native finalization determines supported strides independently of the Q/K scale layout.
    pub scale_value: TensorId,
    /// Scalar attention-logit multiplier, separate from quantization scales.
    pub attention_scale: TensorId,
}

impl SdpaMxfp8Inputs {
    /// Records MXFP8 operands, their block scales, and a separate scalar attention multiplier.
    #[must_use]
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

/// FP8 attention backward operands with explicit scale and descale tensors.
/// Probability and score-gradient scales and descales participate in the gradient matmuls.
/// Quantization scales broadcast at their point of use in the composite graph.
/// The attention-scale operand remains scalar.
/// Its encoding reaches native support selection, with `F64` preserved when combining a numeric factor.
/// Fixed scalar scales with `F16` or `BF16` gradients express C++ current scaling.
/// Input and gradient encodings reach native support selection, matching the C++ per-tensor composition.
/// `F16` and `BF16` gradients retain the C++ Blackwell-or-newer requirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SdpaFp8BackwardInputs {
    /// Query tensor in logical batch, head, sequence, hidden order.
    pub query: TensorId,
    /// Key tensor in logical batch, head, sequence, hidden order.
    pub key: TensorId,
    /// Value tensor in logical batch, head, sequence, hidden order.
    pub value: TensorId,
    /// Saved forward attention output.
    pub output: TensorId,
    /// Gradient with respect to the forward attention output.
    pub output_gradient: TensorId,
    /// Saved softmax log-sum-exp statistics from the forward operation.
    pub stats: TensorId,
    /// Scalar attention-logit multiplier, separate from quantization scales.
    pub attention_scale: TensorId,
    /// Dequantization multiplier for the query tensor.
    pub descale_query: TensorId,
    /// Dequantization multiplier for the key tensor.
    pub descale_key: TensorId,
    /// Dequantization multiplier for the value tensor.
    pub descale_value: TensorId,
    /// Dequantization multiplier for the saved forward output.
    pub descale_output: TensorId,
    /// Dequantization multiplier for the output gradient.
    pub descale_output_gradient: TensorId,
    /// Dequantization multiplier for the quantized softmax-probability tensor S.
    pub descale_scores: TensorId,
    /// Dequantization multiplier for the intermediate gradient used in the query/key gradient matmuls.
    pub descale_probability_gradient: TensorId,
    /// Quantization multiplier for the softmax-probability tensor S.
    pub scale_scores: TensorId,
    /// Quantization multiplier for the query-gradient output.
    pub scale_query_gradient: TensorId,
    /// Quantization multiplier for the key-gradient output.
    pub scale_key_gradient: TensorId,
    /// Quantization multiplier for the value-gradient output.
    pub scale_value_gradient: TensorId,
    /// Quantization multiplier for the intermediate gradient used in the query/key gradient matmuls.
    pub scale_probability_gradient: TensorId,
}

impl SdpaFp8BackwardInputs {
    /// Records the saved forward operands, output gradient, statistics, and per-tensor FP8 scale tensors.
    #[must_use]
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

/// Explicit or inferred per-tensor FP8 gradients and their four required maxima.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SdpaFp8BackwardGradientTensors {
    /// Gradient with respect to the query tensor.
    pub query_gradient: TensorId,
    /// Gradient with respect to the key tensor.
    pub key_gradient: TensorId,
    /// Gradient with respect to the value tensor.
    pub value_gradient: TensorId,
    /// Optional gradient with respect to the attention sink tensor.
    pub sink_token_gradient: Option<TensorId>,
    /// Absolute maximum of the query gradient.
    pub absolute_max_query_gradient: TensorId,
    /// Absolute maximum of the key gradient.
    pub absolute_max_key_gradient: TensorId,
    /// Absolute maximum of the value gradient.
    pub absolute_max_value_gradient: TensorId,
    /// Absolute maximum of the intermediate gradient before quantization for the query/key gradient matmuls.
    pub absolute_max_probability_gradient: TensorId,
}

impl SdpaFp8BackwardGradientTensors {
    /// Records the three primary gradient identities and four required absolute-maximum outputs, without a sink gradient.
    #[must_use]
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

    /// Sets the sink gradient or clears it with `None`.
    #[must_use]
    pub fn with_sink_token_gradient(
        mut self,
        sink_token_gradient: impl Into<Option<TensorId>>,
    ) -> Self {
        self.sink_token_gradient = sink_token_gradient.into();
        self
    }
}

/// Independently quantized operands for MXFP8 attention backward.
/// All value tensors use logical `[batch, heads, sequence, hidden]` dimensions.
/// Sequence-scaled copies use blocks of 32 sequence positions instead of 32 hidden elements.
/// These copies retain the original logical dimensions and may have independently declared strides.
/// Scale tensors use `F8UE8M0` with `F8_128x4` reordering and may include tile padding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SdpaMxfp8BackwardInputs {
    /// Query tensor in logical batch, head, sequence, hidden order.
    pub query: TensorId,
    /// Query quantized along its sequence dimension, corresponding to C++ `Q_T`.
    pub sequence_scaled_query: TensorId,
    /// Key tensor in logical batch, head, sequence, hidden order.
    pub key: TensorId,
    /// Key quantized along its sequence dimension, corresponding to C++ `K_T`.
    pub sequence_scaled_key: TensorId,
    /// Value tensor in logical batch, head, sequence, hidden order.
    pub value: TensorId,
    /// Saved forward attention output.
    pub output: TensorId,
    /// Unquantized gradient used with the saved output for the softmax dot product.
    pub output_gradient: TensorId,
    /// Output gradient quantized along its hidden dimension.
    pub quantized_output_gradient: TensorId,
    /// Output gradient quantized along its sequence dimension, corresponding to C++ `dO_T`.
    pub sequence_scaled_output_gradient: TensorId,
    /// Saved softmax log-sum-exp statistics from the forward operation.
    pub stats: TensorId,
    /// E8M0 block scales for hidden-dimension blocks of the query tensor.
    pub scale_query: TensorId,
    /// E8M0 block scales for sequence-dimension blocks of `sequence_scaled_query`.
    pub sequence_scale_query: TensorId,
    /// E8M0 block scales for hidden-dimension blocks of the key tensor.
    pub scale_key: TensorId,
    /// E8M0 block scales for sequence-dimension blocks of `sequence_scaled_key`.
    pub sequence_scale_key: TensorId,
    /// E8M0 block scales for sequence-dimension blocks of the value tensor.
    pub scale_value: TensorId,
    /// E8M0 block scales for hidden-dimension blocks of the quantized output gradient.
    pub scale_output_gradient: TensorId,
    /// E8M0 block scales for sequence-dimension blocks of the sequence-scaled output gradient.
    pub sequence_scale_output_gradient: TensorId,
    /// Scalar attention-logit multiplier, separate from quantization scales.
    pub attention_scale: TensorId,
}

/// Caller-declared MXFP8 gradients with independent optional auxiliary outputs.
/// Amax reductions observe the converted gradient outputs, matching C++ MXFP8 backward.
/// Also returned by inference, with Amax identities present only when requested.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SdpaMxfp8BackwardGradientTensors {
    /// Gradient with respect to the query tensor.
    pub query_gradient: TensorId,
    /// Gradient with respect to the key tensor.
    pub key_gradient: TensorId,
    /// Gradient with respect to the value tensor.
    pub value_gradient: TensorId,
    /// Optional gradient with respect to the attention sink tensor.
    pub sink_token_gradient: Option<TensorId>,
    /// Optional absolute maximum of the query gradient.
    pub absolute_max_query_gradient: Option<TensorId>,
    /// Optional absolute maximum of the key gradient.
    pub absolute_max_key_gradient: Option<TensorId>,
    /// Optional absolute maximum of the value gradient.
    pub absolute_max_value_gradient: Option<TensorId>,
}

impl SdpaMxfp8BackwardGradientTensors {
    /// Records the three primary gradient identities with sink and absolute-maximum outputs disabled.
    #[must_use]
    pub fn new(query_gradient: TensorId, key_gradient: TensorId, value_gradient: TensorId) -> Self {
        Self {
            query_gradient,
            key_gradient,
            value_gradient,
            sink_token_gradient: None,
            absolute_max_query_gradient: None,
            absolute_max_key_gradient: None,
            absolute_max_value_gradient: None,
        }
    }

    /// Sets the sink gradient or clears it with `None`.
    #[must_use]
    pub fn with_sink_token_gradient(mut self, tensor: impl Into<Option<TensorId>>) -> Self {
        self.sink_token_gradient = tensor.into();
        self
    }

    /// Sets the query-gradient Amax or clears it with `None`.
    #[must_use]
    pub fn with_absolute_max_query_gradient(mut self, tensor: impl Into<Option<TensorId>>) -> Self {
        self.absolute_max_query_gradient = tensor.into();
        self
    }

    /// Sets the key-gradient Amax or clears it with `None`.
    #[must_use]
    pub fn with_absolute_max_key_gradient(mut self, tensor: impl Into<Option<TensorId>>) -> Self {
        self.absolute_max_key_gradient = tensor.into();
        self
    }

    /// Sets the value-gradient Amax or clears it with `None`.
    #[must_use]
    pub fn with_absolute_max_value_gradient(mut self, tensor: impl Into<Option<TensorId>>) -> Self {
        self.absolute_max_value_gradient = tensor.into();
        self
    }
}

pub(crate) struct UnifiedSdpaSubgraph {
    pub(crate) graph: Graph,
    pub(crate) input: TensorId,
    pub(crate) output: TensorId,
    pub(crate) tensor_captures: BTreeMap<TensorId, TensorId>,
    pub(crate) captured_tensors: BTreeMap<TensorId, TensorSpec>,
}
