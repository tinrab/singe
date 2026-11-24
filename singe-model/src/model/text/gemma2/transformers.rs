//! Reads a `transformers` Gemma 2 config and imports its checkpoint.

use serde::{Deserialize, de::IgnoredAny};
use singe::nn::Activation;

use crate::{
    assembly::{DecoderConfig, llama_rules},
    error::{Error, Result},
    format::{
        Rules,
        transformers::{DecoderDefaults, DecoderFields, LayerType, activation_field},
    },
    model::text::gemma2::Gemma2Config,
    source::Source,
};

const fn default_activation() -> Activation {
    Activation::Gelu
}

const fn default_query_scalar() -> f64 {
    256.0
}

#[expect(
    clippy::unnecessary_wraps,
    reason = "an explicit null in the config turns the window off"
)]
const fn default_window() -> Option<i64> {
    Some(4096)
}

#[expect(
    clippy::unnecessary_wraps,
    reason = "an explicit null in the config turns the cap off"
)]
const fn default_attention_softcap() -> Option<f32> {
    Some(50.0)
}

#[expect(
    clippy::unnecessary_wraps,
    reason = "an explicit null in the config turns the cap off"
)]
const fn default_logit_softcap() -> Option<f32> {
    Some(30.0)
}

/// The config's fields; those it leaves out take the defaults of `transformers`' `Gemma2Config`, and an explicit `null` turns the window or a cap off.
#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    #[serde(flatten)]
    decoder: DecoderFields,
    /// The activation `transformers` reads, where it ignores the `hidden_act` Gemma 2 configs also carry.
    #[serde(default = "default_activation", deserialize_with = "activation_field")]
    hidden_activation: Activation,
    /// The ignored `hidden_act`, taken here so the decoder's fields do not read it.
    #[serde(default, rename = "hidden_act")]
    ignored_activation: Option<IgnoredAny>,
    #[serde(default = "default_query_scalar")]
    query_pre_attn_scalar: f64,
    #[serde(default = "default_window")]
    sliding_window: Option<i64>,
    #[serde(default = "default_attention_softcap")]
    attn_logit_softcapping: Option<f32>,
    #[serde(default = "default_logit_softcap")]
    final_logit_softcapping: Option<f32>,
    #[serde(default)]
    use_bidirectional_attention: Option<bool>,
}

pub(crate) fn config(source: &Source) -> Result<Gemma2Config> {
    let fields: Fields = source.json("config.json")?;
    // A checkpoint using this would load and then compute something else.
    if fields.use_bidirectional_attention == Some(true) {
        return Err(Error::unsupported("bidirectional attention"));
    }

    let decoder = DecoderConfig {
        activation: fields.hidden_activation,
        ..fields.decoder.config(DecoderDefaults {
            tied: true,
            head_dim: Some(256),
        })?
    };
    // Without layer types, `transformers` slides every other layer, starting with the first.
    let windows = fields
        .decoder
        .layer_types(|index| {
            if index.is_multiple_of(2) {
                LayerType::SlidingAttention
            } else {
                LayerType::FullAttention
            }
        })?
        .into_iter()
        .map(|layer| match layer {
            LayerType::SlidingAttention => Ok(fields.sliding_window),
            LayerType::FullAttention => Ok(None),
            // A checkpoint using one of these would load and then compute something else.
            LayerType::LinearAttention => Err(Error::unsupported("a linear attention layer")),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Gemma2Config {
        decoder,
        attention_scale: fields.query_pre_attn_scalar.sqrt().recip() as f32,
        windows,
        attention_softcap: fields.attn_logit_softcapping,
        logit_softcap: fields.final_logit_softcapping,
    })
}

/// Returns the rules that import a checkpoint.
/// Gemma 2's `post_attention_layernorm` norms what attention returns, where Llama's norms the feed-forward layer's input, so these rules go before Llama's.
pub(crate) fn rules() -> Rules {
    Rules::new()
        .rename_each(
            "model.layers.*",
            "blocks.*",
            &[
                ("post_attention_layernorm", "attention_output_norm"),
                ("pre_feedforward_layernorm", "feed_forward_norm"),
                ("post_feedforward_layernorm", "feed_forward_output_norm"),
            ],
        )
        .then(llama_rules("model"))
}
