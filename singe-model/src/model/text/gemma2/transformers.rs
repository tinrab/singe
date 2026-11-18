// Reads a `transformers` Gemma 2 config, and imports its checkpoint.

use serde::{Deserialize, de::IgnoredAny};
use singe::nn::Activation;

use super::Gemma2Config;
use crate::{
    assembly::{DecoderConfig, llama_rules},
    error::{Error, Result},
    format::{
        Rules,
        transformers::{DecoderDefaults, DecoderFields, activation_field},
    },
    source::Source,
};

// The defaults `transformers`' `Gemma2Config` gives the fields a config leaves out; an explicit `null` turns the window or a cap off.
const WINDOW: Option<i64> = Some(4096);

const ATTENTION_SOFTCAP: Option<f32> = Some(50.0);

const LOGIT_SOFTCAP: Option<f32> = Some(30.0);

const fn default_activation() -> Activation {
    Activation::Gelu
}

const fn default_query_scalar() -> f64 {
    256.0
}

const fn default_window() -> Option<i64> {
    WINDOW
}

const fn default_attention_softcap() -> Option<f32> {
    ATTENTION_SOFTCAP
}

const fn default_logit_softcap() -> Option<f32> {
    LOGIT_SOFTCAP
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    #[serde(flatten)]
    decoder: DecoderFields,
    // The activation `transformers` reads, where it ignores the `hidden_act` Gemma 2 configs also carry.
    #[serde(default = "default_activation", deserialize_with = "activation_field")]
    hidden_activation: Activation,
    // Taken here, so the decoder's fields do not read it.
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

#[expect(
    clippy::cast_possible_truncation,
    reason = "the scale is defined in F32, as the references compute it"
)]
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
    let layers = usize::try_from(decoder.layers)
        .map_err(|_| Error::unsupported("a negative number of layers"))?;
    // Without layer types, `transformers` slides every other layer, starting with the first.
    let layer_types = fields.decoder.layer_types.clone().unwrap_or_else(|| {
        (0..layers)
            .map(|index| {
                if index.is_multiple_of(2) {
                    "sliding_attention"
                } else {
                    "full_attention"
                }
            })
            .map(str::to_owned)
            .collect()
    });
    let windows = (0..layers)
        .map(|index| match layer_types.get(index).map(String::as_str) {
            Some("sliding_attention") => Ok(fields.sliding_window),
            Some("full_attention") => Ok(None),
            // A checkpoint using one of these would load and then compute something else.
            _ => Err(Error::unsupported(
                "a layer that is neither full nor sliding",
            )),
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

// Gemma 2's `post_attention_layernorm` norms what attention returns, where Llama's norms the feed-forward layer's input, so these rules go before Llama's.
pub(crate) fn rules() -> Rules {
    let layer = |name: &str| format!("model.layers.*.{name}");
    Rules::new()
        .rename(
            &layer("post_attention_layernorm"),
            "blocks.*.attention_output_norm",
        )
        .rename(
            &layer("pre_feedforward_layernorm"),
            "blocks.*.feed_forward_norm",
        )
        .rename(
            &layer("post_feedforward_layernorm"),
            "blocks.*.feed_forward_output_norm",
        )
        .then(llama_rules("model"))
}
