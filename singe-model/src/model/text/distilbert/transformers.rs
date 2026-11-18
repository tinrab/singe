// Reads a `transformers` DistilBERT config and imports its checkpoint, whose task heads store the encoder under `distilbert.`.

use serde::Deserialize;
use singe::nn::Activation;

use crate::{
    assembly::PostNormEncoderConfig,
    error::Result,
    format::{Rules, transformers::activation_field},
    head::Head,
    source::Source,
};

// The epsilon of every DistilBERT LayerNorm, which its config does not hold.
const LAYER_NORM_EPS: f32 = 1e-12;

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    vocab_size: i64,
    dim: i64,
    n_layers: i64,
    n_heads: i64,
    hidden_dim: i64,
    #[serde(deserialize_with = "activation_field")]
    activation: Activation,
    max_position_embeddings: i64,
}

// A snapshot with `sinusoidal_pos_embds` stores its sinusoid table as learned positions, which load as any others.
pub(crate) fn config(source: &Source) -> Result<PostNormEncoderConfig> {
    let fields: Fields = source.json("config.json")?;
    Ok(PostNormEncoderConfig {
        vocab_size: fields.vocab_size,
        width: fields.dim,
        layers: fields.n_layers,
        heads: fields.n_heads,
        inner: fields.hidden_dim,
        activation: fields.activation,
        max_positions: fields.max_position_embeddings,
        first_position: 0,
        token_types: 0,
        eps: LAYER_NORM_EPS,
    })
}

// Imports the encoder stored after `prefix`.
fn prefixed_rules(prefix: &str) -> Rules {
    let embeddings = |name: &str| format!("{prefix}embeddings.{name}");
    let layer = |name: &str| format!("{prefix}transformer.layer.*.{name}");
    Rules::new()
        .rename(&embeddings("word_embeddings"), "embeddings.tokens")
        .rename(&embeddings("position_embeddings"), "embeddings.positions")
        .rename(&embeddings("LayerNorm"), "embeddings.norm")
        .rename(&layer("attention.q_lin"), "blocks.*.attention.query")
        .rename(&layer("attention.k_lin"), "blocks.*.attention.key")
        .rename(&layer("attention.v_lin"), "blocks.*.attention.value")
        .rename(&layer("attention.out_lin"), "blocks.*.attention.output")
        .rename(&layer("sa_layer_norm"), "blocks.*.attention_norm")
        .rename(&layer("ffn.lin1"), "blocks.*.feed_forward.up")
        .rename(&layer("ffn.lin2"), "blocks.*.feed_forward.down")
        .rename(&layer("output_layer_norm"), "blocks.*.feed_forward_norm")
}

pub(crate) fn encoder_rules() -> Rules {
    prefixed_rules("")
        .then(prefixed_rules("distilbert."))
        // The heads' tensors, which only their own head's rules read.
        .skip("pre_classifier")
        .skip("classifier")
        .skip("qa_outputs")
        .skip("vocab_transform")
        .skip("vocab_layer_norm")
        .skip("vocab_projector")
}

pub(crate) fn head_rules(head: Head) -> Rules {
    match head {
        Head::SequenceClassification => Rules::new()
            .rename("pre_classifier", "classifier.projection")
            .rename("classifier", "classifier.output"),
        Head::TokenClassification => Rules::new().rename("classifier", "classifier.output"),
        Head::QuestionAnswering => Rules::new().rename("qa_outputs", "span.output"),
        Head::MaskedLm => Rules::new()
            .rename("vocab_transform", "masked_lm.projection")
            .rename("vocab_layer_norm", "masked_lm.norm")
            .rename("vocab_projector", "masked_lm.output"),
        _ => Rules::new(),
    }
}
