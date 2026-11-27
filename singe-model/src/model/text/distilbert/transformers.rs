//! Reads a `transformers` `DistilBERT` config and imports its checkpoint, whose task heads store the encoder under `distilbert`.

use serde::Deserialize;
use singe::{
    Result as SingeResult,
    nn::{Activation, PoolingMode},
};

use crate::{
    assembly::PostNormEncoderConfig,
    error::Result,
    format::{RuleSet, join, transformers::activation_field},
    head::{ClassifierLayout, EncoderBody, EncoderHeadLayout, EncoderModel, Head, post_norm_body},
    model::text::distilbert::DistilBert,
    schema::Schema,
    source::Source,
};

/// The epsilon of every `DistilBERT` `LayerNorm`, which its config does not hold.
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

/// Returns the rules that import the encoder stored under `prefix`.
fn prefixed_rules(prefix: &str) -> RuleSet {
    RuleSet::new()
        .rename_each(
            &join(prefix, "embeddings"),
            "embeddings",
            &[
                ("word_embeddings", "tokens"),
                ("position_embeddings", "positions"),
                ("LayerNorm", "norm"),
            ],
        )
        .rename_each(
            &join(prefix, "transformer.layer.*"),
            "blocks.*",
            &[
                ("attention.q_lin", "attention.query"),
                ("attention.k_lin", "attention.key"),
                ("attention.v_lin", "attention.value"),
                ("attention.out_lin", "attention.output"),
                ("sa_layer_norm", "attention_norm"),
                ("ffn.lin1", "feed_forward.up"),
                ("ffn.lin2", "feed_forward.down"),
                ("output_layer_norm", "feed_forward_norm"),
            ],
        )
}

impl EncoderModel for DistilBert {
    type Config = PostNormEncoderConfig;

    /// A classifier reads the first token's state through `pre_classifier` and `ReLU`.
    const LAYOUTS: EncoderHeadLayout = EncoderHeadLayout {
        classifier: ClassifierLayout {
            pooling: PoolingMode::Cls,
            projection: Some(Activation::Relu),
            output_bias: true,
        },
        masked_lm_activation: None,
    };

    /// Reads a `DistilBERT` config.
    /// A snapshot with `sinusoidal_pos_embds` stores its sinusoid table as learned positions, which load as any others.
    fn config(source: &Source) -> Result<PostNormEncoderConfig> {
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
            keys: None,
        })
    }

    fn create(schema: &mut Schema<'_>, config: &PostNormEncoderConfig) -> SingeResult<EncoderBody> {
        post_norm_body(schema, config)
    }

    fn encoder_rules() -> RuleSet {
        prefixed_rules("")
            .then(prefixed_rules("distilbert"))
            // The heads' tensors, which only their own head's rules read.
            .skip("pre_classifier")
            .skip("classifier")
            .skip("qa_outputs")
            .skip("vocab_transform")
            .skip("vocab_layer_norm")
            .skip("vocab_projector")
    }

    fn head_rules(head: Head) -> RuleSet {
        match head {
            Head::SequenceClassification => RuleSet::new()
                .rename("pre_classifier", "classifier.projection")
                .rename("classifier", "classifier.output"),
            Head::MaskedLm => RuleSet::new()
                .rename("vocab_transform", "masked_lm.projection")
                .rename("vocab_layer_norm", "masked_lm.norm")
                .rename("vocab_projector", "masked_lm.output"),
            _ => RuleSet::new(),
        }
    }
}
