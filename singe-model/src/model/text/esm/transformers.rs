//! Reads a `transformers` ESM config and imports its checkpoint, whose task heads store the encoder under `esm`.

use serde::Deserialize;
use singe::{
    Result as SingeResult,
    nn::{Activation, PoolingMode},
};

use crate::{
    error::{Error, Result},
    format::{Rules, transformers::activation_field},
    head::{ClassifierLayout, EncoderBody, EncoderModel, Head, HeadLayouts},
    model::text::esm::{Esm, EsmConfig, body},
    schema::Schema,
    source::Source,
};

/// The base of ESM's rotary frequencies, which its `RotaryEmbedding` fixes.
const ROTARY_BASE: f64 = 10_000.0;

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    vocab_size: i64,
    hidden_size: i64,
    num_hidden_layers: i64,
    num_attention_heads: i64,
    intermediate_size: i64,
    #[serde(deserialize_with = "activation_field")]
    hidden_act: Activation,
    layer_norm_eps: f32,
    position_embedding_type: String,
    #[serde(default)]
    emb_layer_norm_before: bool,
    #[serde(default)]
    token_dropout: bool,
    mask_token_id: Option<i64>,
    #[serde(default)]
    is_folding_model: bool,
}

impl EncoderModel for Esm {
    type Config = EsmConfig;

    /// A classifier reads ESM's head: a tanh projection of the first token's state.
    const LAYOUTS: HeadLayouts = HeadLayouts {
        classifier: ClassifierLayout {
            pooling: PoolingMode::Cls,
            projection: Some(Activation::Tanh),
            output_bias: true,
        },
        masked_lm_activation: Some(Activation::GeluExact),
    };

    fn config(source: &Source) -> Result<EsmConfig> {
        let fields: Fields = source.json("config.json")?;
        if fields.position_embedding_type != "rotary"
            || fields.emb_layer_norm_before
            || fields.is_folding_model
        {
            return Err(Error::unsupported(
                "an ESM other than ESM-2's: rotary positions, no norm before the encoder, and no folding trunk",
            ));
        }

        let dropped_token = if fields.token_dropout {
            Some(fields.mask_token_id.ok_or_else(|| {
                Error::unsupported("an ESM with token dropout but no `mask_token_id`")
            })?)
        } else {
            None
        };
        Ok(EsmConfig {
            vocab_size: fields.vocab_size,
            width: fields.hidden_size,
            layers: fields.num_hidden_layers,
            heads: fields.num_attention_heads,
            inner: fields.intermediate_size,
            activation: fields.hidden_act,
            eps: fields.layer_norm_eps,
            rotary_base: ROTARY_BASE,
            dropped_token,
        })
    }

    fn create(schema: &mut Schema<'_>, config: &EsmConfig) -> SingeResult<EncoderBody> {
        body(schema, config)
    }

    fn encoder_rules() -> Rules {
        Rules::new()
            // Rotary models never read the absolute table or its positions; each layer's frequencies are fixed.
            .skip("esm.embeddings.position_embeddings")
            .skip("esm.embeddings.position_ids")
            .skip("esm.encoder.layer.*.attention.self.rotary_embeddings.inv_freq")
            // The contact head reads attention probabilities, which no task here asks for.
            .skip("esm.contact_head")
            .rename("esm.embeddings.word_embeddings", "embeddings.tokens")
            .rename("esm.encoder.emb_layer_norm_after", "norm")
            .rename_each(
                "esm.encoder.layer.*",
                "blocks.*",
                &[
                    ("attention.self.query", "attention.query"),
                    ("attention.self.key", "attention.key"),
                    ("attention.self.value", "attention.value"),
                    ("attention.output.dense", "attention.output"),
                    ("attention.LayerNorm", "attention_norm"),
                    ("LayerNorm", "feed_forward_norm"),
                    ("intermediate.dense", "feed_forward.up"),
                    ("output.dense", "feed_forward.down"),
                ],
            )
    }

    fn head_rules(head: Head) -> Rules {
        match head {
            Head::MaskedLm => Rules::new().rename_each(
                "lm_head",
                "masked_lm",
                &[
                    ("dense", "projection"),
                    ("layer_norm", "norm"),
                    ("bias", "output.bias"),
                ],
            ),
            Head::SequenceClassification => Rules::new()
                .rename("classifier.dense", "classifier.projection")
                .rename("classifier.out_proj", "classifier.output"),
            _ => Rules::new(),
        }
    }
}
