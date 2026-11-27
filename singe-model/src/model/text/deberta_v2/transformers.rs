//! Reads a `transformers` DeBERTa-v2 config and imports its checkpoint, whose task heads store the encoder under `deberta`.

use serde::Deserialize;
use singe::{
    Result as SingeResult,
    nn::{Activation, PoolingMode},
};

use crate::{
    error::{Error, Result},
    format::{RuleSet, transformers::activation_field},
    head::{ClassifierLayout, EncoderBody, EncoderHeadLayout, EncoderModel, Head},
    model::text::deberta_v2::{DebertaV2, DebertaV2Config, body},
    schema::Schema,
    source::Source,
};

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
    max_position_embeddings: i64,
    #[serde(default)]
    max_relative_positions: i64,
    #[serde(default)]
    position_buckets: i64,
    #[serde(default)]
    relative_attention: bool,
    #[serde(default)]
    position_biased_input: bool,
    #[serde(default)]
    share_att_key: bool,
    #[serde(default)]
    pos_att_type: Vec<String>,
    #[serde(default)]
    norm_rel_ebd: String,
    #[serde(default)]
    type_vocab_size: i64,
    embedding_size: Option<i64>,
    conv_kernel_size: Option<i64>,
}

impl EncoderModel for DebertaV2 {
    type Config = DebertaV2Config;

    /// A classifier reads the context pooler, a GELU projection of the first token's state.
    const LAYOUTS: EncoderHeadLayout = EncoderHeadLayout {
        classifier: ClassifierLayout {
            pooling: PoolingMode::Cls,
            projection: Some(Activation::GeluExact),
            output_bias: true,
        },
        masked_lm_activation: None,
    };

    fn config(source: &Source) -> Result<DebertaV2Config> {
        let fields: Fields = source.json("config.json")?;
        let mut types = fields.pos_att_type;
        types.sort();
        let is_v3 = fields.relative_attention
            && !fields.position_biased_input
            && fields.share_att_key
            && types == ["c2p", "p2c"]
            && fields.norm_rel_ebd.contains("layer_norm")
            && fields.position_buckets > 0
            && fields.type_vocab_size == 0
            && fields
                .embedding_size
                .is_none_or(|size| size == fields.hidden_size)
            && fields.conv_kernel_size.is_none();
        if !is_v3 {
            return Err(Error::unsupported(
                "a DeBERTa-v2 other than v3's: relative attention through shared keys both ways over normed, bucketed positions, without absolute positions, token types, or convolution",
            ));
        }

        let max_distance = if fields.max_relative_positions < 1 {
            fields.max_position_embeddings
        } else {
            fields.max_relative_positions
        };
        Ok(DebertaV2Config {
            vocab_size: fields.vocab_size,
            width: fields.hidden_size,
            layers: fields.num_hidden_layers,
            heads: fields.num_attention_heads,
            inner: fields.intermediate_size,
            activation: fields.hidden_act,
            eps: fields.layer_norm_eps,
            buckets: fields.position_buckets,
            max_distance,
        })
    }

    fn create(schema: &mut Schema<'_>, config: &DebertaV2Config) -> SingeResult<EncoderBody> {
        body(schema, config)
    }

    fn encoder_rules() -> RuleSet {
        RuleSet::new()
            // The positions an input leaves out, which v3 never reads.
            .skip("deberta.embeddings.position_ids")
            .rename("deberta.embeddings.word_embeddings", "embeddings.tokens")
            .rename("deberta.embeddings.LayerNorm", "embeddings.norm")
            .rename("deberta.encoder.rel_embeddings", "relative")
            .rename("deberta.encoder.LayerNorm", "relative_norm")
            .rename_each(
                "deberta.encoder.layer.*",
                "blocks.*",
                &[
                    ("attention.self.query_proj", "attention.query"),
                    ("attention.self.key_proj", "attention.key"),
                    ("attention.self.value_proj", "attention.value"),
                    ("attention.output.dense", "attention.output"),
                    ("attention.output.LayerNorm", "attention_norm"),
                    ("intermediate.dense", "feed_forward.up"),
                    ("output.dense", "feed_forward.down"),
                    ("output.LayerNorm", "feed_forward_norm"),
                ],
            )
    }

    fn head_rules(head: Head) -> RuleSet {
        match head {
            Head::SequenceClassification => RuleSet::new()
                .rename("pooler.dense", "classifier.projection")
                .rename("classifier", "classifier.output"),
            _ => RuleSet::new(),
        }
    }
}
