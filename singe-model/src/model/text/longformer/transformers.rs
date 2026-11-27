//! Reads a `transformers` Longformer config and imports its checkpoint, whose encoder sits under `longformer` and whose token table is stored once, as the tied output of the masked language model.

use serde::Deserialize;
use singe::{
    Result as SingeResult,
    nn::{Activation, PoolingMode},
};

use crate::{
    assembly::{KeyPattern, PostNormEncoderConfig, post_norm_encoder_rules},
    error::{Error, Result},
    format::{Rules, transformers::EncoderFields},
    head::{ClassifierLayout, EncoderBody, EncoderModel, Head, HeadLayouts, post_norm_body},
    model::text::longformer::Longformer,
    schema::Schema,
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    #[serde(flatten)]
    encoder: EncoderFields,
    pad_token_id: i64,
    attention_window: Vec<i64>,
    attention_mode: String,
}

impl EncoderModel for Longformer {
    type Config = PostNormEncoderConfig;

    /// The masked language model's projection is exact GELU, as `RoBERTa`'s is.
    const LAYOUTS: HeadLayouts = HeadLayouts {
        classifier: ClassifierLayout {
            pooling: PoolingMode::Cls,
            projection: Some(Activation::Tanh),
            output_bias: true,
        },
        masked_lm_activation: Some(Activation::GeluExact),
    };

    /// Reads the encoder as `RoBERTa`'s, positions numbered from just after the padding id, with every layer's window halved into its band.
    fn config(source: &Source) -> Result<PostNormEncoderConfig> {
        let fields: Fields = source.json("config.json")?;
        let window = fields.attention_window.first().copied().unwrap_or_default();
        if fields.attention_mode != "longformer"
            || window <= 0
            || window % 2 != 0
            || fields.attention_window.iter().any(|&layer| layer != window)
        {
            return Err(Error::unsupported(
                "a Longformer other than one of sliding windows of one even size in every layer",
            ));
        }

        let mut config = fields.encoder.config(fields.pad_token_id + 1)?;
        config.keys = Some(KeyPattern::Band(window / 2));
        Ok(config)
    }

    fn create(schema: &mut Schema<'_>, config: &PostNormEncoderConfig) -> SingeResult<EncoderBody> {
        post_norm_body(schema, config)
    }

    fn encoder_rules() -> Rules {
        let layers = "longformer.encoder.layer.*.attention.self";
        Rules::new()
            .rename("lm_head.decoder.weight", "embeddings.tokens.weight")
            // Global tokens' projections, which no loaded head reads.
            .skip(&format!("{layers}.query_global"))
            .skip(&format!("{layers}.key_global"))
            .skip(&format!("{layers}.value_global"))
            .then(post_norm_encoder_rules("longformer", ""))
            .skip("lm_head")
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
            _ => Rules::new(),
        }
    }
}
