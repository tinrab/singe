//! Reads a `transformers` `BigBird` config and imports its checkpoint, whose heads store the encoder under `bert`, as BERT's do.

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
    model::text::big_bird::BigBird,
    schema::Schema,
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    #[serde(flatten)]
    encoder: EncoderFields,
    attention_type: String,
    block_size: i64,
    num_random_blocks: i64,
    #[serde(default)]
    rescale_embeddings: bool,
}

impl EncoderModel for BigBird {
    type Config = PostNormEncoderConfig;

    /// A classifier reads the pooler, a tanh projection of the first token's state.
    const LAYOUTS: HeadLayouts = HeadLayouts {
        classifier: ClassifierLayout {
            pooling: PoolingMode::Cls,
            projection: Some(Activation::Tanh),
            output_bias: true,
        },
        masked_lm_activation: None,
    };

    fn config(source: &Source) -> Result<PostNormEncoderConfig> {
        let fields: Fields = source.json("config.json")?;
        if fields.rescale_embeddings {
            return Err(Error::unsupported("BigBird with rescaled embeddings"));
        }

        let keys = match fields.attention_type.as_str() {
            "original_full" => None,
            "block_sparse" => Some(KeyPattern::Blocks {
                size: fields.block_size,
                random: fields.num_random_blocks,
            }),
            other => {
                return Err(Error::unsupported(format!(
                    "BigBird attention type `{other}`"
                )));
            }
        };
        let mut config = fields.encoder.config(0)?;
        config.keys = keys;
        Ok(config)
    }

    fn create(schema: &mut Schema<'_>, config: &PostNormEncoderConfig) -> SingeResult<EncoderBody> {
        post_norm_body(schema, config)
    }

    fn encoder_rules() -> Rules {
        post_norm_encoder_rules("bert", "")
            // The next-sentence head of pre-training, which no task here reads.
            .skip("cls.seq_relationship")
    }

    fn head_rules(head: Head) -> Rules {
        match head {
            Head::SequenceClassification => Rules::new()
                .rename("bert.pooler", "classifier.projection")
                .rename("classifier", "classifier.output"),
            Head::MaskedLm => Rules::new().rename_each(
                "cls.predictions",
                "masked_lm",
                &[
                    ("transform.dense", "projection"),
                    ("transform.LayerNorm", "norm"),
                    ("bias", "output.bias"),
                ],
            ),
            _ => Rules::new(),
        }
    }
}
