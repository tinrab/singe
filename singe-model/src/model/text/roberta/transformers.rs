//! Reads a `transformers` `RoBERTa` config and imports its checkpoint, whose task heads store the encoder under `roberta`.

use serde::Deserialize;
use singe::{
    Result as SingeResult,
    nn::{Activation, PoolingMode},
};

use crate::{
    assembly::{PostNormEncoderConfig, post_norm_encoder_rules},
    error::Result,
    format::{RuleSet, transformers::EncoderFields},
    head::{ClassifierLayout, EncoderBody, EncoderHeadLayout, EncoderModel, Head, post_norm_body},
    model::text::roberta::Roberta,
    schema::Schema,
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    #[serde(flatten)]
    encoder: EncoderFields,
    pad_token_id: i64,
}

impl EncoderModel for Roberta {
    type Config = PostNormEncoderConfig;

    /// A classifier reads the first token's state through a tanh projection, with no pooler, and the masked language model's projection is exact GELU whatever the encoder's activation.
    const LAYOUTS: EncoderHeadLayout = EncoderHeadLayout {
        classifier: ClassifierLayout {
            pooling: PoolingMode::Cls,
            projection: Some(Activation::Tanh),
            output_bias: true,
        },
        masked_lm_activation: Some(Activation::GeluExact),
    };

    /// Reads a `RoBERTa` config, whose positions number from just after the padding id, as fairseq numbers them.
    fn config(source: &Source) -> Result<PostNormEncoderConfig> {
        let fields: Fields = source.json("config.json")?;
        fields.encoder.config(fields.pad_token_id + 1)
    }

    fn create(schema: &mut Schema<'_>, config: &PostNormEncoderConfig) -> SingeResult<EncoderBody> {
        post_norm_body(schema, config)
    }

    fn encoder_rules() -> RuleSet {
        post_norm_encoder_rules("", "")
            .then(post_norm_encoder_rules("roberta", ""))
            // The heads' tensors, which only their own head's rules read.
            .skip("classifier")
            .skip("qa_outputs")
            .skip("lm_head")
    }

    fn head_rules(head: Head) -> RuleSet {
        match head {
            Head::SequenceClassification => RuleSet::new().rename_each(
                "classifier",
                "classifier",
                &[("dense", "projection"), ("out_proj", "output")],
            ),
            Head::MaskedLm => RuleSet::new()
                .rename_each(
                    "lm_head",
                    "masked_lm",
                    &[
                        ("dense", "projection"),
                        ("layer_norm", "norm"),
                        ("decoder.weight", "output.weight"),
                        ("bias", "output.bias"),
                    ],
                )
                // The same parameter as `lm_head.bias`.
                .skip("lm_head.decoder.bias"),
            _ => RuleSet::new(),
        }
    }
}
