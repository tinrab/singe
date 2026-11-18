// RoBERTa: `assembly`'s post-norm encoder with positions numbered after the padding id, serving every head over a text encoder.

mod transformers;

use singe::nn::{Activation, PoolingMode};

use crate::{
    assembly::PostNormEncoderConfig,
    error::Result,
    format::Rules,
    head::{ClassifierLayout, EncoderModel, Head, HeadLayouts, encoder_heads},
    model::ModelEntry,
    source::{Format, Source},
};

#[derive(Debug, Clone, Copy)]
pub struct Roberta;

impl EncoderModel for Roberta {
    // A classifier reads the first token's state through a tanh projection, with no pooler, and the masked language model's projection is exact GELU whatever the encoder's activation.
    const LAYOUTS: HeadLayouts = HeadLayouts {
        classifier: ClassifierLayout {
            pooling: PoolingMode::Cls,
            projection: Some(Activation::Tanh),
            output_bias: true,
        },
        masked_lm_activation: Some(Activation::GeluExact),
    };

    fn config(source: &Source) -> Result<PostNormEncoderConfig> {
        transformers::config(source)
    }

    fn encoder_rules() -> Rules {
        transformers::encoder_rules()
    }

    fn head_rules(head: Head) -> Rules {
        transformers::head_rules(head)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "roberta",
    formats: &[(Format::Transformers, &["roberta", "xlm-roberta"])],
    heads: &encoder_heads::<Roberta>(),
};
