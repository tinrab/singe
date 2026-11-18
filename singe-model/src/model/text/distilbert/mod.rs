// DistilBERT: `assembly`'s post-norm encoder without token types, serving every head over a text encoder.

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
pub struct DistilBert;

impl EncoderModel for DistilBert {
    // A classifier reads the first token's state through `pre_classifier` and ReLU.
    const LAYOUTS: HeadLayouts = HeadLayouts {
        classifier: ClassifierLayout {
            pooling: PoolingMode::Cls,
            projection: Some(Activation::Relu),
            output_bias: true,
        },
        masked_lm_activation: None,
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
    name: "distilbert",
    formats: &[(Format::Transformers, &["distilbert"])],
    heads: &encoder_heads::<DistilBert>(),
};
