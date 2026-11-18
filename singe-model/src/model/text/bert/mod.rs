// BERT: `assembly`'s post-norm encoder, serving every head over a text encoder.

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
pub struct Bert;

impl EncoderModel for Bert {
    // A classifier reads the pooler, a tanh projection of the first token's state.
    const LAYOUTS: HeadLayouts = HeadLayouts {
        classifier: ClassifierLayout {
            pooling: PoolingMode::Cls,
            projection: Some(Activation::Tanh),
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
    name: "bert",
    formats: &[(Format::Transformers, &["bert"])],
    heads: &encoder_heads::<Bert>(),
};
