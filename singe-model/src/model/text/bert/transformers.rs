//! Reads a `transformers` BERT config and imports its checkpoint, whose task heads store the encoder under `bert`.

use singe::nn::{Activation, PoolingMode};

use crate::{
    assembly::{PostNormEncoderConfig, post_norm_encoder_rules},
    error::Result,
    format::{Rules, transformers::EncoderFields},
    head::{ClassifierLayout, EncoderModel, Head, HeadLayouts},
    model::text::bert::Bert,
    source::Source,
};

impl EncoderModel for Bert {
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
        source.json::<EncoderFields>("config.json")?.config(0)
    }

    fn encoder_rules() -> Rules {
        post_norm_encoder_rules("", "")
            .then(post_norm_encoder_rules("bert", ""))
            // The heads' tensors, which only their own head's rules read.
            .skip("cls")
            .skip("classifier")
            .skip("qa_outputs")
    }

    fn head_rules(head: Head) -> Rules {
        match head {
            Head::SequenceClassification => Rules::new()
                .rename("bert.pooler.dense", "classifier.projection")
                .rename("classifier", "classifier.output"),
            Head::MaskedLm => Rules::new()
                .rename_each(
                    "cls.predictions",
                    "masked_lm",
                    &[
                        ("transform.dense", "projection"),
                        ("transform.LayerNorm", "norm"),
                        ("decoder.weight", "output.weight"),
                        ("bias", "output.bias"),
                    ],
                )
                // The same parameter as `cls.predictions.bias`.
                .skip("cls.predictions.decoder.bias"),
            _ => Rules::new(),
        }
    }
}
