// Reads a `transformers` BERT config and imports its checkpoint, whose task heads store the encoder under `bert.`.

use crate::{
    assembly::{PostNormEncoderConfig, post_norm_encoder_rules},
    error::Result,
    format::{Rules, transformers::EncoderFields},
    head::Head,
    source::Source,
};

pub(crate) fn config(source: &Source) -> Result<PostNormEncoderConfig> {
    source.json::<EncoderFields>("config.json")?.config(0)
}

pub(crate) fn encoder_rules() -> Rules {
    post_norm_encoder_rules("", "")
        .then(post_norm_encoder_rules("bert.", ""))
        // The heads' tensors, which only their own head's rules read.
        .skip("cls")
        .skip("classifier")
        .skip("qa_outputs")
}

pub(crate) fn head_rules(head: Head) -> Rules {
    match head {
        Head::SequenceClassification => Rules::new()
            .rename("bert.pooler.dense", "classifier.projection")
            .rename("classifier", "classifier.output"),
        Head::TokenClassification => Rules::new().rename("classifier", "classifier.output"),
        Head::QuestionAnswering => Rules::new().rename("qa_outputs", "span.output"),
        Head::MaskedLm => Rules::new()
            .rename("cls.predictions.transform.dense", "masked_lm.projection")
            .rename("cls.predictions.transform.LayerNorm", "masked_lm.norm")
            .rename("cls.predictions.decoder.weight", "masked_lm.output.weight")
            .rename("cls.predictions.bias", "masked_lm.output.bias")
            // The same parameter as `cls.predictions.bias`.
            .skip("cls.predictions.decoder.bias"),
        _ => Rules::new(),
    }
}
