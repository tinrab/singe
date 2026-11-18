// Reads a `transformers` RoBERTa config and imports its checkpoint, whose task heads store the encoder under `roberta.`.

use serde::Deserialize;

use crate::{
    assembly::{PostNormEncoderConfig, post_norm_encoder_rules},
    error::Result,
    format::{Rules, transformers::EncoderFields},
    head::Head,
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    #[serde(flatten)]
    encoder: EncoderFields,
    pad_token_id: i64,
}

// RoBERTa numbers a sequence's positions from just after its padding id, as fairseq did.
pub(crate) fn config(source: &Source) -> Result<PostNormEncoderConfig> {
    let fields: Fields = source.json("config.json")?;
    fields.encoder.config(fields.pad_token_id + 1)
}

pub(crate) fn encoder_rules() -> Rules {
    post_norm_encoder_rules("", "")
        .then(post_norm_encoder_rules("roberta.", ""))
        // The heads' tensors, which only their own head's rules read.
        .skip("classifier")
        .skip("qa_outputs")
        .skip("lm_head")
}

pub(crate) fn head_rules(head: Head) -> Rules {
    match head {
        Head::SequenceClassification => Rules::new()
            .rename("classifier.dense", "classifier.projection")
            .rename("classifier.out_proj", "classifier.output"),
        Head::TokenClassification => Rules::new().rename("classifier", "classifier.output"),
        Head::QuestionAnswering => Rules::new().rename("qa_outputs", "span.output"),
        Head::MaskedLm => Rules::new()
            .rename("lm_head.dense", "masked_lm.projection")
            .rename("lm_head.layer_norm", "masked_lm.norm")
            .rename("lm_head.decoder.weight", "masked_lm.output.weight")
            .rename("lm_head.bias", "masked_lm.output.bias")
            // The same parameter as `lm_head.bias`.
            .skip("lm_head.decoder.bias"),
        _ => Rules::new(),
    }
}
