//! Reads a `transformers` Audio Spectrogram Transformer config and its feature extractor's, and imports its checkpoint, a `ViT`'s under `audio_spectrogram_transformer.`.

use serde::Deserialize;

use crate::{
    error::Result,
    format::{RuleSet, transformers::classifier_config},
    model::{
        audio::audio_spectrogram_transformer::AstConfig,
        vision::vit::transformers::{VitFields, encoder_rules},
    },
    processor::fbank::FilterBankOptions,
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    #[serde(flatten)]
    vit: VitFields,
    frequency_stride: i64,
    time_stride: i64,
    num_mel_bins: i64,
    max_length: i64,
}

pub(crate) fn config(source: &Source) -> Result<AstConfig> {
    let fields: Fields = source.json("config.json")?;
    Ok(AstConfig {
        encoder: fields.vit.encoder()?,
        layers: fields.vit.num_hidden_layers,
        patch_size: fields.vit.patch_size,
        frequency_stride: fields.frequency_stride,
        time_stride: fields.time_stride,
        mels: fields.num_mel_bins,
        frames: fields.max_length,
        classifier: classifier_config(source)?,
        features: FilterBankOptions::load(source)?,
    })
}

pub(crate) fn rules() -> RuleSet {
    encoder_rules("audio_spectrogram_transformer", "")
        .rename(
            "audio_spectrogram_transformer.embeddings.distillation_token",
            "embeddings.distillation",
        )
        .rename("classifier.layernorm", "head_norm")
        .rename("classifier.dense", "head")
}
