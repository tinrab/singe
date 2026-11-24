//! Models `transformers` documents among its audio models, and the encoder–decoder and transducer speech models they build on.

pub mod audio_spectrogram_transformer;
pub mod clap;
pub mod dac;
pub mod parakeet;
pub mod qwen3_asr;
pub mod speech;
pub mod transducer;
pub mod vits;
pub mod wav2vec2;
pub mod whisper;

use crate::model::ModelEntry;

pub(crate) const MODELS: &[&ModelEntry] = &[
    &audio_spectrogram_transformer::MODEL,
    &clap::MODEL,
    &dac::MODEL,
    &parakeet::MODEL,
    &qwen3_asr::MODEL,
    &vits::MODEL,
    &wav2vec2::MODEL,
    &whisper::MODEL,
];
