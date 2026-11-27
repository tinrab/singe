//! Models `transformers` documents among its audio models, and the encoder–decoder and transducer speech models they build on.

pub mod audio_spectrogram_transformer;
pub mod clap;
pub mod csm;
pub mod dac;
pub mod encodec;
pub mod fastspeech2_conformer;
pub mod mimi;
pub mod musicgen;
pub mod parakeet;
mod quantizer;
pub mod qwen3_asr;
pub mod silero_vad;
pub mod speech;
pub mod transducer;
pub mod univnet;
pub mod vits;
pub mod wav2vec2;
pub mod whisper;

use crate::model::ModelEntry;

pub(crate) const MODELS: &[&ModelEntry] = &[
    &audio_spectrogram_transformer::MODEL,
    &clap::MODEL,
    &csm::MODEL,
    &dac::MODEL,
    &encodec::MODEL,
    &fastspeech2_conformer::MODEL,
    &mimi::MODEL,
    &musicgen::MODEL,
    &parakeet::MODEL,
    &qwen3_asr::MODEL,
    &univnet::MODEL,
    &vits::MODEL,
    &wav2vec2::MODEL,
    &whisper::MODEL,
];
