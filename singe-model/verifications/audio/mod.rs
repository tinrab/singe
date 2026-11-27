// Verifications of the models under `model::audio`.

mod audio_spectrogram_transformer;
mod clap;
mod csm;
mod dac;
mod encodec;
mod fastspeech2_conformer;
mod mimi;
mod musicgen;
mod parakeet;
mod qwen3_asr;
mod univnet;
mod vits;
mod wav2vec2;
mod whisper;

use crate::support::verifications;

verifications!(
    "audio",
    [
        audio_spectrogram_transformer,
        clap,
        csm,
        dac,
        encodec,
        fastspeech2_conformer,
        mimi,
        musicgen,
        parakeet,
        qwen3_asr,
        univnet,
        vits,
        wav2vec2,
        whisper,
    ]
);
