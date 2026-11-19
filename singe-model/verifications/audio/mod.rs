// Verifications of the models under `model::audio`.

mod audio_spectrogram_transformer;
mod clap;
mod dac;
mod qwen3_asr;
mod vits;
mod wav2vec2;
mod whisper;

use crate::support::verifications;

verifications!(
    "audio",
    [
        audio_spectrogram_transformer,
        clap,
        dac,
        qwen3_asr,
        vits,
        wav2vec2,
        whisper,
    ]
);
