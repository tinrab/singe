// Verifications of the models under `model::audio`.

#[path = "audio_spectrogram_transformer/verify.rs"]
mod audio_spectrogram_transformer;
#[path = "clap/verify.rs"]
mod clap;
#[path = "dac/verify.rs"]
mod dac;
#[path = "qwen3_asr/verify.rs"]
mod qwen3_asr;
#[path = "vits/verify.rs"]
mod vits;
#[path = "wav2vec2/verify.rs"]
mod wav2vec2;
#[path = "whisper/verify.rs"]
mod whisper;

use crate::support::Verification;

pub const VERIFICATIONS: &[Verification] = &[
    Verification {
        name: "audio/audio_spectrogram_transformer",
        verify: audio_spectrogram_transformer::verify,
    },
    Verification {
        name: "audio/clap",
        verify: clap::verify,
    },
    Verification {
        name: "audio/dac",
        verify: dac::verify,
    },
    Verification {
        name: "audio/qwen3_asr",
        verify: qwen3_asr::verify,
    },
    Verification {
        name: "audio/vits",
        verify: vits::verify,
    },
    Verification {
        name: "audio/wav2vec2",
        verify: wav2vec2::verify,
    },
    Verification {
        name: "audio/whisper",
        verify: whisper::verify,
    },
];
