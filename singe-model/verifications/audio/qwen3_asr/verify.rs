// Qwen3-ASR, transcribing a clip and answering an instruction about one greedily, as `verify.py` does both with `transformers`' `generate`.

use singe::DataType;
use singe_model::task::{
    audio::automatic_speech_recognition::AutomaticSpeechRecognitionPipeline,
    multimodal::audio_text_to_text::AudioTextToTextPipeline,
};

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<AutomaticSpeechRecognitionPipeline>("transcription", DataType::F32);
    session.check::<AudioTextToTextPipeline>("instructed", DataType::F32);
}
