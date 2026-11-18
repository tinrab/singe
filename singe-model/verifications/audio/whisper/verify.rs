// Whisper, transcribing a clip greedily in English, as `verify.py` transcribes it with `transformers`' `generate`.

use singe::DataType;
use singe_model::task::audio::automatic_speech_recognition::AutomaticSpeechRecognitionPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<AutomaticSpeechRecognitionPipeline>("transcription", DataType::F32);
}
