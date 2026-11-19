// Whisper, transcribing a clip greedily in English, through its pipeline and as an exported CUDA C++ generation loop, as `verify.py` transcribes it with `transformers`' `generate`.

use singe_model::task::audio::automatic_speech_recognition::AutomaticSpeechRecognitionPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<AutomaticSpeechRecognitionPipeline>("transcription");
    session.check_exported("transcription");
}
