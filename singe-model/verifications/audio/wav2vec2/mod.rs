// Wav2Vec2, transcribing a clip by CTC, as `verify.py` transcribes it with `transformers`' automatic-speech-recognition pipeline.

use singe_model::task::audio::automatic_speech_recognition::AutomaticSpeechRecognitionPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<AutomaticSpeechRecognitionPipeline>("transcription");
    session.check_onnx("transcription");
}
