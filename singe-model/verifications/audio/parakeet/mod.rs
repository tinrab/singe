// Parakeet, transcribing a clip, as WAV and as FLAC, its first second, and the clip 8, 50, and 150 times over with a TDT transducer, the 8 times also attending locally with a window wider than the clip, an RNN-T transducer, and a CTC head, as `verify.py` does each with `transformers`' `generate`.

use singe_model::task::audio::automatic_speech_recognition::AutomaticSpeechRecognitionPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<AutomaticSpeechRecognitionPipeline>("tdt");
    session.check::<AutomaticSpeechRecognitionPipeline>("tdt-flac");
    session.check::<AutomaticSpeechRecognitionPipeline>("tdt-short");
    session.check::<AutomaticSpeechRecognitionPipeline>("tdt-long");
    session.check::<AutomaticSpeechRecognitionPipeline>("tdt-long-local");
    session.check::<AutomaticSpeechRecognitionPipeline>("tdt-4m");
    session.check::<AutomaticSpeechRecognitionPipeline>("tdt-12m");
    session.check::<AutomaticSpeechRecognitionPipeline>("rnnt");
    session.check::<AutomaticSpeechRecognitionPipeline>("ctc");
}
