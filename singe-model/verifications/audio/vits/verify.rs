// MMS-TTS, a VITS model, speaking text without noise, as `verify.py` speaks it with `transformers`' text-to-speech pipeline.

use singe::DataType;
use singe_model::task::audio::text_to_audio::TextToAudioPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<TextToAudioPipeline>("speech", DataType::F32);
}
