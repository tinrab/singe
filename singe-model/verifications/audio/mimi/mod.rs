// Mimi, encoding a 24 kHz clip to all its codebooks' codes and decoding it back, as `verify.py` does with `transformers`' `MimiModel`.

use singe_model::task::audio::audio_to_audio::AudioToAudioPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<AudioToAudioPipeline>("reconstruction");
}
