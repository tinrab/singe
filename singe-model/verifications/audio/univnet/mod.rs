// UnivNet, resynthesizing a 24 kHz clip from its spectrogram and seeded noise, as `verify.py` does with `transformers`' `UnivNetModel` and the same noise.

use singe_model::task::audio::audio_to_audio::AudioToAudioPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<AudioToAudioPipeline>("resynthesis");
}
