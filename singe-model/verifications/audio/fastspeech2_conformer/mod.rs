// FastSpeech2Conformer with HiFi-GAN, speaking ARPAbet phonemes, as `verify.py` does with `transformers`' `FastSpeech2ConformerWithHifiGan`.

use singe_model::task::audio::text_to_audio::TextToAudioPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<TextToAudioPipeline>("speech");
}
