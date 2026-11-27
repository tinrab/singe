// MusicGen, generating a clip from a text under the generation config's guidance, greedily and sampled from the likeliest code alone, as `verify.py` does with `transformers`' `generate`.

use singe_model::task::audio::text_to_audio::TextToAudioPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<TextToAudioPipeline>("music");
    session.check::<TextToAudioPipeline>("sampled");
}
