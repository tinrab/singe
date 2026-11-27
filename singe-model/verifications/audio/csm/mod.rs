// CSM, speaking a sentence greedily in F32, alone and after a turn whose clip gives the voice, as `verify.py` does with `transformers` on the host.
// In BF16 the greedy codes part from the reference's within a few frames.

use singe_model::task::audio::text_to_audio::TextToAudioPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<TextToAudioPipeline>("speech");
    session.check::<TextToAudioPipeline>("voice");
}
