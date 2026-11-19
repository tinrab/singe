// CLAP, classifying a clip among candidate labels, as `verify.py` classifies it with `transformers`' zero-shot audio classification pipeline.

use singe_model::task::audio::zero_shot_audio_classification::ZeroShotAudioClassificationPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<ZeroShotAudioClassificationPipeline>("sounds");
}
