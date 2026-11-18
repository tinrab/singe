// The Audio Spectrogram Transformer, classifying a clip, as `verify.py` classifies it with `transformers`' audio-classification pipeline.

use singe::DataType;
use singe_model::task::audio::audio_classification::AudioClassificationPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<AudioClassificationPipeline>("speech", DataType::F32);
}
