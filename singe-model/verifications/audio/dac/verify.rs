// DAC, encoding a clip to its codes and decoding it back, as `verify.py` does with `transformers`' `DacModel`.

use singe::DataType;
use singe_model::task::audio::audio_to_audio::AudioToAudioPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<AudioToAudioPipeline>("reconstruction", DataType::F32);
}
