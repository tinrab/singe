// EnCodec, encoding a 24 kHz clip to its default and its largest bandwidth's codes and decoding it back, as `verify.py` does with `transformers`' `EncodecModel`.

use singe_model::task::audio::audio_to_audio::AudioToAudioPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<AudioToAudioPipeline>("reconstruction");
    session.check::<AudioToAudioPipeline>("full-bandwidth");
}
