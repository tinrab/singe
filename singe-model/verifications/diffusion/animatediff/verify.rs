// AnimateDiff, generating a clip from a prompt and a seed, as `verify.py` generates it with `diffusers`.

use singe::DataType;
use singe_model::task::vision::text_to_video::TextToVideoPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<TextToVideoPipeline>("generation", DataType::F16);
}
