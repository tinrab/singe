// SAM, generating masks of everything in an image, as `verify.py` generates them with `transformers`' mask-generation pipeline.

use singe_model::task::vision::mask_generation::MaskGenerationPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<MaskGenerationPipeline>("everything");
}
