// SAM 2, generating masks of everything in an image and tracking an object through a clip from points or a box, forward and in reverse, as `verify.py` generates them with `transformers`' mask-generation pipeline and tracks it with `Sam2VideoModel`.

use singe_model::task::vision::{
    mask_generation::MaskGenerationPipeline, mask_tracking::MaskTrackingPipeline,
};

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<MaskGenerationPipeline>("everything");
    session.check::<MaskTrackingPipeline>("tracking");
    session.check::<MaskTrackingPipeline>("tracking-outside");
    session.check::<MaskTrackingPipeline>("tracking-reverse");
    session.check::<MaskTrackingPipeline>("tracking-box");
}
