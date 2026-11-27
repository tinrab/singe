// Grounding DINO, finding objects of candidate labels, as `verify.py` finds them with `transformers`' zero-shot object detection pipeline.

use singe_model::task::vision::zero_shot_object_detection::ZeroShotObjectDetectionPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<ZeroShotObjectDetectionPipeline>("detection");
}
