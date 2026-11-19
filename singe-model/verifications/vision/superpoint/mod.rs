// SuperPoint, detecting and describing keypoints, as `verify.py` detects them with `transformers`' processor, model, and post-processing.

use singe_model::task::vision::keypoint_detection::KeypointDetectionPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<KeypointDetectionPipeline>("keypoints");
}
