// EfficientLoFTR, matching two images' keypoints, as `verify.py` matches them with `transformers`' processor, model, and post-processing.

use singe_model::task::vision::keypoint_matching::KeypointMatchingPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<KeypointMatchingPipeline>("matches");
}
