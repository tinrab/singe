// Depth Anything V2, estimating each pixel's depth, as `verify.py` estimates it with `transformers`' depth-estimation pipeline.

use singe_model::task::vision::depth_estimation::DepthEstimationPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<DepthEstimationPipeline>("depth");
    session.check_onnx_at("depth", &[1, 3, 518, 518]);
}
