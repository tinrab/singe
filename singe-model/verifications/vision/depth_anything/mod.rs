// Depth Anything V2, estimating each pixel's depth, as `verify.py` estimates it with `transformers`' depth-estimation pipeline.

use singe_model::task::vision::depth_estimation::DepthEstimationPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<DepthEstimationPipeline>("depth");
    session.check_onnx("depth");
    session.check_served::<DepthEstimationPipeline>(
        "depth",
        &[("batch_size", 1)],
        &[("height", 224), ("width", 224)],
    );
    // A step for each image's processed size, since a program's steps are fixed.
    session.check_cuda::<DepthEstimationPipeline>(
        "depth",
        &[
            ("batch_size", 1),
            ("height", 518),
            ("width", 686),
            ("width", 322),
        ],
    );
}
