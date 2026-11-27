// Deformable DETR, detecting objects, as `verify.py` detects them with `transformers`' object-detection pipeline.
// Its served and CUDA programs read the ONNX file's marked deformable attention as the operation, so they sample with the fused kernel rather than the gathers onnxruntime runs, whose corners would take 305 MB each per encoder layer at 800 × 1066.

use singe_model::task::vision::object_detection::ObjectDetectionPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<ObjectDetectionPipeline>("detection");
    session.check_onnx("detection");
    session.check_served::<ObjectDetectionPipeline>(
        "detection",
        &[("batch_size", 1)],
        &[("height", 800), ("width", 800)],
    );
    // A step for each image's processed size, since a program's steps are fixed.
    session.check_cuda::<ObjectDetectionPipeline>(
        "detection",
        &[
            ("batch_size", 1),
            ("height", 800),
            ("width", 1066),
            ("width", 1120),
        ],
    );
}
