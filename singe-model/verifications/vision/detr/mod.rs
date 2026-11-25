// DETR, detecting objects, as `verify.py` detects them with `transformers`' object-detection pipeline.

use singe_model::task::vision::object_detection::ObjectDetectionPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<ObjectDetectionPipeline>("detection");
    session.check_onnx_at("detection", &[1, 3, 800, 800]);
}
