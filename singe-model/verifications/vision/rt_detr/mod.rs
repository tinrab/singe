// RT-DETR and RT-DETRv2, detecting objects, as `verify.py` detects them with `transformers`' object-detection pipeline.

use singe_model::task::vision::object_detection::ObjectDetectionPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<ObjectDetectionPipeline>("detection");
    session.check_onnx("detection");
    session.check_served::<ObjectDetectionPipeline>("detection", &[("batch_size", 1)], &[]);
    session.check_cuda::<ObjectDetectionPipeline>("detection", &[("batch_size", 1)]);
    session.check::<ObjectDetectionPipeline>("detection_v2");
}
