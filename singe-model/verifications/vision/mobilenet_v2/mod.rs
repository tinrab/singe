// MobileNetV2, classifying images and segmenting them with DeepLabV3's checkpoint, as `verify.py` does with `transformers`' image-classification and image-segmentation pipelines.

use singe_model::task::vision::{
    image_classification::ImageClassificationPipeline,
    image_segmentation::ImageSegmentationPipeline,
};

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<ImageClassificationPipeline>("classification");
    session.check_onnx("classification");
    session.check_cuda::<ImageClassificationPipeline>("classification", &[("batch_size", 2)]);
    session.check::<ImageSegmentationPipeline>("segmentation");
    session.check_onnx("segmentation");
    session.check_cuda::<ImageSegmentationPipeline>("segmentation", &[("batch_size", 1)]);
}
