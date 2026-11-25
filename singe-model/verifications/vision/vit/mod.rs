// ViT, classifying images, as `verify.py` classifies them with `transformers`' image-classification pipeline.

use singe_model::task::vision::image_classification::ImageClassificationPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<ImageClassificationPipeline>("classification");
    session.check_onnx("classification");
}
