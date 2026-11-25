// DINOv2, pooling each image's features, as `verify.py` extracts them in `transformers`, on images of two sizes, whose positions it resizes.

use singe_model::task::vision::image_feature_extraction::ImageFeatureExtractionPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<ImageFeatureExtractionPipeline>("pooled");
    session.check_onnx("pooled");
}
