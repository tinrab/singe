// SAM 3 on random weights, finding what a text names in an image, as `verify.py` finds it with `transformers`' processor, model, and post-processing.

use singe_model::task::vision::image_segmentation::ImageSegmentationPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<ImageSegmentationPipeline>("concept");
}
