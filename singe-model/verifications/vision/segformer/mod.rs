// SegFormer, segmenting images by label, as `verify.py` segments them with `transformers`' image-segmentation pipeline.

use singe_model::task::vision::image_segmentation::ImageSegmentationPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<ImageSegmentationPipeline>("semantic");
}
