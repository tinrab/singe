// Mask2Former, segmenting images into panoptic segments, as `verify.py` segments them with `transformers`' image-segmentation pipeline.
// It has no ONNX check: each decoder layer attends where a mask's probability is below 0.5, and on these images some come within 6e-7 of it, so onnxruntime's summation order flips a bit of a mask that `transformers` and Singe's CUDA program agree on.

use singe_model::task::vision::image_segmentation::ImageSegmentationPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<ImageSegmentationPipeline>("panoptic");
    session.check_cuda::<ImageSegmentationPipeline>("panoptic", &[("batch_size", 1)]);
}
