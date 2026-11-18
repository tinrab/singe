// CLIP, classifying an image among candidate labels, as `verify.py` classifies it with `transformers`' zero-shot image classification pipeline.

use singe::DataType;
use singe_model::task::vision::zero_shot_image_classification::ZeroShotImageClassificationPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<ZeroShotImageClassificationPipeline>("zero-shot", DataType::F32);
}
