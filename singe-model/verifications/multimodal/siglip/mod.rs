// SigLIP and SigLIP 2 NaFlex, classifying images among candidate labels by sigmoid scores, as `verify.py` classifies it with `transformers`' zero-shot image classification pipeline.

use singe_model::task::vision::zero_shot_image_classification::ZeroShotImageClassificationPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<ZeroShotImageClassificationPipeline>("zero-shot");
    session.check::<ZeroShotImageClassificationPipeline>("portrait");
    session.check::<ZeroShotImageClassificationPipeline>("naflex");
    session.check::<ZeroShotImageClassificationPipeline>("naflex-portrait");
}
