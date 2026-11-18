// DDPM, generating a CIFAR-10 image from a seed, as `verify.py` generates it with `diffusers`.

use singe::DataType;
use singe_model::task::vision::unconditional_image_generation::UnconditionalImageGenerationPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<UnconditionalImageGenerationPipeline>("generation", DataType::F32);
}
