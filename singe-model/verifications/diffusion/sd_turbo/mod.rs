// SD-Turbo, generating an image from a prompt and a seed in one step in F16, as `verify.py` generates it with `diffusers`.

use singe_model::task::vision::text_to_image::TextToImagePipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<TextToImagePipeline>("generation");
}
