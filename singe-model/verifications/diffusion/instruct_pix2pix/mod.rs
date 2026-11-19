// InstructPix2Pix, editing an image by an instruction, as `verify.py` edits it with `diffusers`.

use singe_model::task::multimodal::image_text_to_image::ImageTextToImagePipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<ImageTextToImagePipeline>("edit");
}
