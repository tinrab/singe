// TAESD, encoding an image to latents and decoding it back, as `verify.py` reconstructs it with `diffusers`' `AutoencoderTiny`.

use singe::DataType;
use singe_model::task::vision::image_to_image::ImageToImagePipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<ImageToImagePipeline>("reconstruction", DataType::F32);
}
