// Swin2SR, doubling images, as `verify.py` doubles them with `transformers`' processor and model.

use singe::DataType;
use singe_model::task::vision::image_to_image::ImageToImagePipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<ImageToImagePipeline>("upscale", DataType::F32);
}
