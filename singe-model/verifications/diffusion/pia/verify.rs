// PIA, animating an image after a prompt and after an empty one, as `verify.py` animates it with `diffusers`.

use singe::DataType;
use singe_model::task::{
    multimodal::image_text_to_video::ImageTextToVideoPipeline,
    vision::image_to_video::ImageToVideoPipeline,
};

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<ImageTextToVideoPipeline>("animation", DataType::F16);
    session.check::<ImageToVideoPipeline>("image", DataType::F16);
}
