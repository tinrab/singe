// Janus-Pro-1B, describing an image in BF16 and generating one in F32, as `verify.py` does with `transformers`.
// In BF16 the generated codes part on the second step, where guidance at scale 5 magnifies rounding past the 0.375 between the top two codes, so generation runs in F32, whose 6.1 GB of weights fit the 8 GB GPU.

use singe_model::task::{
    multimodal::image_text_to_text::ImageTextToTextPipeline,
    vision::text_to_image::TextToImagePipeline,
};

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<ImageTextToTextPipeline>("image");
    session.check::<TextToImagePipeline>("generate");
}
