// TrOCR, reading a printed line, as `verify.py` reads it with `transformers`' processor, model, and `generate`.

use singe_model::task::vision::image_to_text::ImageToTextPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<ImageToTextPipeline>("printed");
}
