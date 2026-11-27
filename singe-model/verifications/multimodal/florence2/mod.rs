// Florence-2, answering caption, reading, object detection, regions with their text, phrase grounding, segmentation, and region proposal prompts about images, and parsing the answers that locate things, as `verify.py` answers and parses them with `transformers`.

use singe_model::task::vision::image_to_text::ImageToTextPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    for name in [
        "caption",
        "ocr",
        "objects",
        "regions",
        "grounding",
        "segmentation",
        "proposals",
    ] {
        session.check::<ImageToTextPipeline>(name);
    }
}
