// SmolVLM2, answering a question about an image, describing an image, and describing a clip greedily, as `verify.py` does with `transformers`.

use singe_model::task::multimodal::{
    image_text_to_text::ImageTextToTextPipeline, video_text_to_text::VideoTextToTextPipeline,
    visual_question_answering::VisualQuestionAnsweringPipeline,
};

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<VisualQuestionAnsweringPipeline>("question");
    session.check::<ImageTextToTextPipeline>("image");
    session.check::<VideoTextToTextPipeline>("clip");
}
