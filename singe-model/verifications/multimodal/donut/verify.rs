// Donut, answering questions about a document image, as `verify.py` answers them with `transformers`' document-question-answering pipeline.

use singe::DataType;
use singe_model::task::multimodal::document_question_answering::DocumentQuestionAnsweringPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<DocumentQuestionAnsweringPipeline>("invoice", DataType::F32);
}
