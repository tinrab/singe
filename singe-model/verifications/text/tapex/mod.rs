// TAPEX, answering questions about a table, as `verify.py` answers them with `transformers`' BART tokenizer and `generate`.

use singe_model::task::text::table_question_answering::TableQuestionAnsweringPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<TableQuestionAnsweringPipeline>("machines");
}
