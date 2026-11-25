// DistilBERT, through every task a DistilBERT snapshot here serves, as `verify.py` runs them in `transformers`: a sentiment classifier, an inference model classifying zero-shot, a named-entity tagger, a question answerer over short and long contexts, the pretrained masked language model filling blanks, and a sentence embedder that max-pools its tokens.

use singe_model::task::text::{
    fill_mask::FillMaskPipeline, question_answering::QuestionAnsweringPipeline,
    sentence_similarity::SentenceSimilarityPipeline,
    text_classification::TextClassificationPipeline,
    token_classification::TokenClassificationPipeline,
    zero_shot_classification::ZeroShotClassificationPipeline,
};

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<TextClassificationPipeline>("sentiment");
    session.check_onnx("sentiment");
    session.check_cuda::<TextClassificationPipeline>(
        "sentiment",
        &[("batch_size", 2), ("sequence_length", 64)],
    );
    session.check::<ZeroShotClassificationPipeline>("zero-shot");
    session.check_onnx("zero-shot");
    session.check_cuda::<ZeroShotClassificationPipeline>(
        "zero-shot",
        &[("batch_size", 8), ("sequence_length", 64)],
    );
    session.check::<ZeroShotClassificationPipeline>("zero-shot-multi-label");
    session.check::<TokenClassificationPipeline>("entities-simple");
    session.check::<TokenClassificationPipeline>("entities-first");
    session.check::<QuestionAnsweringPipeline>("answers");
    session.check_onnx("answers");
    session.check_cuda::<QuestionAnsweringPipeline>(
        "answers",
        &[("batch_size", 16), ("sequence_length", 384)],
    );
    session.check::<QuestionAnsweringPipeline>("answers-small-windows");
    session.check::<QuestionAnsweringPipeline>("answers-bf16");
    session.check::<FillMaskPipeline>("fill");
    session.check::<SentenceSimilarityPipeline>("similarity-max");
}
