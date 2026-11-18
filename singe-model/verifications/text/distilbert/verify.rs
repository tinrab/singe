// DistilBERT, through every task a DistilBERT snapshot here serves, as `verify.py` runs them in `transformers`: a sentiment classifier, an inference model classifying zero-shot, a named-entity tagger, a question answerer over short and long contexts, the pretrained masked language model filling blanks, and a sentence embedder that max-pools its tokens.

use singe::DataType;
use singe_model::task::text::{
    fill_mask::FillMaskPipeline, question_answering::QuestionAnsweringPipeline,
    sentence_similarity::SentenceSimilarityPipeline,
    text_classification::TextClassificationPipeline,
    token_classification::TokenClassificationPipeline,
    zero_shot_classification::ZeroShotClassificationPipeline,
};

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<TextClassificationPipeline>("sentiment", DataType::F32);
    session.check::<ZeroShotClassificationPipeline>("zero-shot", DataType::F32);
    session.check::<ZeroShotClassificationPipeline>("zero-shot-multi-label", DataType::F32);
    session.check::<TokenClassificationPipeline>("entities-simple", DataType::F32);
    session.check::<TokenClassificationPipeline>("entities-first", DataType::F32);
    session.check::<QuestionAnsweringPipeline>("answers", DataType::F32);
    session.check::<QuestionAnsweringPipeline>("answers-small-windows", DataType::F32);
    session.check::<FillMaskPipeline>("fill", DataType::F32);
    session.check::<SentenceSimilarityPipeline>("similarity-max", DataType::F32);
}
