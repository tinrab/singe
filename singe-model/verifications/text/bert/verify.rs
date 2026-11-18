// BERT, through every task a BERT snapshot here serves, as `verify.py` runs them in `transformers`: a toxicity classifier, a spam classifier, a cross-encoder ranking texts, a named-entity tagger with each way of grouping words, a sentence embedder, and the pretrained masked language model filling blanks; the spam classifier also runs as an exported CUDA C++ program.

use singe::DataType;
use singe_model::task::text::{
    feature_extraction::FeatureExtractionPipeline, fill_mask::FillMaskPipeline,
    sentence_similarity::SentenceSimilarityPipeline,
    text_classification::TextClassificationPipeline, text_ranking::TextRankingPipeline,
    token_classification::TokenClassificationPipeline,
};

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<TextClassificationPipeline>("toxicity", DataType::F32);
    session.check::<TextClassificationPipeline>("spam", DataType::F32);
    session.check_exported("spam");
    session.check::<TextClassificationPipeline>("toxicity-bf16", DataType::Bf16);
    session.check::<TextRankingPipeline>("ranking", DataType::F32);
    for strategy in ["none", "simple", "first", "average", "max"] {
        session
            .check::<TokenClassificationPipeline>(&format!("entities-{strategy}"), DataType::F32);
    }
    session.check::<FeatureExtractionPipeline>("features", DataType::F32);
    session.check::<SentenceSimilarityPipeline>("similarity", DataType::F32);
    session.check::<FillMaskPipeline>("fill", DataType::F32);
}
