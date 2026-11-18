// Qwen3-Embedding, scoring sentences' similarity to a source sentence and giving each token's features, as `verify.py` scores them with `sentence-transformers`.

use singe::DataType;
use singe_model::task::text::{
    feature_extraction::FeatureExtractionPipeline, sentence_similarity::SentenceSimilarityPipeline,
};

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<SentenceSimilarityPipeline>("similarity", DataType::F32);
    session.check::<FeatureExtractionPipeline>("features", DataType::F32);
}
