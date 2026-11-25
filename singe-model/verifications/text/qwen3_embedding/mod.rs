// Qwen3-Embedding, scoring sentences' similarity to a source sentence and giving each token's features, as `verify.py` scores them with `sentence-transformers`, natively and exported as ONNX.

use singe_model::task::text::{
    feature_extraction::FeatureExtractionPipeline, sentence_similarity::SentenceSimilarityPipeline,
};

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<SentenceSimilarityPipeline>("similarity");
    session.check_onnx("similarity");
    session.check::<FeatureExtractionPipeline>("features");
    session.check_onnx("features");
}
