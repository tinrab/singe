// BERT, through every task a BERT snapshot here serves, as `verify.py` runs them in `transformers`: a toxicity classifier, a spam classifier, a cross-encoder ranking texts, a named-entity tagger with each way of grouping words, a sentence embedder, and its embeddings pooled by the mean over the square root of the length in a snapshot derived from it, and the pretrained masked language model filling blanks; the spam classifier also runs as an exported CUDA C++ program.

use anyhow::{Result, bail};
use singe::Error as SingeError;
use singe_model::{
    LoadOptions, Source,
    task::{
        Pipeline,
        text::{
            feature_extraction::FeatureExtractionPipeline,
            fill_mask::FillMaskPipeline,
            sentence_similarity::{SentenceSimilarityParameters, SentenceSimilarityPipeline},
            text_classification::TextClassificationPipeline,
            text_ranking::TextRankingPipeline,
            token_classification::TokenClassificationPipeline,
        },
    },
};

use crate::support::{Runner, Session};

// A `sentence-transformers` snapshot's pooled vector of each sentence, compared whole, since a similarity would hide its scale.
struct SentenceEmbeddings(SentenceSimilarityPipeline);

impl Runner for SentenceEmbeddings {
    const TASK: &'static str = "sentence-embeddings";

    type Input = String;

    type Parameters = SentenceSimilarityParameters;

    type Output = Vec<f32>;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self(<SentenceSimilarityPipeline as Pipeline>::load(
            source, options,
        )?))
    }

    fn run(
        &self,
        inputs: &[String],
        parameters: &SentenceSimilarityParameters,
    ) -> Result<Vec<Vec<f32>>> {
        let texts = inputs.iter().map(String::as_str).collect::<Vec<_>>();
        let vectors = self.0.embed(&texts, parameters.max_length)?;
        let &[_, width] = vectors.shape().dimensions() else {
            bail!("embeddings of shape {:?}", vectors.shape().dimensions());
        };
        let width =
            usize::try_from(width).map_err(|_| SingeError::out_of_range("embedding width"))?;
        Ok(vectors
            .to_vec::<f32>()?
            .chunks(width)
            .map(<[f32]>::to_vec)
            .collect())
    }
}

pub fn verify(session: &mut Session) {
    session.check::<TextClassificationPipeline>("toxicity");
    session.check::<TextClassificationPipeline>("spam");
    session.check_exported("spam");
    session.check_onnx("spam");
    session.check_onnx("toxicity");
    session.check::<TextClassificationPipeline>("toxicity-bf16");
    session.check::<TextRankingPipeline>("ranking");
    session.check_onnx("ranking");
    for strategy in ["none", "simple", "first", "average", "max"] {
        session.check::<TokenClassificationPipeline>(&format!("entities-{strategy}"));
    }
    session.check::<FeatureExtractionPipeline>("features");
    session.check_onnx("features");
    session.check::<SentenceSimilarityPipeline>("similarity");
    session.check_onnx("similarity");
    session.check::<SentenceEmbeddings>("embeddings-sqrt");
    session.check::<FillMaskPipeline>("fill");
    session.check_onnx("fill");
    session.check_onnx("entities-none");
}
