// `text-ranking`: a query and texts to each text's relevance, best first, as `sentence-transformers`' `CrossEncoder.rank` ranks them.
// The model is a cross-encoder: a sequence classification head of one label over the query and each text as a pair.

use serde::{Deserialize, Serialize};

use crate::{
    error::{Error, Result},
    head::{ScoreFunction, SequenceClassifier},
    model,
    processor::text::{EncodeOptions, TextInput, TextTokenizer},
    source::{LoadOptions, Source},
    task::{Pipeline, Task},
};

#[derive(Debug)]
pub struct TextRankingPipeline {
    model: SequenceClassifier,
    tokenizer: TextTokenizer,
    // The function a score goes through unless a caller names another.
    function: ScoreFunction,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct TextRankingInput {
    pub query: String,
    pub texts: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(default)]
pub struct TextRankingParameters {
    // Keeps the best this many texts, or all when absent.
    pub top_k: Option<usize>,
    pub function_to_apply: Option<ScoreFunction>,
    // Truncates each pair to this many tokens, cutting the longer text first.
    pub max_length: Option<usize>,
}

// A text's place among the input's texts, and its score.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RankedText {
    pub corpus_id: usize,
    pub score: f32,
}

impl Pipeline for TextRankingPipeline {
    const TASK: Task = Task::TextRanking;

    type Input = TextRankingInput;

    type Parameters = TextRankingParameters;

    type Output = Vec<RankedText>;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        let model: SequenceClassifier = model::load(source, options)?;
        if model.config().labels.len() != 1 {
            return Err(Error::unsupported(
                "ranking with a classifier of other than one label",
            ));
        }

        // A cross-encoder's scores go through the function its config names, or its classifier's, a sigmoid for one label, as `sentence-transformers` chooses.
        let config = model.config();
        let function = config.ranking_function.unwrap_or(config.function);
        Ok(Self {
            function,
            model,
            tokenizer: TextTokenizer::load(source.directory())?,
        })
    }

    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>> {
        let function = parameters.function_to_apply.unwrap_or(self.function);
        let device = self.model.model().state().device()?;
        inputs
            .iter()
            .map(|input| {
                if input.texts.is_empty() {
                    return Ok(Vec::new());
                }

                let pairs = input
                    .texts
                    .iter()
                    .map(|text| TextInput::Pair(&input.query, text))
                    .collect::<Vec<_>>();
                let options = EncodeOptions {
                    max_length: parameters.max_length,
                    ..EncodeOptions::default()
                };
                let encoded = self.tokenizer.encode_inputs(&pairs, &options, device)?;
                let scores = self.model.scores(&encoded.batch)?;
                let mut ranked = scores
                    .iter()
                    .flat_map(|row| function.apply(row))
                    .enumerate()
                    .map(|(corpus_id, score)| RankedText { corpus_id, score })
                    .collect::<Vec<_>>();
                ranked.sort_by(|left, right| right.score.total_cmp(&left.score));
                ranked.truncate(parameters.top_k.unwrap_or(usize::MAX));
                Ok(ranked)
            })
            .collect()
    }
}
