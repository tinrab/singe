// `feature-extraction`: text to each token's last hidden state.

use serde::Deserialize;
use singe::{DataType, Tensor, nn::TokenBatch};

use crate::{
    error::{Error, Result},
    head::TextEncoder,
    model,
    processing::text::TextTokenizer,
    source::{LoadOptions, Source},
    task::{Pipeline, Task},
};

#[derive(Debug)]
pub struct FeatureExtractionPipeline {
    model: TextEncoder,
    tokenizer: TextTokenizer,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize)]
#[serde(default)]
pub struct FeatureExtractionParameters {
    // Truncates each text to this many tokens, keeping its special tokens.
    pub max_length: Option<usize>,
}

impl FeatureExtractionPipeline {
    #[must_use]
    pub const fn model(&self) -> &TextEncoder {
        &self.model
    }
}

impl Pipeline for FeatureExtractionPipeline {
    const TASK: Task = Task::FeatureExtraction;

    type Input = String;

    type Parameters = FeatureExtractionParameters;

    // Each real token's hidden state, `[T, W]`, with special tokens.
    type Output = Vec<Vec<f32>>;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            tokenizer: TextTokenizer::load(source.directory())?,
        })
    }

    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>> {
        let texts = inputs.iter().map(String::as_str).collect::<Vec<_>>();
        let device = self.model.state().device()?;
        let batch = self
            .tokenizer
            .encode_batch(&texts, parameters.max_length, device)?;
        let hidden = self.model.forward_group(&batch)?;
        rows(&hidden, &batch)
    }
}

// Splits hidden states `[B, T, W]` into each row's real tokens.
fn rows(hidden: &Tensor, batch: &TokenBatch) -> Result<Vec<Vec<Vec<f32>>>> {
    let dimensions = hidden.shape().dimensions();
    let width = usize::try_from(dimensions[2]).map_err(|_| Error::unsupported("this width"))?;
    let length = usize::try_from(dimensions[1]).map_err(|_| Error::unsupported("this length"))?;
    let values = hidden.cast(DataType::F32)?.to_vec::<f32>()?;
    let counts = batch.key_lengths()?.cast(DataType::I64)?.to_vec::<i64>()?;
    counts
        .iter()
        .enumerate()
        .map(|(row, &count)| {
            let count = usize::try_from(count).map_err(|_| Error::unsupported("this count"))?;
            Ok((0..count)
                .map(|token| {
                    let start = (row * length + token) * width;
                    values[start..start + width].to_vec()
                })
                .collect())
        })
        .collect()
}
