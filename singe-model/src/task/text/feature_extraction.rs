//! `feature-extraction`: text to each token's last hidden state.

use std::sync::Arc;

use serde::Deserialize;
use singe::{
    DataType, Error as SingeError, Tensor,
    nn::{Layer, TokenBatch},
};

use crate::{
    error::Result,
    function::forward::{External, Forward},
    head::Boxed,
    model,
    processor::text::TextTokenizer,
    source::{LoadOptions, Source},
    task::{Pipeline, run_alone},
};

#[derive(Debug)]
pub struct FeatureExtractionPipeline {
    model: Forward<Boxed<dyn Layer<TokenBatch>>>,
    tokenizer: TextTokenizer,
    options: LoadOptions,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize)]
#[serde(default)]
pub struct FeatureExtractionParameters {
    /// The tokens each text is truncated to, keeping its special tokens, with no truncation for `None`.
    pub max_length: Option<usize>,
}

impl Pipeline for FeatureExtractionPipeline {
    type Input = String;

    type Parameters = FeatureExtractionParameters;

    /// Each real token's hidden state, `[T, W]`, with special tokens.
    type Output = Vec<Vec<f32>>;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        let model = match source.function() {
            Some(function) => Forward::External(External::create(
                Arc::clone(function),
                &["last_hidden_state"],
            )?),
            None => Forward::Native(model::load(source, options)?),
        };
        Ok(Self {
            model,
            tokenizer: TextTokenizer::load(source)?,
            options,
        })
    }

    fn run_one(&self, input: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        run_alone(self, input, parameters)
    }

    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>> {
        let texts = inputs.iter().map(String::as_str).collect::<Vec<_>>();
        let batch =
            self.tokenizer
                .encode_batch(&texts, parameters.max_length, self.options.device)?;
        let hidden = self.model.forward_group(&batch)?;
        rows(&hidden, &batch)
    }
}

/// Splits hidden states `[B, T, W]` into each row's real tokens.
fn rows(hidden: &Tensor, batch: &TokenBatch) -> Result<Vec<Vec<Vec<f32>>>> {
    let &[_, length, width] = hidden.shape().dimensions() else {
        return Err(SingeError::invalid_argument("hidden states, [B, T, W]").into());
    };
    let width = usize::try_from(width).map_err(|_| SingeError::out_of_range("feature width"))?;
    let length = usize::try_from(length).map_err(|_| SingeError::out_of_range("batch length"))?;
    let values = hidden.cast(DataType::F32)?.to_vec::<f32>()?;
    let counts = batch.lengths().to_vec::<i64>()?;
    counts
        .iter()
        .enumerate()
        .map(|(row, &count)| {
            let count =
                usize::try_from(count).map_err(|_| SingeError::out_of_range("token count"))?;
            Ok((0..count)
                .map(|token| {
                    let start = (row * length + token) * width;
                    values[start..start + width].to_vec()
                })
                .collect())
        })
        .collect()
}
