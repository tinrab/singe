// Extractive question answering, as `transformers`' `*ForQuestionAnswering` heads: every token's hidden state to the scores of an answer starting and ending there, `[B, T, 2]`.
// It is written once for every model; its state is `span.output`.

use singe::{
    DataType, Error as SingeError, Result as SingeResult, Tensor,
    nn::{Layer, Linear, Model, ModelState, TokenBatch},
};

use super::{EncoderHead, HeadLayouts};
use crate::{assembly::PostNormEncoder, error::Result, schema::Schema, source::Source};

#[derive(Debug)]
pub struct QuestionAnswererLayers {
    encoder: Box<dyn Layer<TokenBatch>>,
    output: Linear,
}

impl Layer<TokenBatch> for QuestionAnswererLayers {
    fn forward(&self, input: &TokenBatch, state: &ModelState) -> SingeResult<Tensor> {
        self.output
            .forward(&self.encoder.forward(input, state)?, state)
    }
}

// A loaded question answerer, whose scores' last axis holds each token's start score, then its end score.
#[derive(Debug)]
pub struct QuestionAnswerer {
    model: Model<QuestionAnswererLayers>,
}

impl QuestionAnswerer {
    #[must_use]
    pub const fn model(&self) -> &Model<QuestionAnswererLayers> {
        &self.model
    }

    // Returns each row's tokens' start and end scores, padded tokens included.
    pub fn scores(&self, batch: &TokenBatch) -> Result<Vec<Vec<[f32; 2]>>> {
        let length = usize::try_from(batch.tokens().shape().dimensions()[1])
            .map_err(|_| SingeError::out_of_range("batch length"))?;
        let scores = self
            .model
            .forward_group(batch)?
            .cast(DataType::F32)?
            .to_vec::<f32>()?;
        Ok(scores
            .chunks(length * 2)
            .map(|row| row.chunks(2).map(|pair| [pair[0], pair[1]]).collect())
            .collect())
    }
}

impl EncoderHead for QuestionAnswerer {
    type Config = ();

    type Layers = QuestionAnswererLayers;

    fn transformers_config(_source: &Source) -> Result<()> {
        Ok(())
    }

    fn layers(
        schema: &mut Schema<'_>,
        encoder: PostNormEncoder,
        (): &(),
        _layouts: &HeadLayouts,
    ) -> SingeResult<QuestionAnswererLayers> {
        let output = schema
            .scope("span")
            .linear("output", encoder.config().width, 2, true)?;
        Ok(QuestionAnswererLayers {
            encoder: Box::new(encoder),
            output,
        })
    }

    fn new(model: Model<QuestionAnswererLayers>, (): ()) -> Self {
        Self { model }
    }
}
