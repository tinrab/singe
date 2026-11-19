// Extractive question answering, as `transformers`' `*ForQuestionAnswering` heads: every token's hidden state to the scores of an answer starting and ending there, `[B, T, 2]`.
// It is written once for every model; its state is `span.output`.

use singe::{
    DataType, Error as SingeError, Result as SingeResult,
    nn::{Model, TokenBatch},
};

use crate::{
    assembly::PostNormEncoder,
    error::Result,
    head::{EncoderHead, HeadLayouts, TokenScoreLayers},
    schema::Schema,
    source::Source,
};

// A loaded question answerer, whose scores' last axis holds each token's start score, then its end score.
#[derive(Debug)]
pub struct QuestionAnswerer {
    model: Model<TokenScoreLayers>,
}

impl QuestionAnswerer {
    #[must_use]
    pub const fn model(&self) -> &Model<TokenScoreLayers> {
        &self.model
    }

    // Returns each row's tokens' start and end scores, padded tokens included.
    pub fn scores(&self, batch: &TokenBatch) -> Result<Vec<Vec<[f32; 2]>>> {
        let &[_, length] = batch.tokens().shape().dimensions() else {
            return Err(SingeError::invalid_argument("token batch, [B, T]").into());
        };
        let length =
            usize::try_from(length).map_err(|_| SingeError::out_of_range("batch length"))?;
        let scores = self
            .model
            .forward_group(batch)?
            .cast(DataType::F32)?
            .to_vec::<f32>()?;
        Ok(scores
            .chunks(length * 2)
            .map(|row| row.as_chunks::<2>().0.to_vec())
            .collect())
    }
}

impl EncoderHead for QuestionAnswerer {
    type Config = ();

    type Layers = TokenScoreLayers;

    fn transformers_config(_source: &Source) -> Result<()> {
        Ok(())
    }

    fn layers(
        schema: &mut Schema<'_>,
        encoder: PostNormEncoder,
        (): &(),
        _layouts: &HeadLayouts,
    ) -> SingeResult<TokenScoreLayers> {
        TokenScoreLayers::create(schema, "span", encoder, 2)
    }

    fn new(model: Model<TokenScoreLayers>, (): ()) -> Self {
        Self { model }
    }
}
