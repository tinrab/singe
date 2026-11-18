use singe::{
    Result as SingeResult, Tensor,
    nn::{Layer, Model, ModelState, TokenBatch},
};

use super::{EncoderHead, HeadLayouts};
use crate::{assembly::PostNormEncoder, error::Result, schema::Schema, source::Source};

// A text encoder's layers, from a batch of token sequences to each token's last hidden state, `[B, T, W]`.
#[derive(Debug)]
pub struct TextEncoderLayers(pub Box<dyn Layer<TokenBatch>>);

impl Layer<TokenBatch> for TextEncoderLayers {
    fn forward(&self, input: &TokenBatch, state: &ModelState) -> SingeResult<Tensor> {
        self.0.forward(input, state)
    }
}

pub type TextEncoder = Model<TextEncoderLayers>;

// The bare encoder, whose snapshots' heads, if any, it leaves unread.
impl EncoderHead for TextEncoder {
    type Config = ();

    type Layers = TextEncoderLayers;

    fn transformers_config(_source: &Source) -> Result<()> {
        Ok(())
    }

    fn layers(
        _schema: &mut Schema<'_>,
        encoder: PostNormEncoder,
        (): &(),
        _layouts: &HeadLayouts,
    ) -> SingeResult<TextEncoderLayers> {
        Ok(TextEncoderLayers(Box::new(encoder)))
    }

    fn new(model: Self, (): ()) -> Self {
        model
    }
}
