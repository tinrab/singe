use singe::{
    Result as SingeResult,
    nn::{Layer, Model, TokenBatch},
};

use crate::{
    error::Result,
    head::{Boxed, EncoderBody, EncoderHead, HeadLayouts},
    schema::Schema,
    source::Source,
};

pub type TextEncoder = Model<Boxed<dyn Layer<TokenBatch>>>;

/// The bare encoder, which leaves unread any head a snapshot holds.
impl EncoderHead for TextEncoder {
    type Config = ();

    type Layers = Boxed<dyn Layer<TokenBatch>>;

    fn transformers_config(_source: &Source) -> Result<()> {
        Ok(())
    }

    fn layers(
        _schema: &mut Schema<'_>,
        encoder: EncoderBody,
        (): &(),
        _layouts: &HeadLayouts,
    ) -> SingeResult<Boxed<dyn Layer<TokenBatch>>> {
        Ok(Boxed(encoder.layers))
    }

    fn new(model: Self, (): ()) -> Self {
        model
    }
}
