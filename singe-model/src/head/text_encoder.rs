use singe::{
    Result as SingeResult,
    nn::{Layer, Model, TokenBatch},
};

use crate::{
    assembly::PostNormEncoder,
    error::Result,
    head::{Boxed, EncoderHead, HeadLayouts},
    schema::Schema,
    source::Source,
};

pub type TextEncoder = Model<Boxed<dyn Layer<TokenBatch>>>;

// The bare encoder, whose snapshots' heads, if any, it leaves unread.
impl EncoderHead for TextEncoder {
    type Config = ();

    type Layers = Boxed<dyn Layer<TokenBatch>>;

    fn transformers_config(_source: &Source) -> Result<()> {
        Ok(())
    }

    fn layers(
        _schema: &mut Schema<'_>,
        encoder: PostNormEncoder,
        (): &(),
        _layouts: &HeadLayouts,
    ) -> SingeResult<Boxed<dyn Layer<TokenBatch>>> {
        Ok(Boxed(Box::new(encoder)))
    }

    fn new(model: Self, (): ()) -> Self {
        model
    }
}
