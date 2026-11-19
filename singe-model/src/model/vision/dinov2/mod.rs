// DINOv2: a vision transformer whose class token embeds the image, which image classifiers and Depth Anything build on.

pub(crate) mod transformers;

use singe::{
    Result as SingeResult, Tensor,
    nn::{Layer, ModelState, PatchTokens},
};

use crate::{
    assembly::{FinalNorm, PreNormEncoder, VisionTowerConfig, patch_embedding},
    error::Result,
    format::Loader,
    head::{Boxed, HeadLoader, ImageEncoder, PooledLayer},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

// DINOv2: a class token and patches at the trained image size, pre-norm blocks with layer scale, and a final norm.
// It returns every token's final hidden state, `[B, 1 + N, D]`, of which the class token's is the image's embedding.
#[derive(Debug, Clone)]
pub struct Dinov2Layers {
    embeddings: PatchTokens,
    encoder: PreNormEncoder,
}

impl Dinov2Layers {
    // Registers the encoder in `schema`: `embeddings` (`patches`, `class`, `positions`), `blocks`, and `norm`.
    pub fn create(schema: &mut Schema<'_>, config: &VisionTowerConfig) -> SingeResult<Self> {
        let width = config.encoder.width;
        let embeddings = {
            let mut schema = schema.scope("embeddings");
            let patches =
                patch_embedding(&mut schema, config.channels, width, config.patch_size, true)?;
            PatchTokens::new(patches)
                .with_leading_token(schema.parameter("class", [1, 1, width])?)
                .with_positions(schema.parameter("positions", [1, config.patches() + 1, width])?)
                .with_resized_positions(config.patch_size)
        };
        let encoder =
            PreNormEncoder::create(schema, &config.encoder, config.layers, FinalNorm::Included)?;
        Ok(Self {
            embeddings,
            encoder,
        })
    }

    // The final norm of every token's state after each block of `stages`, counted from one, `[B, 1 + N, D]` each, as `Dinov2Backbone` gives its feature maps.
    pub fn features(
        &self,
        input: &Tensor,
        stages: &[usize],
        state: &ModelState,
    ) -> SingeResult<Vec<Tensor>> {
        self.encoder
            .features(&input.apply(&self.embeddings, state)?, stages, state)
    }
}

impl Layer for Dinov2Layers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        self.encoder
            .forward(&input.apply(&self.embeddings, state)?, state)
    }
}

impl PooledLayer for Dinov2Layers {
    // Each image's class token, `[B, D]`, which `transformers` returns as the pooled output.
    fn pooled(&self, hidden: &Tensor) -> SingeResult<Tensor> {
        hidden.narrow(1, 0, 1)?.squeeze(1)
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "dinov2",
    formats: &[(Format::Transformers, &["dinov2"])],
    loaders: &[HeadLoader::ImageEncoder(load_encoder)],
};

pub fn load_encoder(loader: &mut Loader<'_>) -> Result<ImageEncoder> {
    let config: VisionTowerConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        Dinov2Layers::create(schema, &config)
            .map(|layers| Boxed::<dyn PooledLayer>(Box::new(layers)))
    })?;
    loader.load_whole(&model, &transformers::rules())
}
