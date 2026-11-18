// DINOv2: a vision transformer whose class token embeds the image, which image classifiers and Depth Anything build on.

pub(crate) mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Result as SingeResult, Tensor,
    nn::{
        Activation, Conv2dConfig, DenseAttentionInput, Layer, LayerNorm, LayerScale, Mlp, Model,
        ModelState, PatchEmbedding, PatchTokens, PreNormBlock, SelfAttention, StateBuilder,
    },
};

use crate::{
    error::Result,
    format::Loader,
    head::{HeadLoader, ImageEncoderLayers, PooledLayer},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Dinov2Config {
    pub width: i64,
    pub layers: i64,
    pub heads: i64,
    pub inner: i64,
    pub activation: Activation,
    pub eps: f32,
    // The square image side the position embeddings are trained for, a whole number of patches.
    pub image_size: i64,
    pub patch_size: i64,
    pub channels: i64,
    // Whether the queries, keys, and values have biases; the output projection always does.
    pub attention_bias: bool,
}

impl Dinov2Config {
    // The class token and one per patch.
    const fn tokens(&self) -> i64 {
        let side = self.image_size / self.patch_size;
        side * side + 1
    }
}

type Block = PreNormBlock<SelfAttention, Mlp, LayerNorm>;

// DINOv2: a class token and patches at the trained image size, pre-norm blocks with layer scale, and a final norm.
// It returns every token's final hidden state, `[B, 1 + N, D]`, of which the class token's is the image's embedding.
#[derive(Debug, Clone)]
pub struct Dinov2Layers {
    embeddings: PatchTokens,
    blocks: Vec<Block>,
    norm: LayerNorm,
}

pub type Dinov2 = Model<Dinov2Layers>;

impl Layer for Dinov2Layers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let mut hidden = input.apply(&self.embeddings, state)?;
        for block in &self.blocks {
            hidden = block.forward(
                &DenseAttentionInput {
                    hidden: &hidden,
                    angles: None,
                    key_lengths: None,
                },
                state,
            )?;
        }
        hidden.apply(&self.norm, state)
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
    heads: &[HeadLoader::ImageEncoder(|loader| {
        let model = load(loader)?;
        Ok(Model::new(
            ImageEncoderLayers(Box::new(model.layer().clone())),
            model.state().clone(),
        ))
    })],
};

fn block(schema: &mut Schema<'_>, config: &Dinov2Config) -> SingeResult<Block> {
    let width = config.width;
    let eps = config.eps;
    let attention_norm = schema.layer_norm("attention_norm", width, eps)?;
    let attention = {
        let mut schema = schema.scope("attention");
        SelfAttention::create(
            schema.linear("query", width, width, config.attention_bias)?,
            schema.linear("key", width, width, config.attention_bias)?,
            schema.linear("value", width, width, config.attention_bias)?,
            schema.linear("output", width, width, true)?,
            config.heads,
            width,
        )?
    };
    let attention_scale = LayerScale::new(schema.parameter("attention_scale.scale", [width])?);
    let feed_forward_norm = schema.layer_norm("feed_forward_norm", width, eps)?;
    let feed_forward = {
        let mut schema = schema.scope("feed_forward");
        Mlp::new(
            schema.linear("up", width, config.inner, true)?,
            schema.linear("down", config.inner, width, true)?,
            config.activation,
        )
    };
    let feed_forward_scale =
        LayerScale::new(schema.parameter("feed_forward_scale.scale", [width])?);
    Ok(
        PreNormBlock::new(attention_norm, attention, feed_forward_norm, feed_forward)
            .with_layer_scales(attention_scale, feed_forward_scale),
    )
}

impl Dinov2Layers {
    // Registers the encoder in `schema`: `embeddings` (`patches`, `class`, `positions`), `blocks`, and `norm`.
    pub fn create(schema: &mut Schema<'_>, config: &Dinov2Config) -> SingeResult<Self> {
        let width = config.width;
        let embeddings = {
            let mut schema = schema.scope("embeddings");
            let patches = schema.conv2d(
                "patches",
                config.channels,
                width,
                config.patch_size,
                Conv2dConfig {
                    stride: (config.patch_size, config.patch_size),
                    ..Conv2dConfig::default()
                },
                true,
            )?;
            PatchTokens::new(PatchEmbedding::new(patches))
                .with_class_token(schema.parameter("class", [1, 1, width])?)
                .with_positions(schema.parameter("positions", [1, config.tokens(), width])?)
                .with_resized_positions(config.patch_size)
        };
        let blocks = (0..config.layers)
            .map(|layer| block(&mut schema.scope(format!("blocks.{layer}")), config))
            .collect::<SingeResult<Vec<_>>>()?;
        let norm = schema.layer_norm("norm", width, config.eps)?;
        Ok(Self {
            embeddings,
            blocks,
            norm,
        })
    }

    // The final norm of every token's state after each block of `stages`, counted from one, `[B, 1 + N, D]` each, as `Dinov2Backbone` gives its feature maps.
    pub fn features(
        &self,
        input: &Tensor,
        stages: &[usize],
        state: &ModelState,
    ) -> SingeResult<Vec<Tensor>> {
        let mut hidden = input.apply(&self.embeddings, state)?;
        let mut features = Vec::with_capacity(stages.len());
        for (index, block) in self.blocks.iter().enumerate() {
            hidden = block.forward(
                &DenseAttentionInput {
                    hidden: &hidden,
                    angles: None,
                    key_lengths: None,
                },
                state,
            )?;
            if stages.contains(&(index + 1)) {
                features.push(hidden.apply(&self.norm, state)?);
            }
        }
        Ok(features)
    }
}

// Builds the encoder on `Device::Meta`.
fn schema(config: &Dinov2Config, data_type: DataType) -> Result<Dinov2> {
    let mut builder = StateBuilder::new();
    let layers = Dinov2Layers::create(&mut Schema::root(&mut builder, data_type), config)?;
    Ok(Model::new(layers, builder.finish()?))
}

pub fn load(loader: &mut Loader<'_>) -> Result<Dinov2> {
    let config: Dinov2Config = loader.config(transformers::config)?;
    let schema = schema(&config, loader.options().data_type)?;
    let model = loader.load("model", &schema, &transformers::rules())?;
    loader.finish()?;
    Ok(model)
}
