// CLIP: a text tower and a vision tower of pre-norm blocks, each projected into one space where a text and an image compare by cosine; OWL-ViT builds on both towers.

pub(crate) mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult, Tensor,
    nn::{
        Conv2dConfig, Embedding, Layer, LayerNorm, Linear, Model, ModelState, PatchEmbedding,
        PatchTokens, Pooling, PoolingInput, PoolingMode, StateBuilder, TokenBatch, TokenEmbeddings,
    },
};

use crate::{
    assembly::{PreNormEncoder, PreNormEncoderSpec},
    error::Result,
    format::Loader,
    head::{
        ContrastiveEncoder, HeadLoader, PixelLayers, TextEmbeddingLayers, TextEncoder,
        TextEncoderLayers,
    },
    model::ModelEntry,
    schema::Schema,
    source::{Format, LoadOptions, Source},
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipTextConfig {
    pub encoder: PreNormEncoderSpec,
    pub layers: i64,
    pub vocab_size: i64,
    pub positions: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipVisionConfig {
    pub encoder: PreNormEncoderSpec,
    pub layers: i64,
    pub image_size: i64,
    pub patch_size: i64,
    pub channels: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipConfig {
    pub text: ClipTextConfig,
    pub vision: ClipVisionConfig,
    // The width of the space both towers project into.
    pub projection: i64,
}

// CLIP's text tower: token and learned position embeddings, causal pre-norm blocks, and a final norm, from token sequences to their last hidden states, `[B, T, W]`; a sequence's embedding is its last token's, the end-of-text token's.
#[derive(Debug, Clone)]
pub struct ClipText {
    embeddings: TokenEmbeddings,
    encoder: PreNormEncoder,
}

impl ClipText {
    // Registers the tower in `schema`: `embeddings` (`tokens`, `positions`), `blocks`, and `norm`.
    pub fn create(schema: &mut Schema<'_>, config: &ClipTextConfig) -> SingeResult<Self> {
        let width = config.encoder.width;
        let embeddings = {
            let mut schema = schema.scope("embeddings");
            TokenEmbeddings::new(Embedding::new(
                schema.parameter("tokens.weight", [config.vocab_size, width])?,
            ))
            .with_positions(schema.parameter("positions.weight", [config.positions, width])?)
        };
        let encoder = PreNormEncoder::create(schema, &config.encoder, config.layers, true)?;
        Ok(Self {
            embeddings,
            encoder,
        })
    }

    // Each sequence's last token's final state, `[B, W]`.
    pub fn pooled(&self, input: &TokenBatch, state: &ModelState) -> SingeResult<Tensor> {
        let hidden = self.encoder.encode(
            &self.embeddings.forward(input, state)?,
            Some(&input.key_lengths()?),
            state,
        )?;
        Pooling::new(PoolingMode::LastToken, false).forward(
            &PoolingInput {
                hidden: &hidden,
                padding: input.padding(),
            },
            state,
        )
    }
}

// CLIP's vision tower: a class token and patches with learned positions, normed before pre-norm blocks, from images to every token's final state, `[B, 1 + N, W]`, before the `post_norm` a pooled or per-patch output takes.
#[derive(Debug, Clone)]
pub struct ClipVision {
    embeddings: PatchTokens,
    pre_norm: LayerNorm,
    encoder: PreNormEncoder,
    post_norm: LayerNorm,
}

impl ClipVision {
    // Registers the tower in `schema`: `embeddings` (`patches`, `class`, `positions`), `pre_norm`, `blocks`, and `post_norm`.
    pub fn create(schema: &mut Schema<'_>, config: &ClipVisionConfig) -> SingeResult<Self> {
        let width = config.encoder.width;
        let side = config.image_size / config.patch_size;
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
                false,
            )?;
            PatchTokens::new(PatchEmbedding::new(patches))
                .with_class_token(schema.parameter("class", [1, 1, width])?)
                .with_positions(schema.parameter("positions", [1, side * side + 1, width])?)
        };
        let pre_norm = schema.layer_norm("pre_norm", width, config.encoder.eps)?;
        let encoder = PreNormEncoder::create(schema, &config.encoder, config.layers, false)?;
        let post_norm = schema.layer_norm("post_norm", width, config.encoder.eps)?;
        Ok(Self {
            embeddings,
            pre_norm,
            encoder,
            post_norm,
        })
    }

    pub fn hidden(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let tokens = input
            .apply(&self.embeddings, state)?
            .apply(&self.pre_norm, state)?;
        self.encoder.forward(&tokens, state)
    }

    #[must_use]
    pub const fn post_norm(&self) -> &LayerNorm {
        &self.post_norm
    }
}

// CLIP's images to their projected embeddings, `[B, P]`, from the class token's normed final state.
#[derive(Debug, Clone)]
struct ClipImageLayers {
    vision: ClipVision,
    projection: Linear,
}

impl Layer for ClipImageLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        self.vision
            .hidden(input, state)?
            .select(1, 0)?
            .apply(self.vision.post_norm(), state)?
            .apply(&self.projection, state)
    }
}

// CLIP's texts to their projected embeddings, `[N, P]`.
#[derive(Debug, Clone)]
pub(crate) struct ClipTextLayers {
    pub(crate) text: ClipText,
    pub(crate) projection: Linear,
}

impl Layer<TokenBatch> for ClipTextLayers {
    fn forward(&self, input: &TokenBatch, state: &ModelState) -> SingeResult<Tensor> {
        self.text
            .pooled(input, state)?
            .apply(&self.projection, state)
    }
}

#[derive(Debug, Clone)]
struct ClipLayers {
    images: ClipImageLayers,
    texts: ClipTextLayers,
}

// Every token's normed final state, `[B, T, W]`, as `CLIPTextModel`'s `last_hidden_state` is, which Stable Diffusion conditions on.
impl Layer<TokenBatch> for ClipText {
    fn forward(&self, input: &TokenBatch, state: &ModelState) -> SingeResult<Tensor> {
        self.encoder.encode(
            &self.embeddings.forward(input, state)?,
            Some(&input.key_lengths()?),
            state,
        )
    }
}

// CLIP's text tower alone, as `CLIPTextModel` stores it.
pub(crate) const TEXT_MODEL: ModelEntry = ModelEntry {
    name: "clip_text",
    formats: &[(Format::Transformers, &["clip_text_model"])],
    heads: &[HeadLoader::TextEncoder(load_text)],
};

pub fn load_text(loader: &mut Loader<'_>) -> Result<TextEncoder> {
    let config: ClipTextConfig = loader.config(transformers::text_model_config)?;
    let mut builder = StateBuilder::new();
    let text = {
        let mut schema = Schema::root(&mut builder, loader.options().data_type);
        ClipText::create(&mut schema, &config)?
    };
    let model = loader.load(
        "model",
        &Model::new(TextEncoderLayers(Box::new(text)), builder.finish()?),
        &transformers::text_rules("text_model", ""),
    )?;
    loader.finish()?;
    Ok(model)
}

// Loads a `CLIPTextModelWithProjection` from `source`: each text's end-of-text state projected, as its `text_embeds` are.
pub fn load_text_with_projection(
    source: &Source,
    options: LoadOptions,
) -> Result<Model<TextEmbeddingLayers>> {
    let mut loader = Loader::new(source, options);
    let config: ClipTextConfig = loader.config(transformers::text_model_config)?;
    let projection: transformers::ProjectionFields = source.json("config.json")?;
    let mut builder = StateBuilder::new();
    let layers = {
        let mut schema = Schema::root(&mut builder, options.data_type);
        ClipTextLayers {
            text: ClipText::create(&mut schema.scope("text"), &config)?,
            projection: schema.linear(
                "projection",
                config.encoder.width,
                projection.projection_dim,
                false,
            )?,
        }
    };
    let model = loader.load(
        "model",
        &Model::new(TextEmbeddingLayers(Box::new(layers)), builder.finish()?),
        &transformers::text_rules("text_model", "text.").rename("text_projection", "projection"),
    )?;
    loader.finish()?;
    Ok(model)
}

// The patches' final states, `[B, N, W]`, before the post-norm, as `CLIPVisionModel`'s `last_hidden_state` past its class token is.
#[derive(Debug, Clone)]
struct ClipPatchStates(ClipVision);

impl Layer for ClipPatchStates {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let hidden = self.0.hidden(input, state)?;
        let length = hidden.shape().dimensions()[1];
        hidden.narrow(1, 1, length - 1)
    }
}

// Loads a `CLIPVisionModel` from `source` as its patches' final states.
pub fn load_vision_states(source: &Source, options: LoadOptions) -> Result<Model<PixelLayers>> {
    let mut loader = Loader::new(source, options);
    let config: ClipVisionConfig = loader.config(transformers::vision_model_config)?;
    let mut builder = StateBuilder::new();
    let vision = {
        let mut schema = Schema::root(&mut builder, options.data_type);
        ClipVision::create(&mut schema.scope("vision"), &config)?
    };
    let model = loader.load(
        "model",
        &Model::new(
            PixelLayers(Box::new(ClipPatchStates(vision))),
            builder.finish()?,
        ),
        &transformers::vision_rules("vision_model", "vision."),
    )?;
    loader.finish()?;
    Ok(model)
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "clip",
    formats: &[(Format::Transformers, &["clip"])],
    heads: &[HeadLoader::ZeroShotImageClassification(load_image_text)],
};

pub fn load_image_text(loader: &mut Loader<'_>) -> Result<ContrastiveEncoder> {
    let config: ClipConfig = loader.config(transformers::config)?;
    let mut builder = StateBuilder::new();
    let (layers, logit_scale) = {
        let mut schema = Schema::root(&mut builder, loader.options().data_type);
        let vision = ClipVision::create(&mut schema.scope("vision"), &config.vision)?;
        let text = ClipText::create(&mut schema.scope("text"), &config.text)?;
        let image_projection = schema.linear(
            "image_projection",
            config.vision.encoder.width,
            config.projection,
            false,
        )?;
        let text_projection = schema.linear(
            "text_projection",
            config.text.encoder.width,
            config.projection,
            false,
        )?;
        let logit_scale = schema.parameter("logit_scale", [])?;
        (
            ClipLayers {
                images: ClipImageLayers {
                    vision,
                    projection: image_projection,
                },
                texts: ClipTextLayers {
                    text,
                    projection: text_projection,
                },
            },
            logit_scale,
        )
    };
    let schema = Model::new(layers, builder.finish()?);
    let model = loader.load("model", &schema, &transformers::clip_rules(""))?;
    loader.finish()?;
    let state = model.state().clone();
    Ok(ContrastiveEncoder::new(
        Model::new(
            PixelLayers(Box::new(model.layer().images.clone())),
            state.clone(),
        ),
        Model::new(
            TextEmbeddingLayers(Box::new(model.layer().texts.clone())),
            state,
        ),
        logit_scale,
    ))
}
