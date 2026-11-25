//! CLIP: a text tower and a vision tower of pre-norm blocks, each projected into one space where a text and an image compare by cosine; OWL-ViT builds on both towers.

pub(crate) mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Error as SingeError, Result as SingeResult, Tensor,
    nn::{
        Embedding, Layer, LayerNorm, Linear, Model, ModelState, PatchTokens, Pooling, PoolingInput,
        PoolingMode, StateKey, TokenBatch, TokenEmbeddings,
    },
};

use crate::{
    assembly::{
        FinalNorm, PreNormEncoder, PreNormEncoderConfig, VisionTowerConfig, patch_embedding,
    },
    error::Result,
    format::Loader,
    head::{Boxed, ContrastiveEncoder, HeadLoader, TextEncoder},
    model::ModelEntry,
    schema::Schema,
    source::{Format, LoadOptions, Source},
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipTextConfig {
    pub encoder: PreNormEncoderConfig,
    pub layers: i64,
    pub vocab_size: i64,
    pub max_positions: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipConfig {
    pub text: ClipTextConfig,
    pub vision: VisionTowerConfig,
    /// The width of the space both towers project into.
    pub projection: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipTextProjectionConfig {
    pub text: ClipTextConfig,
    pub projection: i64,
}

/// CLIP's text tower: token and learned position embeddings, causal pre-norm blocks, and a final norm, from token sequences to their last hidden states, `[B, T, W]`.
///
/// A sequence's embedding is its last token's, the end-of-text token's.
#[derive(Debug, Clone)]
pub struct ClipText {
    embeddings: TokenEmbeddings,
    encoder: PreNormEncoder,
}

impl ClipText {
    /// Registers the tower in `schema`: `embeddings` (`tokens`, `positions`), `blocks`, and `norm`.
    pub fn create(schema: &mut Schema<'_>, config: &ClipTextConfig) -> SingeResult<Self> {
        let width = config.encoder.width;
        let embeddings = {
            let mut schema = schema.scope("embeddings");
            TokenEmbeddings::new(Embedding::new(
                schema.parameter("tokens.weight", [config.vocab_size, width])?,
            ))
            .with_positions(schema.parameter("positions.weight", [config.max_positions, width])?)
        };
        let encoder =
            PreNormEncoder::create(schema, &config.encoder, config.layers, FinalNorm::Included)?;
        Ok(Self {
            embeddings,
            encoder,
        })
    }

    /// Returns each sequence's last token's final state, `[B, W]`.
    pub fn pooled(&self, input: &TokenBatch, state: &ModelState) -> SingeResult<Tensor> {
        let hidden = self.encoder.encode(
            &self.embeddings.forward(input, state)?,
            Some(&input.key_lengths()?),
            state,
        )?;
        Pooling::new(PoolingMode::LastToken, false).forward(
            &PoolingInput {
                hidden: &hidden,
                mask: &input.mask()?,
            },
            state,
        )
    }
}

impl Layer<TokenBatch> for ClipText {
    /// Returns every token's normed final state, `[B, T, W]`, as `CLIPTextModel`'s `last_hidden_state` is, which Stable Diffusion conditions on.
    fn forward(&self, input: &TokenBatch, state: &ModelState) -> SingeResult<Tensor> {
        self.encoder.encode(
            &self.embeddings.forward(input, state)?,
            Some(&input.key_lengths()?),
            state,
        )
    }
}

/// CLIP's vision tower: a class token and patches with learned positions, normed before pre-norm blocks, from images to every token's final state, `[B, 1 + N, W]`, before the `post_norm` a pooled or per-patch output takes.
#[derive(Debug, Clone)]
pub struct ClipVision {
    embeddings: PatchTokens,
    pre_norm: LayerNorm,
    encoder: PreNormEncoder,
    post_norm: LayerNorm,
}

impl ClipVision {
    /// Registers the tower in `schema`: `embeddings` (`patches`, `class`, `positions`), `pre_norm`, `blocks`, and `post_norm`.
    pub fn create(schema: &mut Schema<'_>, config: &VisionTowerConfig) -> SingeResult<Self> {
        let width = config.encoder.width;
        let embeddings = {
            let mut schema = schema.scope("embeddings");
            let patches = patch_embedding(
                &mut schema,
                config.channels,
                width,
                config.patch_size,
                false,
            )?;
            PatchTokens::new(patches)
                .with_leading_token(schema.parameter("class", [1, 1, width])?)
                .with_positions(schema.parameter("positions", [1, config.patches() + 1, width])?)
        };
        let pre_norm = schema.layer_norm("pre_norm", width, config.encoder.eps)?;
        let encoder =
            PreNormEncoder::create(schema, &config.encoder, config.layers, FinalNorm::Omitted)?;
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

/// CLIP's image side: from images to their projected embeddings, `[B, P]`, from the class token's normed final state.
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

/// CLIP's text side: from texts to their projected embeddings, `[N, P]`.
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
pub(crate) struct ImageTextLayers<I, T> {
    pub(crate) images: I,
    pub(crate) texts: T,
}

impl<I: Layer + Clone + 'static, T: Layer<TokenBatch> + Clone + 'static> ImageTextLayers<I, T> {
    pub(crate) fn split(&self, state: &ModelState) -> (Model<Boxed<dyn Layer>>, TextEncoder) {
        (
            Model::new(
                Boxed::<dyn Layer>(Box::new(self.images.clone())),
                state.clone(),
            ),
            Model::new(
                Boxed::<dyn Layer<TokenBatch>>(Box::new(self.texts.clone())),
                state.clone(),
            ),
        )
    }
}

#[derive(Debug, Clone)]
struct ClipLayers {
    sides: ImageTextLayers<ClipImageLayers, ClipTextLayers>,
    logit_scale: StateKey,
}

impl ClipLayers {
    fn create(schema: &mut Schema<'_>, config: &ClipConfig) -> SingeResult<Self> {
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
        Ok(Self {
            sides: ImageTextLayers {
                images: ClipImageLayers {
                    vision,
                    projection: image_projection,
                },
                texts: ClipTextLayers {
                    text,
                    projection: text_projection,
                },
            },
            logit_scale: schema.parameter("logit_scale", [])?,
        })
    }
}

/// The patches' final states, `[B, N, W]`, before the post-norm, as `CLIPVisionModel`'s `last_hidden_state` past its class token is.
#[derive(Debug, Clone)]
struct ClipPatchStates(ClipVision);

impl Layer for ClipPatchStates {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let hidden = self.0.hidden(input, state)?;
        let &[_, length, _] = hidden.shape().dimensions() else {
            return Err(SingeError::invalid_argument("the vision tower's states"));
        };
        hidden.narrow(1, 1, length - 1)
    }
}

/// CLIP's text tower alone, as `CLIPTextModel` stores it.
pub(crate) const TEXT_MODEL: ModelEntry = ModelEntry {
    name: "clip_text",
    formats: &[(Format::Transformers, &["clip_text_model"])],
    loaders: &[HeadLoader::TextEncoder(load_text)],
};

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "clip",
    formats: &[(Format::Transformers, &["clip"])],
    loaders: &[HeadLoader::ZeroShotImageClassification(load_image_text)],
};

pub fn load_text(loader: &mut Loader<'_>) -> Result<TextEncoder> {
    let config: ClipTextConfig = loader.config(transformers::text_model_config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        ClipText::create(schema, &config).map(|text| Boxed::<dyn Layer<TokenBatch>>(Box::new(text)))
    })?;
    loader.load_whole(&model, &transformers::text_rules("text_model", ""))
}

/// Loads a `CLIPTextModelWithProjection` from `source` as a model from texts to their end-of-text states, projected, as its `text_embeds` are.
pub fn load_text_with_projection(
    source: &Source,
    options: LoadOptions,
) -> Result<Model<Boxed<dyn Layer<TokenBatch>>>> {
    let mut loader = Loader::new(source, options);
    let config: ClipTextProjectionConfig =
        loader.config(transformers::text_projection_model_config)?;
    let model = Schema::model(options.data_type, |schema| {
        let layers = ClipTextLayers {
            text: ClipText::create(&mut schema.scope("text"), &config.text)?,
            projection: schema.linear(
                "projection",
                config.text.encoder.width,
                config.projection,
                false,
            )?,
        };
        Ok::<_, SingeError>(Boxed::<dyn Layer<TokenBatch>>(Box::new(layers)))
    })?;
    loader.load_whole(
        &model,
        &transformers::text_rules("text_model", "text").rename("text_projection", "projection"),
    )
}

/// Loads a `CLIPVisionModel` from `source` as a model from images to their patches' final states.
pub fn load_vision_states(
    source: &Source,
    options: LoadOptions,
) -> Result<Model<Boxed<dyn Layer>>> {
    let mut loader = Loader::new(source, options);
    let config: VisionTowerConfig = loader.config(transformers::vision_model_config)?;
    let model = Schema::model(options.data_type, |schema| {
        ClipVision::create(&mut schema.scope("vision"), &config)
            .map(|vision| Boxed::<dyn Layer>(Box::new(ClipPatchStates(vision))))
    })?;
    loader.load_whole(
        &model,
        &transformers::vision_rules("vision_model", "vision"),
    )
}

pub fn load_image_text(loader: &mut Loader<'_>) -> Result<ContrastiveEncoder> {
    let config: ClipConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        ClipLayers::create(schema, &config)
    })?;
    let model = loader.load_whole(&model, &transformers::clip_rules(""))?;
    let layers = model.layer();
    let (images, texts) = layers.sides.split(model.state());
    Ok(ContrastiveEncoder::new(
        images,
        texts,
        layers.logit_scale.clone(),
    ))
}
