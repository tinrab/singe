//! `SigLIP`: a vision tower of patches with learned positions, pooled by attention from a learned probe, and a text tower pooled at its last position, compared by scaled cosines plus a bias whose sigmoid scores each text alone.
//! `SigLIP` 2's `NaFlex` towers project each patch's pixels with a linear layer and resize their grid of positions to each image's patches.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Error as SingeError, Result as SingeResult, Shape, Tensor,
    nn::{Embedding, Layer, LayerNorm, Linear, Mlp, ModelState, PatchTokens, StateKey, TokenBatch},
};

use crate::{
    assembly::{
        Attention, AttentionSpec, FinalNorm, PreNormEncoder, PreNormEncoderConfig,
        VisionTowerConfig, patch_embedding,
    },
    error::Result,
    format::Loader,
    head::{ContrastiveEncoder, HeadLoader},
    model::{ModelEntry, multimodal::clip::ImageTextLayers},
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SiglipTextConfig {
    pub encoder: PreNormEncoderConfig,
    pub layers: i64,
    pub vocab_size: i64,
    /// The positions every text is padded to and read at, the last of which it is pooled at.
    pub max_positions: i64,
    pub pad: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SiglipConfig {
    pub text: SiglipTextConfig,
    pub vision: VisionTowerConfig,
    /// The side of `NaFlex`'s grid of learned positions, which resizes to each image's patches, or `None` for a convolution over the square image the positions were learned for.
    pub flexible_grid: Option<i64>,
}

/// How the vision tower embeds patches and their positions.
#[derive(Debug, Clone)]
enum PatchInput {
    Convolution(PatchTokens),
    /// Each patch's pixels, in row, column, channel order, through `projection`, with the `grid × grid` positions resized to the image's patches.
    Flexible {
        projection: Linear,
        positions: StateKey,
        grid: i64,
        patch: i64,
    },
}

impl PatchInput {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let (projection, positions, grid, patch) = match self {
            Self::Convolution(tokens) => return input.apply(tokens, state),
            Self::Flexible {
                projection,
                positions,
                grid,
                patch,
            } => (projection, positions, *grid, *patch),
        };
        let &[batch, channels, height, width] = input.shape().dimensions() else {
            return Err(SingeError::invalid_argument(
                "SigLIP 2 pixels, [B, 3, H, W]",
            ));
        };
        let (rows, columns) = (height / patch, width / patch);
        let patches = input
            .reshape([batch, channels, rows, patch, columns, patch])?
            .permute(&[0, 2, 4, 3, 5, 1])?
            .reshape([batch, rows * columns, patch * patch * channels])?
            .apply(projection, state)?;
        let (row_weights, column_weights) = (axis_weights(rows, grid), axis_weights(columns, grid));
        let weights = row_weights
            .iter()
            .flat_map(|row| {
                column_weights.iter().flat_map(move |column| {
                    row.iter()
                        .flat_map(move |&down| column.iter().map(move |&across| down * across))
                })
            })
            .collect::<Vec<_>>();
        let weights = Tensor::from_vec(
            input.device(),
            Shape::create([rows * columns, grid * grid])?,
            weights,
        )?;
        let resized = weights
            .matmul(&state.tensor(positions)?.cast(DataType::F32)?)?
            .cast(patches.data_type())?;
        patches.add(&resized.unsqueeze(0)?)
    }
}

/// Returns the weights, `[outputs, grid]`, that resize `grid` positions to `outputs` along one axis, as `resize_positional_embeddings` writes antialiased bilinear interpolation, in `F32`.
fn axis_weights(outputs: i64, grid: i64) -> Vec<Vec<f32>> {
    let scale = grid as f32 / outputs as f32;
    let support = scale.max(1.0);
    (0..outputs)
        .map(|output| {
            let center = (output as f32 + 0.5) * scale - 0.5;
            let weights = (0..grid)
                .map(|source| (1.0 - (source as f32 - center).abs() / support).max(0.0))
                .collect::<Vec<_>>();
            let total = weights.iter().sum::<f32>();
            weights.into_iter().map(|weight| weight / total).collect()
        })
        .collect()
}

/// `SigLIP`'s attention pooling: a learned probe attends to every patch, then a feed-forward is added to the normed result.
#[derive(Debug, Clone)]
struct AttentionPooling {
    probe: StateKey,
    attention: Attention,
    norm: LayerNorm,
    feed_forward: Mlp,
}

impl AttentionPooling {
    fn create(schema: &mut Schema<'_>, encoder: &PreNormEncoderConfig) -> SingeResult<Self> {
        let width = encoder.width;
        Ok(Self {
            probe: schema.parameter("probe", [1, 1, width])?,
            attention: Attention::create(
                &mut schema.scope("attention"),
                AttentionSpec::square(width, encoder.heads),
            )?,
            norm: schema.layer_norm("norm", width, encoder.eps)?,
            feed_forward: schema.mlp(
                "feed_forward",
                [width, encoder.inner, width],
                encoder.activation,
                true,
            )?,
        })
    }
}

impl Layer for AttentionPooling {
    /// Returns each image's pooled state, `[B, W]`, from its patches' states, `[B, N, W]`.
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let &[batch, _, width] = input.shape().dimensions() else {
            return Err(SingeError::invalid_argument(
                "SigLIP patch states, [B, N, W]",
            ));
        };
        let probe = state
            .tensor(&self.probe)?
            .expand(Shape::create([batch, 1, width])?)?;
        let pooled = self.attention.forward(&probe, input, input, state)?;
        pooled
            .add(
                &pooled
                    .apply(&self.norm, state)?
                    .apply(&self.feed_forward, state)?,
            )?
            .select(1, 0)
    }
}

/// `SigLIP`'s vision tower and its pooling: from images to their embeddings, `[B, W]`.
#[derive(Debug, Clone)]
struct SiglipImageLayers {
    embeddings: PatchInput,
    encoder: PreNormEncoder,
    post_norm: LayerNorm,
    pooling: AttentionPooling,
}

impl SiglipImageLayers {
    fn create(
        schema: &mut Schema<'_>,
        config: &VisionTowerConfig,
        flexible_grid: Option<i64>,
    ) -> SingeResult<Self> {
        let width = config.encoder.width;
        let embeddings = {
            let mut schema = schema.scope("embeddings");
            let patch = config.patch_size;
            if let Some(grid) = flexible_grid {
                PatchInput::Flexible {
                    projection: schema.linear(
                        "patches",
                        patch * patch * config.channels,
                        width,
                        true,
                    )?,
                    positions: schema.parameter("positions", [grid * grid, width])?,
                    grid,
                    patch,
                }
            } else {
                let patches = patch_embedding(&mut schema, config.channels, width, patch, true)?;
                PatchInput::Convolution(
                    PatchTokens::new(patches).with_positions(
                        schema.parameter("positions", [1, config.patches(), width])?,
                    ),
                )
            }
        };
        Ok(Self {
            embeddings,
            encoder: PreNormEncoder::create(
                schema,
                &config.encoder,
                config.layers,
                FinalNorm::Omitted,
            )?,
            post_norm: schema.layer_norm("post_norm", width, config.encoder.eps)?,
            pooling: AttentionPooling::create(&mut schema.scope("pooling"), &config.encoder)?,
        })
    }
}

impl Layer for SiglipImageLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let tokens = self.embeddings.forward(input, state)?;
        self.encoder
            .forward(&tokens, state)?
            .apply(&self.post_norm, state)?
            .apply(&self.pooling, state)
    }
}

/// `SigLIP`'s text tower: from texts to their embeddings, `[N, W]`.
/// Every text is padded to the positions the tower learned, and every position, padding too, is read, as `SiglipTextModel` reads the tokenizer's ids without a mask; a text's embedding is its last position's state through `head`.
#[derive(Debug, Clone)]
struct SiglipTextLayers {
    tokens: Embedding,
    positions: StateKey,
    encoder: PreNormEncoder,
    head: Linear,
    length: i64,
    pad: i64,
}

impl SiglipTextLayers {
    fn create(schema: &mut Schema<'_>, config: &SiglipTextConfig) -> SingeResult<Self> {
        let width = config.encoder.width;
        Ok(Self {
            tokens: Embedding::new(
                schema.parameter("embeddings.tokens.weight", [config.vocab_size, width])?,
            ),
            positions: schema
                .parameter("embeddings.positions.weight", [config.max_positions, width])?,
            encoder: PreNormEncoder::create(
                schema,
                &config.encoder,
                config.layers,
                FinalNorm::Included,
            )?,
            head: schema.linear("head", width, width, true)?,
            length: config.max_positions,
            pad: config.pad,
        })
    }
}

impl Layer<TokenBatch> for SiglipTextLayers {
    fn forward(&self, input: &TokenBatch, state: &ModelState) -> SingeResult<Tensor> {
        let length = self.length;
        let tokens = input.tokens();
        let &[batch, count] = tokens.shape().dimensions() else {
            return Err(SingeError::invalid_argument("SigLIP tokens, [B, T]"));
        };
        // Each text's own tokens, then the pad id to the learned length, whatever the batch padded with.
        let pad = Tensor::from_vec(tokens.device(), Shape::scalar(), vec![self.pad])?;
        let tokens = Tensor::where_select(&input.mask()?.cast(DataType::Bool)?, tokens, &pad)?
            .narrow(1, 0, count.min(length))?;
        let tokens = if count < length {
            let pads = Tensor::from_vec(
                tokens.device(),
                Shape::create([batch, length - count])?,
                vec![self.pad; usize::try_from(batch * (length - count)).unwrap_or(0)],
            )?;
            Tensor::concat(&[tokens, pads], 1)?
        } else {
            tokens
        };
        let hidden = tokens
            .apply(&self.tokens, state)?
            .add(&state.tensor(&self.positions)?.unsqueeze(0)?)?;
        self.encoder
            .forward(&hidden, state)?
            .select(1, length - 1)?
            .apply(&self.head, state)
    }
}

#[derive(Debug, Clone)]
struct SiglipLayers {
    sides: ImageTextLayers<SiglipImageLayers, SiglipTextLayers>,
    logit_scale: StateKey,
    logit_bias: StateKey,
}

impl SiglipLayers {
    fn create(schema: &mut Schema<'_>, config: &SiglipConfig) -> SingeResult<Self> {
        Ok(Self {
            sides: ImageTextLayers {
                images: SiglipImageLayers::create(
                    &mut schema.scope("vision"),
                    &config.vision,
                    config.flexible_grid,
                )?,
                texts: SiglipTextLayers::create(&mut schema.scope("text"), &config.text)?,
            },
            logit_scale: schema.parameter("logit_scale", [1])?,
            logit_bias: schema.parameter("logit_bias", [1])?,
        })
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "siglip",
    formats: &[(Format::Transformers, &["siglip", "siglip2"])],
    loaders: &[HeadLoader::ZeroShotImageClassification(load_image_text)],
};

pub fn load_image_text(loader: &mut Loader<'_>) -> Result<ContrastiveEncoder> {
    let config: SiglipConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        SiglipLayers::create(schema, &config)
    })?;
    let model = loader.load_whole(&model, &transformers::rules(&config))?;
    let layers = model.layer();
    let (images, texts) = layers.sides.split(model.state());
    Ok(
        ContrastiveEncoder::new(images, texts, layers.logit_scale.clone())
            .with_sigmoid_scores(layers.logit_bias.clone()),
    )
}
