//! Segment Anything: a vision transformer with windowed attention and decomposed relative positions embeds an image once, then a prompt encoder and a two-way transformer decode masks for points against that embedding.

pub(crate) mod transformers;

use std::f64::consts::PI;

use serde::{Deserialize, Serialize};
use singe::{
    Device, Error as SingeError, Result as SingeResult, Shape, Tensor,
    nn::{
        Activation, Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig, LayerNorm,
        Linear, Mlp, ModelState, PatchEmbedding, StateKey,
    },
    shape::spatial::{Padding, Spatial2d},
};

use crate::{
    assembly::{
        Attention, AttentionSpec, LinearStack, PYTORCH_NORM_EPS, channel_norm, merge_windows,
        partition_windows, patch_embedding, same_2d, to_map, to_tokens,
    },
    error::Result,
    format::Loader,
    head::{Boxed, HeadLoader, MaskGenerator, PromptableLayers},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SamConfig {
    pub width: i64,
    pub layers: i64,
    pub heads: i64,
    pub inner: i64,
    pub eps: f32,
    pub patch_size: i64,
    pub image_size: i64,
    /// The side of the windows most blocks attend within.
    pub window: i64,
    /// The blocks, counted from zero, that attend globally instead.
    pub global: Vec<i64>,
    pub embedding: i64,
    pub decoder_layers: i64,
    pub decoder_heads: i64,
    pub decoder_inner: i64,
    pub decoder_activation: Activation,
    pub downsample: i64,
    pub masks: i64,
    pub iou_depth: i64,
    pub iou_hidden: i64,
}

impl SamConfig {
    const fn side(&self) -> i64 {
        self.image_size / self.patch_size
    }

    const fn two_way(&self) -> TwoWaySizes {
        TwoWaySizes {
            embedding: self.embedding,
            heads: self.decoder_heads,
            inner: self.decoder_inner,
            activation: self.decoder_activation,
            downsample: self.downsample,
            eps: self.eps,
        }
    }
}

/// A two-way transformer's sizes, which SAM and SAM 2 share.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TwoWaySizes {
    pub embedding: i64,
    pub heads: i64,
    pub inner: i64,
    pub activation: Activation,
    /// What the cross-attentions' width is divided by.
    pub downsample: i64,
    pub eps: f32,
}

impl TwoWaySizes {
    pub(crate) const fn cross_attention(&self) -> AttentionSpec {
        AttentionSpec {
            width: self.embedding,
            context: self.embedding,
            inner: self.embedding / self.downsample,
            heads: self.heads,
            qkv_bias: true,
            output_bias: true,
        }
    }
}

/// Returns random Fourier features of coordinates in [0, 1], `[..., 2]` to `[..., 2 · F]`: sines and cosines of `2π (2c - 1) G` for the Gaussian matrix `gaussian`, `[2, F]`.
pub(crate) fn fourier(coordinates: &Tensor, gaussian: &Tensor) -> SingeResult<Tensor> {
    let projected = coordinates
        .mul_scalar(2.0)?
        .add_scalar(-1.0)?
        .matmul(gaussian)?
        .mul_scalar(2.0 * PI)?;
    let last = projected.shape().rank() - 1;
    Tensor::concat(&[projected.sin()?, projected.cos()?], last)
}

/// Returns the Fourier positions of a `side × side` grid, `[1, side², C]`, each point at its center.
pub(crate) fn grid_positions(side: i64, gaussian: &Tensor, device: Device) -> SingeResult<Tensor> {
    let coordinates = (0..side)
        .flat_map(|row| {
            (0..side).flat_map(move |column| {
                [
                    (column as f32 + 0.5) / side as f32,
                    (row as f32 + 0.5) / side as f32,
                ]
            })
        })
        .collect::<Vec<_>>();
    let coordinates = Tensor::from_vec(device, Shape::create([1, side * side, 2])?, coordinates)?;
    fourier(&coordinates, gaussian)
}

/// Returns each pair of a side's positions' relative embedding, `[side, side, C]`, from a table of `2 · side - 1` offsets, as `get_rel_pos` reads it when queries and keys have one size.
fn relative(table: &Tensor, side: i64) -> SingeResult<Tensor> {
    let &[_, channels] = table.shape().dimensions() else {
        return Err(SingeError::invalid_argument("SAM relative position table"));
    };
    let index = (0..side)
        .flat_map(|query| (0..side).map(move |key| query - key + side - 1))
        .collect::<Vec<_>>();
    let index = Tensor::from_vec(table.device(), Shape::create([side * side, 1])?, index)?
        .expand(Shape::create([side * side, channels])?)?;
    table.gather(0, &index)?.reshape([side, side, channels])
}

#[derive(Debug, Clone)]
struct VisionBlock {
    attention_norm: LayerNorm,
    qkv: Linear,
    output: Linear,
    relative_height: StateKey,
    relative_width: StateKey,
    feed_forward_norm: LayerNorm,
    feed_forward: Mlp,
    heads: i64,
    window: i64,
}

impl VisionBlock {
    fn create(schema: &mut Schema<'_>, config: &SamConfig, window: i64) -> SingeResult<Self> {
        let width = config.width;
        let span = if window > 0 { window } else { config.side() };
        let head = width / config.heads;
        let attention_norm = schema.layer_norm("attention_norm", width, config.eps)?;
        let mut attention = schema.scope("attention");
        let qkv = attention.linear("qkv", width, 3 * width, true)?;
        let output = attention.linear("output", width, width, true)?;
        let relative_height = attention.parameter("relative_height", [2 * span - 1, head])?;
        let relative_width = attention.parameter("relative_width", [2 * span - 1, head])?;
        Ok(Self {
            attention_norm,
            qkv,
            output,
            relative_height,
            relative_width,
            feed_forward_norm: schema.layer_norm("feed_forward_norm", width, config.eps)?,
            feed_forward: schema.mlp(
                "feed_forward",
                [width, config.inner, width],
                Activation::GeluExact,
                true,
            )?,
            heads: config.heads,
            window,
        })
    }

    /// Attends among `[N, side, side, C]` tokens, with decomposed relative positions.
    fn attend(&self, hidden: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let &[count, side, _, width] = hidden.shape().dimensions() else {
            return Err(SingeError::invalid_argument("SAM attention input"));
        };
        let heads = self.heads;
        let head = width / heads;
        let length = side * side;
        let qkv = hidden
            .reshape([count, length, width])?
            .apply(&self.qkv, state)?
            .reshape([count, length, 3, heads, head])?
            .permute(&[2, 0, 3, 1, 4])?
            // One copy lays each head out whole, so the matmuls below read it as cuBLAS's strided batches rather than falling to a generated kernel.
            .contiguous()?;
        let part = |index: i64| qkv.select(0, index);
        let (query, key, value) = (part(0)?, part(1)?, part(2)?);

        // The query's dot products with each row's and each column's relative embedding, broadcast over the other axis.
        let grid = query.reshape([count * heads, side, side, head])?;
        let along = |table: &StateKey, rows: bool| -> SingeResult<Tensor> {
            let table = relative(&state.tensor(table)?.cast(query.data_type())?, side)?;
            // `[side, count · heads · side, head]` against `[side, head, side]`, the shared axis first.
            let axis_first = if rows {
                grid.permute(&[1, 0, 2, 3])?
            } else {
                grid.permute(&[2, 0, 1, 3])?
            };
            let products = axis_first
                .reshape([side, count * heads * side, head])?
                .matmul(&table.transpose(1, 2)?)?
                .reshape([side, count * heads, side, side])?;
            if rows {
                products.permute(&[1, 0, 2, 3])
            } else {
                products.permute(&[1, 2, 0, 3])
            }
        };
        let rows = along(&self.relative_height, true)?;
        let columns = along(&self.relative_width, false)?;
        let bias = rows
            .unsqueeze(4)?
            .add(&columns.unsqueeze(3)?)?
            .reshape([count, heads, length, length])?;

        let scale = (head as f64).powf(-0.5);
        query
            .mul_scalar(scale)?
            .matmul(&key.transpose(2, 3)?)?
            .add(&bias)?
            .softmax(-1)?
            .matmul(&value)?
            .transpose(1, 2)?
            .reshape([count, side, side, width])?
            .apply(&self.output, state)
    }

    fn forward(&self, hidden: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let &[batch, height, width, _] = hidden.shape().dimensions() else {
            return Err(SingeError::invalid_argument("SAM vision block input"));
        };
        let normed = hidden.apply(&self.attention_norm, state)?;
        let attended = if self.window > 0 {
            let window = self.window;
            let padded_height = (height + window - 1) / window * window;
            let padded_width = (width + window - 1) / window * window;
            let padded = normed.pad(
                &[
                    Padding::ZERO,
                    Padding::new(0, padded_height - height),
                    Padding::new(0, padded_width - width),
                    Padding::ZERO,
                ],
                0.0,
            )?;
            let windows = partition_windows(&padded, window)?;
            let &[count, _, channels] = windows.shape().dimensions() else {
                return Err(SingeError::invalid_argument("SAM windows"));
            };
            let attended = self
                .attend(&windows.reshape([count, window, window, channels])?, state)?
                .reshape([count, window * window, channels])?;
            merge_windows(&attended, window, batch, (padded_height, padded_width))?
                .narrow(1, 0, height)?
                .narrow(2, 0, width)?
        } else {
            self.attend(&normed, state)?
        };
        let hidden = hidden.add(&attended)?;
        hidden
            .apply(&self.feed_forward_norm, state)?
            .apply(&self.feed_forward, state)?
            .add(&hidden)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PromptPositions {
    Skipped,
    Added,
}

#[derive(Debug, Clone)]
pub(crate) struct TwoWayBlock {
    attention: Attention,
    attention_norm: LayerNorm,
    token_to_image: Attention,
    token_to_image_norm: LayerNorm,
    feed_forward: Mlp,
    feed_forward_norm: LayerNorm,
    image_to_token: Attention,
    image_to_token_norm: LayerNorm,
    prompt_positions: PromptPositions,
}

impl TwoWayBlock {
    pub(crate) fn create(
        schema: &mut Schema<'_>,
        config: TwoWaySizes,
        prompt_positions: PromptPositions,
    ) -> SingeResult<Self> {
        let embedding = config.embedding;
        Ok(Self {
            attention: Attention::create(
                &mut schema.scope("attention"),
                AttentionSpec::square(embedding, config.heads),
            )?,
            attention_norm: schema.layer_norm("attention_norm", embedding, config.eps)?,
            token_to_image: Attention::create(
                &mut schema.scope("token_to_image"),
                config.cross_attention(),
            )?,
            token_to_image_norm: schema.layer_norm("token_to_image_norm", embedding, config.eps)?,
            feed_forward: schema.mlp(
                "feed_forward",
                [embedding, config.inner, embedding],
                config.activation,
                true,
            )?,
            feed_forward_norm: schema.layer_norm("feed_forward_norm", embedding, config.eps)?,
            image_to_token: Attention::create(
                &mut schema.scope("image_to_token"),
                config.cross_attention(),
            )?,
            image_to_token_norm: schema.layer_norm("image_to_token_norm", embedding, config.eps)?,
            prompt_positions,
        })
    }

    pub(crate) fn forward(
        &self,
        (queries, keys): (&Tensor, &Tensor),
        (prompts, image_positions): (&Tensor, &Tensor),
        state: &ModelState,
    ) -> SingeResult<(Tensor, Tensor)> {
        let queries = match self.prompt_positions {
            PromptPositions::Skipped => self.attention.forward(queries, queries, queries, state)?,
            PromptPositions::Added => {
                let positioned = queries.add(prompts)?;
                queries.add(
                    &self
                        .attention
                        .forward(&positioned, &positioned, queries, state)?,
                )?
            }
        }
        .apply(&self.attention_norm, state)?;
        let key = keys.add(image_positions)?;
        let queries = queries
            .add(
                &self
                    .token_to_image
                    .forward(&queries.add(prompts)?, &key, keys, state)?,
            )?
            .apply(&self.token_to_image_norm, state)?;
        let fed = queries.apply(&self.feed_forward, state)?;
        let queries = queries.add(&fed)?.apply(&self.feed_forward_norm, state)?;
        let query = queries.add(prompts)?;
        let keys = keys
            .add(&self.image_to_token.forward(
                &keys.add(image_positions)?,
                &query,
                &queries,
                state,
            )?)?
            .apply(&self.image_to_token_norm, state)?;
        Ok((queries, keys))
    }
}

#[derive(Debug, Clone)]
pub struct SamLayers {
    patches: PatchEmbedding,
    positions: StateKey,
    blocks: Vec<VisionBlock>,
    neck: [Conv2d; 2],
    neck_norms: [LayerNorm; 2],
    gaussian: StateKey,
    point_embeddings: Vec<StateKey>,
    not_a_point: StateKey,
    no_mask: StateKey,
    output_tokens: [StateKey; 2],
    two_way: Vec<TwoWayBlock>,
    final_attention: Attention,
    final_norm: LayerNorm,
    upscale: [ConvTranspose2d; 2],
    upscale_norm: LayerNorm,
    hypernetworks: Vec<LinearStack>,
    iou_head: LinearStack,
    config: SamConfig,
}

impl SamLayers {
    pub fn create(schema: &mut Schema<'_>, config: &SamConfig) -> SingeResult<Self> {
        let (width, embedding) = (config.width, config.embedding);
        let side = config.side();
        let patches = patch_embedding(schema, 3, width, config.patch_size, true)?;
        let positions = schema.parameter("positions", [1, side, side, width])?;
        let blocks = (0..config.layers)
            .map(|index| {
                let window = if config.global.contains(&index) {
                    0
                } else {
                    config.window
                };
                VisionBlock::create(&mut schema.scope(format!("blocks.{index}")), config, window)
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let neck = [
            schema.conv2d(
                "neck.0",
                width,
                embedding,
                1,
                Conv2dConfig::default(),
                false,
            )?,
            schema.conv2d("neck.1", embedding, embedding, 3, same_2d(3, 1), false)?,
        ];
        let neck_norms = [
            schema.layer_norm("neck_norms.0", embedding, 1e-6)?,
            schema.layer_norm("neck_norms.1", embedding, 1e-6)?,
        ];
        let gaussian = schema.parameter("gaussian", [2, embedding / 2])?;
        let point_embeddings = (0..4)
            .map(|index| schema.parameter(&format!("point_embeddings.{index}"), [1, embedding]))
            .collect::<SingeResult<Vec<_>>>()?;
        let not_a_point = schema.parameter("not_a_point", [1, embedding])?;
        let no_mask = schema.parameter("no_mask", [1, embedding])?;
        let output_tokens = [
            schema.parameter("iou_token", [1, embedding])?,
            schema.parameter("mask_tokens", [config.masks + 1, embedding])?,
        ];
        let two_way = (0..config.decoder_layers)
            .map(|index| {
                let prompt_positions = if index == 0 {
                    PromptPositions::Skipped
                } else {
                    PromptPositions::Added
                };
                TwoWayBlock::create(
                    &mut schema.scope(format!("two_way.{index}")),
                    config.two_way(),
                    prompt_positions,
                )
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let final_attention = Attention::create(
            &mut schema.scope("final_attention"),
            config.two_way().cross_attention(),
        )?;
        let final_norm = schema.layer_norm("final_norm", embedding, PYTORCH_NORM_EPS)?;
        let doubling = ConvTranspose2dConfig {
            stride: Spatial2d::square(2),
            ..ConvTranspose2dConfig::default()
        };
        let upscale = [
            schema.conv_transpose2d("upscale.0", embedding, embedding / 4, 2, doubling, true)?,
            schema.conv_transpose2d(
                "upscale.1",
                embedding / 4,
                embedding / 8,
                2,
                doubling,
                true,
            )?,
        ];
        let upscale_norm = schema.layer_norm("upscale_norm", embedding / 4, 1e-6)?;
        let hypernetworks = (0..=config.masks)
            .map(|index| {
                LinearStack::create(
                    &mut schema.scope(format!("hypernetworks.{index}")),
                    [embedding, embedding, embedding / 8],
                    3,
                    Activation::Relu,
                )
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let iou_head = LinearStack::create(
            &mut schema.scope("iou_head"),
            [embedding, config.iou_hidden, config.masks + 1],
            config.iou_depth,
            Activation::Relu,
        )?;
        Ok(Self {
            patches,
            positions,
            blocks,
            neck,
            neck_norms,
            gaussian,
            point_embeddings,
            not_a_point,
            no_mask,
            output_tokens,
            two_way,
            final_attention,
            final_norm,
            upscale,
            upscale_norm,
            hypernetworks,
            iou_head,
            config: config.clone(),
        })
    }
}

impl PromptableLayers for SamLayers {
    fn embed(&self, pixels: &Tensor, state: &ModelState) -> SingeResult<Vec<Tensor>> {
        let side = self.config.side();
        let tokens = pixels.apply(&self.patches, state)?;
        let &[batch, _, width] = tokens.shape().dimensions() else {
            return Err(SingeError::invalid_argument("SAM patches"));
        };
        let mut hidden = tokens
            .reshape([batch, side, side, width])?
            .add(state.tensor(&self.positions)?)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden, state)?;
        }

        let hidden = hidden.permute(&[0, 3, 1, 2])?;
        let hidden = channel_norm(
            &hidden.apply(&self.neck[0], state)?,
            &self.neck_norms[0],
            state,
        )?;
        Ok(vec![channel_norm(
            &hidden.apply(&self.neck[1], state)?,
            &self.neck_norms[1],
            state,
        )?])
    }

    fn decode(
        &self,
        embedding: &[Tensor],
        points: &[[f32; 2]],
        state: &ModelState,
    ) -> SingeResult<(Tensor, Tensor)> {
        let [embedding] = embedding else {
            return Err(SingeError::invalid_argument("SAM's one image embedding"));
        };
        let device = embedding.device();
        let &[_, channels, side, _] = embedding.shape().dimensions() else {
            return Err(SingeError::invalid_argument("SAM image embedding"));
        };
        let count = i64::try_from(points.len()).map_err(|_| SingeError::out_of_range("points"))?;
        let input = self.config.image_size as f32;

        // Each point, shifted to its pixel's center, as a positive point, and a padding point after it.
        let coordinates = points
            .iter()
            .flat_map(|&[x, y]| [(x + 0.5) / input, (y + 0.5) / input])
            .collect::<Vec<_>>();
        let coordinates = Tensor::from_vec(device, Shape::create([count, 1, 2])?, coordinates)?;
        let positive = state.tensor(&self.point_embeddings[1])?;
        let points = fourier(&coordinates, state.tensor(&self.gaussian)?)?.add(positive)?;
        let padding = state
            .tensor(&self.not_a_point)?
            .unsqueeze(0)?
            .expand(Shape::create([count, 1, channels])?)?;
        let tokens = Tensor::concat(
            &[
                state.tensor(&self.output_tokens[0])?.clone(),
                state.tensor(&self.output_tokens[1])?.clone(),
            ],
            0,
        )?;
        let &[output_count, _] = tokens.shape().dimensions() else {
            return Err(SingeError::invalid_argument("SAM output tokens"));
        };
        let tokens =
            tokens
                .unsqueeze(0)?
                .expand(Shape::create([count, output_count, channels])?)?;
        let prompts = Tensor::concat(&[tokens, points, padding], 1)?;

        let image = to_tokens(
            &embedding.add(&state.tensor(&self.no_mask)?.reshape([1, channels, 1, 1])?)?,
        )?
        .expand(Shape::create([count, side * side, channels])?)?;
        let image_positions = grid_positions(side, state.tensor(&self.gaussian)?, device)?
            .expand(Shape::create([count, side * side, channels])?)?;

        let (mut queries, mut keys) = (prompts.clone(), image);
        for block in &self.two_way {
            (queries, keys) =
                block.forward((&queries, &keys), (&prompts, &image_positions), state)?;
        }

        let attended = self.final_attention.forward(
            &queries.add(&prompts)?,
            &keys.add(&image_positions)?,
            &keys,
            state,
        )?;
        let queries = queries.add(&attended)?.apply(&self.final_norm, state)?;

        let image = to_map(&keys, (side, side))?;
        let upscaled = channel_norm(
            &image.apply(&self.upscale[0], state)?,
            &self.upscale_norm,
            state,
        )?
        .apply(&Activation::GeluExact, state)?
        .apply(&self.upscale[1], state)?
        .apply(&Activation::GeluExact, state)?;
        let &[_, features, height, width] = upscaled.shape().dimensions() else {
            return Err(SingeError::invalid_argument("SAM upscaled embedding"));
        };
        let masks = self.config.masks + 1;
        let hyper = (1..)
            .zip(&self.hypernetworks)
            .map(|(token, network)| queries.narrow(1, token, 1)?.apply(network, state))
            .collect::<SingeResult<Vec<_>>>()?;
        let masks_out = Tensor::concat(&hyper, 1)?
            .matmul(&upscaled.reshape([count, features, height * width])?)?
            .reshape([count, masks, height, width])?;
        let iou = queries
            .narrow(1, 0, 1)?
            .squeeze(1)?
            .apply(&self.iou_head, state)?;
        // The several masks' outputs, leaving out the single mask's.
        Ok((
            masks_out.narrow(1, 1, masks - 1)?,
            iou.narrow(1, 1, masks - 1)?,
        ))
    }

    fn image_size(&self) -> i64 {
        self.config.image_size
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "sam",
    formats: &[(Format::Transformers, &["sam"])],
    loaders: &[HeadLoader::MaskGeneration(load_generator)],
};

pub fn load_generator(loader: &mut Loader<'_>) -> Result<MaskGenerator> {
    let config: SamConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        SamLayers::create(schema, &config)
            .map(|layers| Boxed::<dyn PromptableLayers>(Box::new(layers)))
    })?;
    let model = loader.load_whole(&model, &transformers::rules(config.iou_depth))?;
    Ok(MaskGenerator::new(model))
}
