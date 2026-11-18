// Segment Anything: a vision transformer with windowed attention and decomposed relative positions embeds an image once, then a prompt encoder and a two-way transformer decode masks for points against that embedding.

mod transformers;

use std::f64::consts::PI;

use serde::{Deserialize, Serialize};
use singe::{
    Device, Result as SingeResult, Shape, Tensor,
    nn::{
        Activation, Conv2d, Conv2dConfig, ConvTranspose2d, ConvTransposeConfig, LayerNorm, Linear,
        Model, ModelState, StateBuilder, StateKey,
    },
};

use crate::{
    assembly::{Attention, Mlp},
    error::Result,
    format::Loader,
    head::{HeadLoader, MaskGenerator, PromptableLayers},
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
    // The side of the windows most blocks attend within, and the blocks that attend globally.
    pub window: i64,
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

// A layer norm over the channels of `[B, C, H, W]`.
fn channel_norm(input: &Tensor, norm: &LayerNorm, state: &ModelState) -> SingeResult<Tensor> {
    input
        .permute(&[0, 2, 3, 1])?
        .apply(norm, state)?
        .permute(&[0, 3, 1, 2])
}

#[derive(Debug, Clone)]
struct VisionBlock {
    attention_norm: LayerNorm,
    qkv: Linear,
    output: Linear,
    relative_height: StateKey,
    relative_width: StateKey,
    feed_forward_norm: LayerNorm,
    up: Linear,
    down: Linear,
    window: i64,
}

// Each pair of a side's positions' relative embedding, `[side, side, C]`, from a table of `2 · side - 1` offsets, as `get_rel_pos` reads it when queries and keys have one size.
fn relative(table: &Tensor, side: i64) -> SingeResult<Tensor> {
    let channels = table.shape().dimensions()[1];
    let index = (0..side)
        .flat_map(|query| (0..side).map(move |key| query - key + side - 1))
        .collect::<Vec<_>>();
    let index = Tensor::from_vec(table.device(), Shape::create([side * side, 1])?, index)?
        .expand(Shape::create([side * side, channels])?)?;
    table.gather(0, &index)?.reshape([side, side, channels])
}

impl VisionBlock {
    // Attention among `[N, side, side, C]` tokens, with decomposed relative positions.
    fn attend(&self, hidden: &Tensor, heads: i64, state: &ModelState) -> SingeResult<Tensor> {
        let dimensions = hidden.shape().dimensions();
        let (count, side, width) = (dimensions[0], dimensions[1], dimensions[3]);
        let head = width / heads;
        let length = side * side;
        let qkv = hidden
            .reshape([count, length, width])?
            .apply(&self.qkv, state)?
            .reshape([count, length, 3, heads, head])?
            .permute(&[2, 0, 3, 1, 4])?;
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

        #[expect(clippy::cast_precision_loss, reason = "head widths are small")]
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

    fn forward(&self, hidden: &Tensor, heads: i64, state: &ModelState) -> SingeResult<Tensor> {
        let dimensions = hidden.shape().dimensions();
        let (batch, height, width, channels) =
            (dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
        let normed = hidden.apply(&self.attention_norm, state)?;
        let attended = if self.window > 0 {
            let window = self.window;
            let padded_height = (height + window - 1) / window * window;
            let padded_width = (width + window - 1) / window * window;
            let padded = normed.pad(
                &[
                    (0, 0),
                    (0, padded_height - height),
                    (0, padded_width - width),
                    (0, 0),
                ],
                0.0,
            )?;
            let windows = padded
                .reshape([
                    batch,
                    padded_height / window,
                    window,
                    padded_width / window,
                    window,
                    channels,
                ])?
                .permute(&[0, 1, 3, 2, 4, 5])?
                .reshape([
                    batch * (padded_height / window) * (padded_width / window),
                    window,
                    window,
                    channels,
                ])?;
            self.attend(&windows, heads, state)?
                .reshape([
                    batch,
                    padded_height / window,
                    padded_width / window,
                    window,
                    window,
                    channels,
                ])?
                .permute(&[0, 1, 3, 2, 4, 5])?
                .reshape([batch, padded_height, padded_width, channels])?
                .narrow(1, 0, height)?
                .narrow(2, 0, width)?
        } else {
            self.attend(&normed, heads, state)?
        };
        let hidden = hidden.add(&attended)?;
        hidden
            .apply(&self.feed_forward_norm, state)?
            .apply(&self.up, state)?
            .apply(&Activation::GeluExact, state)?
            .apply(&self.down, state)?
            .add(&hidden)
    }
}

#[derive(Debug, Clone)]
struct TwoWayBlock {
    self_attention: Attention,
    norm1: LayerNorm,
    token_to_image: Attention,
    norm2: LayerNorm,
    up: Linear,
    down: Linear,
    norm3: LayerNorm,
    image_to_token: Attention,
    norm4: LayerNorm,
}

#[derive(Debug, Clone)]
pub struct SamLayers {
    patches: Conv2d,
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
    hypernetworks: Vec<Mlp>,
    iou_head: Mlp,
    config: SamConfig,
}

impl SamLayers {
    // Random Fourier features of coordinates in [0, 1], `[..., 2]` to `[..., 2 · F]`: sines and cosines of `2π (2c - 1) G`.
    fn fourier(&self, coordinates: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let projected = coordinates
            .mul_scalar(2.0)?
            .add_scalar(-1.0)?
            .matmul(state.tensor(&self.gaussian)?)?
            .mul_scalar(2.0 * PI)?;
        let last = projected.shape().rank() - 1;
        Tensor::concat(&[projected.sin()?, projected.cos()?], last)
    }

    // The positions of the embedding's grid, `[1, H · W, C]`, each point at its center.
    #[expect(clippy::cast_precision_loss, reason = "grid sides are small")]
    fn grid_positions(&self, side: i64, device: Device, state: &ModelState) -> SingeResult<Tensor> {
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
        let coordinates =
            Tensor::from_vec(device, Shape::create([1, side * side, 2])?, coordinates)?;
        self.fourier(&coordinates, state)
    }
}

impl PromptableLayers for SamLayers {
    fn embed(&self, pixels: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let mut hidden = pixels
            .apply(&self.patches, state)?
            .permute(&[0, 2, 3, 1])?
            .add(state.tensor(&self.positions)?)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden, self.config.heads, state)?;
        }

        let hidden = hidden.permute(&[0, 3, 1, 2])?;
        let hidden = channel_norm(
            &hidden.apply(&self.neck[0], state)?,
            &self.neck_norms[0],
            state,
        )?;
        channel_norm(
            &hidden.apply(&self.neck[1], state)?,
            &self.neck_norms[1],
            state,
        )
    }

    fn decode(
        &self,
        embedding: &Tensor,
        points: &[[f32; 2]],
        state: &ModelState,
    ) -> SingeResult<(Tensor, Tensor)> {
        let device = embedding.device();
        let dimensions = embedding.shape().dimensions();
        let (channels, side) = (dimensions[1], dimensions[2]);
        let count =
            i64::try_from(points.len()).map_err(|_| singe::Error::out_of_range("points"))?;
        #[expect(clippy::cast_precision_loss, reason = "image sides are small")]
        let input = self.config.image_size as f32;

        // Each point, shifted to its pixel's center, as a positive point, and a padding point after it.
        let coordinates = points
            .iter()
            .flat_map(|&[x, y]| [(x + 0.5) / input, (y + 0.5) / input])
            .collect::<Vec<_>>();
        let coordinates = Tensor::from_vec(device, Shape::create([count, 1, 2])?, coordinates)?;
        let positive = state.tensor(&self.point_embeddings[1])?;
        let points = self.fourier(&coordinates, state)?.add(positive)?;
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
        let output_count = tokens.shape().dimensions()[0];
        let tokens =
            tokens
                .unsqueeze(0)?
                .expand(Shape::create([count, output_count, channels])?)?;
        let prompts = Tensor::concat(&[tokens, points, padding], 1)?;

        let image = embedding
            .add(&state.tensor(&self.no_mask)?.reshape([1, channels, 1, 1])?)?
            .flatten(2, 3)?
            .transpose(1, 2)?
            .expand(Shape::create([count, side * side, channels])?)?;
        let image_positions = self
            .grid_positions(side, device, state)?
            .expand(Shape::create([count, side * side, channels])?)?;

        let (mut queries, mut keys) = (prompts.clone(), image);
        for (index, block) in self.two_way.iter().enumerate() {
            queries = if index == 0 {
                block
                    .self_attention
                    .forward(&queries, &queries, &queries, state)?
            } else {
                let positioned = queries.add(&prompts)?;
                queries.add(&block.self_attention.forward(
                    &positioned,
                    &positioned,
                    &queries,
                    state,
                )?)?
            }
            .apply(&block.norm1, state)?;
            let key = keys.add(&image_positions)?;
            queries = queries
                .add(
                    &block
                        .token_to_image
                        .forward(&queries.add(&prompts)?, &key, &keys, state)?,
                )?
                .apply(&block.norm2, state)?;
            let fed = queries
                .apply(&block.up, state)?
                .apply(&self.config.decoder_activation, state)?
                .apply(&block.down, state)?;
            queries = queries.add(&fed)?.apply(&block.norm3, state)?;
            let query = queries.add(&prompts)?;
            keys = keys
                .add(&block.image_to_token.forward(
                    &keys.add(&image_positions)?,
                    &query,
                    &queries,
                    state,
                )?)?
                .apply(&block.norm4, state)?;
        }

        let attended = self.final_attention.forward(
            &queries.add(&prompts)?,
            &keys.add(&image_positions)?,
            &keys,
            state,
        )?;
        let queries = queries.add(&attended)?.apply(&self.final_norm, state)?;

        let image = keys
            .transpose(1, 2)?
            .reshape([count, channels, side, side])?;
        let upscaled = channel_norm(
            &image.apply(&self.upscale[0], state)?,
            &self.upscale_norm,
            state,
        )?
        .apply(&Activation::GeluExact, state)?
        .apply(&self.upscale[1], state)?
        .apply(&Activation::GeluExact, state)?;
        let upscaled_dimensions = upscaled.shape().dimensions();
        let (features, height, width) = (
            upscaled_dimensions[1],
            upscaled_dimensions[2],
            upscaled_dimensions[3],
        );
        let masks = self.config.masks + 1;
        let hyper = (0..masks)
            .map(|index| {
                queries.narrow(1, 1 + index, 1)?.apply(
                    &self.hypernetworks[usize::try_from(index)
                        .map_err(|_| singe::Error::out_of_range("masks"))?],
                    state,
                )
            })
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
    heads: &[HeadLoader::MaskGeneration(load_generator)],
};

#[expect(
    clippy::too_many_lines,
    reason = "the schema lists every part of the model once"
)]
pub fn load_generator(loader: &mut Loader<'_>) -> Result<MaskGenerator> {
    let config: SamConfig = loader.config(transformers::config)?;
    let (width, embedding, iou_depth) = (config.width, config.embedding, config.iou_depth);
    let side = config.image_size / config.patch_size;
    let decoder_inner = embedding / config.downsample;
    let mut builder = StateBuilder::new();
    let layers = {
        let mut schema = Schema::root(&mut builder, loader.options().data_type);
        let patches = schema.conv2d(
            "patches",
            3,
            width,
            config.patch_size,
            Conv2dConfig {
                stride: (config.patch_size, config.patch_size),
                ..Conv2dConfig::default()
            },
            true,
        )?;
        let positions = schema.parameter("positions", [1, side, side, width])?;
        let head = width / config.heads;
        let blocks = (0..config.layers)
            .map(|index| {
                let mut schema = schema.scope(format!("blocks.{index}"));
                let window = if config.global.contains(&index) {
                    0
                } else {
                    config.window
                };
                let span = if window > 0 { window } else { side };
                Ok(VisionBlock {
                    attention_norm: schema.layer_norm("attention_norm", width, config.eps)?,
                    qkv: schema.linear("qkv", width, 3 * width, true)?,
                    output: schema.linear("output", width, width, true)?,
                    relative_height: schema.parameter("relative_height", [2 * span - 1, head])?,
                    relative_width: schema.parameter("relative_width", [2 * span - 1, head])?,
                    feed_forward_norm: schema.layer_norm("feed_forward_norm", width, config.eps)?,
                    up: schema.linear("up", width, config.inner, true)?,
                    down: schema.linear("down", config.inner, width, true)?,
                    window,
                })
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
            schema.conv2d(
                "neck.1",
                embedding,
                embedding,
                3,
                Conv2dConfig {
                    padding: (1, 1, 1, 1),
                    ..Conv2dConfig::default()
                },
                false,
            )?,
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
                let mut schema = schema.scope(format!("two_way.{index}"));
                Ok(TwoWayBlock {
                    self_attention: Attention::create(
                        &mut schema.scope("self_attention"),
                        embedding,
                        embedding,
                        config.decoder_heads,
                    )?,
                    norm1: schema.layer_norm("norm1", embedding, config.eps)?,
                    token_to_image: Attention::create(
                        &mut schema.scope("token_to_image"),
                        embedding,
                        decoder_inner,
                        config.decoder_heads,
                    )?,
                    norm2: schema.layer_norm("norm2", embedding, config.eps)?,
                    up: schema.linear("up", embedding, config.decoder_inner, true)?,
                    down: schema.linear("down", config.decoder_inner, embedding, true)?,
                    norm3: schema.layer_norm("norm3", embedding, config.eps)?,
                    image_to_token: Attention::create(
                        &mut schema.scope("image_to_token"),
                        embedding,
                        decoder_inner,
                        config.decoder_heads,
                    )?,
                    norm4: schema.layer_norm("norm4", embedding, config.eps)?,
                })
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let final_attention = Attention::create(
            &mut schema.scope("final_attention"),
            embedding,
            decoder_inner,
            config.decoder_heads,
        )?;
        // `nn.LayerNorm`'s default, which the two-way transformer's last norm keeps.
        let final_norm = schema.layer_norm("final_norm", embedding, 1e-5)?;
        let doubling = ConvTransposeConfig {
            stride: [2, 2],
            ..ConvTransposeConfig::default()
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
                Mlp::create(
                    &mut schema.scope(format!("hypernetworks.{index}")),
                    embedding,
                    embedding,
                    embedding / 8,
                    3,
                )
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let iou_head = Mlp::create(
            &mut schema.scope("iou_head"),
            embedding,
            config.iou_hidden,
            config.masks + 1,
            config.iou_depth,
        )?;
        SamLayers {
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
            config,
        }
    };
    let layers: Box<dyn PromptableLayers> = Box::new(layers);
    let schema = Model::new(layers, builder.finish()?);
    let model = loader.load("model", &schema, &transformers::rules(iou_depth))?;
    loader.finish()?;
    Ok(MaskGenerator::new(model))
}
