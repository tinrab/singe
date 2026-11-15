// Qwen3.5 reading images: the image front end, the vision encoder under `model.visual`, and the text decoder, whose prompts read the encoder's rows in place of their image tokens on M-RoPE's grid of positions.

use std::{collections::HashMap, path::Path};

use serde::Deserialize;
use serde_json::Value;
use singe::{
    DataType, Device, Error as SingeError, Result as SingeResult, Shape, Tensor,
    generation::{CachedDecoder, generate_batch},
    graph::{Graph, capture},
    nn::{
        Activation, DecoderFeedForward, DenseAttentionInput, Embedding, GroupedQueryAttention,
        GroupedQueryLayers, HeadProjection, Layer, LayerNorm, Linear, Mlp, Model, ModelState,
        OffsetRmsNorm, PreNormBlock, RotaryAngles, RotaryLayout, SourcedCausalLm, StateBuilder,
        StateKey,
    },
    tensor::AttentionOptions,
};

use crate::{
    BatchOptions,
    assembly::{load_expert_weights, load_weights},
    config::{Snapshot, unsupported},
    error::{Error, Result},
    schema::Schema,
    text::text_generation::qwen3_5::{Qwen35Layer, Qwen35TextConfig, decoder},
};

// The eps of every vision LayerNorm.
const LAYER_NORM_EPS: f32 = 1e-6;

// The vision encoder's fields of a Qwen3.5 `config.json`'s `vision_config`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Qwen35VisionConfig {
    pub depth: usize,
    pub hidden_size: i64,
    pub hidden_act: Activation,
    pub intermediate_size: i64,
    pub num_heads: i64,
    pub num_position_embeddings: i64,
    pub out_hidden_size: i64,
    pub patch_size: i64,
    pub spatial_merge_size: i64,
    pub temporal_patch_size: i64,
    pub in_channels: i64,
    #[serde(default)]
    pub deepstack_visual_indexes: Vec<usize>,
}

impl Qwen35VisionConfig {
    // The features of one patch: every channel of every frame of its pixels.
    const fn patch_features(&self) -> i64 {
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }

    // The side of the square table of learned positions.
    fn position_side(&self) -> Result<i64> {
        let side = (1..=self.num_position_embeddings)
            .find(|side| side * side >= self.num_position_embeddings)
            .unwrap_or(0);
        if side * side == self.num_position_embeddings {
            Ok(side)
        } else {
            Err(unsupported(
                "a table of learned positions that is not square",
            ))
        }
    }
}

// What the image front end reads from `preprocessor_config.json`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Qwen35ImageProcessorConfig {
    pub size: ImageSizeBounds,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

// The fewest and most pixels an image is resized into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct ImageSizeBounds {
    #[serde(rename = "shortest_edge")]
    pub min_pixels: i64,
    #[serde(rename = "longest_edge")]
    pub max_pixels: i64,
}

// Qwen3.5's whole `config.json`, for a snapshot that reads images.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Qwen35VisionLanguageConfig {
    pub vision_config: Qwen35VisionConfig,
    pub text_config: Value,
    pub image_token_id: i64,
    pub vision_start_token_id: i64,
    pub vision_end_token_id: i64,
}

// The size `transformers`' `smart_resize` gives an image of `height` by `width`: each side a multiple of `factor`, the area within `bounds`, and the aspect ratio kept as closely as that allows.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "image sides are far below where `F64` rounds them"
)]
#[must_use]
pub fn smart_resize(height: i64, width: i64, factor: i64, bounds: ImageSizeBounds) -> (i64, i64) {
    let (h, w, f) = (height as f64, width as f64, factor as f64);
    let mut resized = (
        ((h / f).round_ties_even() as i64) * factor,
        ((w / f).round_ties_even() as i64) * factor,
    );
    if resized.0 * resized.1 > bounds.max_pixels {
        let beta = (h * w / bounds.max_pixels as f64).sqrt();
        resized = (
            factor.max(((h / beta / f).floor() as i64) * factor),
            factor.max(((w / beta / f).floor() as i64) * factor),
        );
    } else if resized.0 * resized.1 < bounds.min_pixels {
        let beta = (bounds.min_pixels as f64 / (h * w)).sqrt();
        resized = (
            ((h * beta / f).ceil() as i64) * factor,
            ((w * beta / f).ceil() as i64) * factor,
        );
    }
    resized
}

// An image as the encoder reads it: its patches, `[N, C · T · P · P]` in `F32`, in the order that puts each merged block's patches together, and its grid of patches, rows then columns.
#[derive(Debug, Clone)]
pub struct ImagePatches {
    pub patches: Tensor,
    pub grid: [i64; 2],
}

// Qwen3.5's vision encoder: patches embedded by a linear map of their pixels plus learned positions resampled to the image's grid, blocks of LayerNorm, attention rotated by each patch's row and column, and an MLP, and a merger of each 2 × 2 block of patches into one row of the decoder's width.
#[derive(Debug, Clone)]
pub struct Qwen35VisionEncoder {
    config: Qwen35VisionConfig,
    // The side of the square table of learned positions.
    side: i64,
    patch_weight: StateKey,
    patch_bias: StateKey,
    positions: Embedding,
    blocks: Vec<PreNormBlock<GroupedQueryAttention, Mlp, LayerNorm>>,
    merger_norm: LayerNorm,
    merger: [Linear; 2],
}

impl Qwen35VisionEncoder {
    // Encodes one image's patches, `[N, C · T · P · P]` in `F32` with `N` a multiple of the merged blocks' patches, into a row per merged block, `[N / 4, out]`.
    // `grid`, `I64` `[2]`, is the image's grid of patches, rows then columns, whose patches come first; the rest are padding, which no real patch attends to.
    pub fn encode(
        &self,
        patches: &Tensor,
        grid: &Tensor,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let config = &self.config;
        let device = patches.device();
        let count = patches.shape().dimensions()[0];
        let width = config.hidden_size;
        let data_type = state.tensor(&self.patch_bias)?.data_type();
        let weight = state
            .tensor(&self.patch_weight)?
            .reshape([width, config.patch_features()])?;
        let mut hidden = patches
            .cast(data_type)?
            .matmul(&weight.transpose(0, 1)?)?
            .add(state.tensor(&self.patch_bias)?)?;
        let (rows, columns) = patch_places(config, device, grid, count)?;
        let (indices, weights) = self.position_taps(device, grid, &rows, &columns)?;
        let positions = indices
            .apply(&self.positions, state)?
            .reshape([count, 4, width])?
            .mul(&weights.cast(data_type)?)?
            .sum(&[1], false)?;
        hidden = hidden.add(&positions)?.unsqueeze(0)?;
        let (cos, sin) = patch_angles(config, device, &rows, &columns)?;
        let angles = RotaryAngles::create(
            &cos.cast(data_type)?,
            &sin.cast(data_type)?,
            RotaryLayout::HalfSplit,
        )?;
        let real = axis(device, grid, 0)?
            .mul(&axis(device, grid, 1)?)?
            .cast(DataType::I32)?;
        for block in &self.blocks {
            hidden = block.forward(
                &DenseAttentionInput {
                    hidden: &hidden,
                    angles: Some(&angles),
                    key_lengths: Some(&real),
                },
                state,
            )?;
        }

        let merge = config.spatial_merge_size * config.spatial_merge_size;
        hidden
            .squeeze(0)?
            .apply(&self.merger_norm, state)?
            .reshape([count / merge, width * merge])?
            .apply(&self.merger[0], state)?
            .apply(&Activation::GeluExact, state)?
            .apply(&self.merger[1], state)
    }

    // The four rows of the learned position table each patch reads, `[N · 4]`, and their weights, `[N, 4, 1]` in `F32`, resampling the square table to the image's grid bilinearly with its corners aligned, as `transformers`' `get_vision_interpolation_indices_and_weights` does.
    fn position_taps(
        &self,
        device: Device,
        grid: &Tensor,
        rows: &Tensor,
        columns: &Tensor,
    ) -> SingeResult<(Tensor, Tensor)> {
        let side = self.side;
        // Each patch's two taps along one axis of the table, `[N, 2]`, and their weights.
        let taps = |places: &Tensor, dimension: i64| -> SingeResult<(Tensor, Tensor)> {
            let extent = axis(device, grid, dimension)?
                .sub(&integers(device, &[1])?)?
                .maximum(&integers(device, &[1])?)?
                .cast(DataType::F32)?;
            #[expect(
                clippy::cast_precision_loss,
                reason = "the table's side is far below where `F32` rounds it"
            )]
            let source = places
                .cast(DataType::F32)?
                .mul_scalar((side - 1) as f64)?
                .div(&extent)?;
            let floor = source.floor()?;
            let mut indices = Vec::new();
            let mut weights = Vec::new();
            for offset in [0.0, 1.0] {
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "the table's side is far below where `F32` rounds it"
                )]
                let tap = floor.add_scalar(offset)?.clamp(0.0, (side - 1) as f32)?;
                indices.push(tap.cast(DataType::I64)?);
                weights.push(
                    source
                        .sub(&floor)?
                        .sub_scalar(offset)?
                        .abs()?
                        .neg()?
                        .add_scalar(1.0)?
                        .relu()?,
                );
            }
            Ok((Tensor::stack(&indices, 1)?, Tensor::stack(&weights, 1)?))
        };
        let (row_taps, row_weights) = taps(rows, 0)?;
        let (column_taps, column_weights) = taps(columns, 1)?;
        let count = rows.shape().dimensions()[0];
        let indices = row_taps
            .unsqueeze(2)?
            .mul(&integers(device, &[side])?)?
            .add(&column_taps.unsqueeze(1)?)?
            .reshape([count * 4])?;
        let weights = row_weights
            .unsqueeze(2)?
            .mul(&column_weights.unsqueeze(1)?)?
            .reshape([count, 4, 1])?;
        Ok((indices, weights))
    }
}

fn integers(device: Device, values: &[i64]) -> SingeResult<Tensor> {
    let length = i64::try_from(values.len()).map_err(|_| SingeError::out_of_range("literal"))?;
    capture::literal(device, |device| {
        Tensor::from_vec(device, Shape::create([length])?, values.to_vec())
    })
}

fn quotient(values: &Tensor, divisors: &Tensor) -> SingeResult<Tensor> {
    values.div(divisors)?.floor()?.cast(DataType::I64)
}

fn axis(device: Device, values: &Tensor, index: i64) -> SingeResult<Tensor> {
    values.index_select(0, &integers(device, &[index])?)
}

fn patch_places(
    config: &Qwen35VisionConfig,
    device: Device,
    grid: &Tensor,
    count: i64,
) -> SingeResult<(Tensor, Tensor)> {
    let merge = integers(device, &[config.spatial_merge_size])?;
    let block = integers(
        device,
        &[config.spatial_merge_size * config.spatial_merge_size],
    )?;
    let patches = integers(device, &(0..count).collect::<Vec<_>>())?;
    let (index, within) = (quotient(&patches, &block)?, patches.remainder(&block)?);
    let block_columns = quotient(&axis(device, grid, 1)?, &merge)?;
    let rows = quotient(&index, &block_columns)?
        .mul(&merge)?
        .add(&quotient(&within, &merge)?)?;
    let columns = index
        .remainder(&block_columns)?
        .mul(&merge)?
        .add(&within.remainder(&merge)?)?;
    Ok((rows, columns))
}

// Each patch's rotary cosines and sines, `[N, D / 2]` in `F32`: a quarter of a head turns by the patch's row and the next quarter by its column, as `transformers`' axial vision rotary does with base 10000.
#[expect(
    clippy::cast_precision_loss,
    reason = "positions and frequencies are defined in F32, as the reference computes them"
)]
fn patch_angles(
    config: &Qwen35VisionConfig,
    device: Device,
    rows: &Tensor,
    columns: &Tensor,
) -> SingeResult<(Tensor, Tensor)> {
    let head_dim = config.hidden_size / config.num_heads;
    let spatial = head_dim / 2;
    let frequencies = (0..spatial / 2)
        .map(|index| 1.0 / 10_000_f32.powf((2 * index) as f32 / spatial as f32))
        .collect::<Vec<_>>();
    let length = spatial / 2;
    let frequencies = capture::literal(device, |device| {
        Tensor::from_vec(device, Shape::create([1, length])?, frequencies.clone())
    })?;
    let turned = |places: &Tensor| places.cast(DataType::F32)?.unsqueeze(1)?.mul(&frequencies);
    let angles = Tensor::concat(&[turned(rows)?, turned(columns)?], 1)?;
    Ok((angles.cos()?, angles.sin()?))
}

// Each patch's row and column, in the order that puts each merged block's patches together, as the image front end cuts them.
fn patch_order(config: &Qwen35VisionConfig, [rows, columns]: [i64; 2]) -> Vec<(i64, i64)> {
    let merge = config.spatial_merge_size;
    let mut places = Vec::with_capacity(usize::try_from(rows * columns).unwrap_or(0));
    for block_row in 0..rows / merge {
        for block_column in 0..columns / merge {
            for row in 0..merge {
                for column in 0..merge {
                    places.push((block_row * merge + row, block_column * merge + column));
                }
            }
        }
    }
    places
}

// Qwen3.5's text decoder reading images: its prompts take the encoder's rows in place of their image tokens, and rotary positions from M-RoPE's grid.
pub type Qwen35ImageDecoder =
    SourcedCausalLm<PreNormBlock<Qwen35Layer, DecoderFeedForward, OffsetRmsNorm>, OffsetRmsNorm>;

// Qwen3.5 reading images and text: the image front end's settings, the vision encoder, and the decoder, each with state of its own.
#[derive(Debug, Clone)]
pub struct Qwen35VisionLanguage {
    config: Qwen35VisionLanguageConfig,
    processor: Qwen35ImageProcessorConfig,
    encoder: Model<Qwen35VisionEncoder>,
    decoder: Model<Qwen35ImageDecoder>,
    device: Device,
}

// A prompt that reads one image: its tokens, with one image token per row of `rows`, and the encoder's rows, `[N, W]`, of an image whose merged grid is `grid`.
#[derive(Debug, Clone)]
pub struct ImagePrompt {
    pub tokens: Vec<i64>,
    pub rows: Tensor,
    pub grid: [i64; 2],
}

impl Qwen35VisionLanguage {
    #[must_use]
    pub const fn config(&self) -> &Qwen35VisionLanguageConfig {
        &self.config
    }

    #[must_use]
    pub const fn processor(&self) -> &Qwen35ImageProcessorConfig {
        &self.processor
    }

    // The side every resized image's sides are a multiple of.
    #[must_use]
    pub const fn size_factor(&self) -> i64 {
        self.config.vision_config.patch_size * self.config.vision_config.spatial_merge_size
    }

    // Cuts an image, `[3, H, W]` channel-major RGB in `[0, 1]` with each side a multiple of `size_factor`, into normalized patches on `device`, each frame of a patch the image itself, as `transformers`' processor does for a still image.
    pub fn patches(
        &self,
        pixels: &[f32],
        height: i64,
        width: i64,
        device: Device,
    ) -> Result<ImagePatches> {
        let vision = &self.config.vision_config;
        let factor = self.size_factor();
        if height <= 0 || width <= 0 || height % factor != 0 || width % factor != 0 {
            return Err(SingeError::invalid_argument(format!(
                "an image's sides are positive multiples of {factor}"
            ))
            .into());
        }

        let (height, width) = (to_usize(height)?, to_usize(width)?);
        if pixels.len() != 3 * height * width {
            return Err(SingeError::length_mismatch(
                "image pixels",
                3 * height * width,
                pixels.len(),
            )
            .into());
        }

        let patch = to_usize(vision.patch_size)?;
        let frames = to_usize(vision.temporal_patch_size)?;
        let grid = [
            i64::try_from(height / patch).map_err(|_| SingeError::out_of_range("image"))?,
            i64::try_from(width / patch).map_err(|_| SingeError::out_of_range("image"))?,
        ];
        let places = patch_order(vision, grid);
        let mut values = Vec::with_capacity(places.len() * 3 * frames * patch * patch);
        for (row, column) in places {
            let (top, left) = (to_usize(row)? * patch, to_usize(column)? * patch);
            for channel in 0..3 {
                let (mean, std) = (
                    self.processor.image_mean[channel],
                    self.processor.image_std[channel],
                );
                for _ in 0..frames {
                    for y in top..top + patch {
                        let line = &pixels[(channel * height + y) * width + left..][..patch];
                        values.extend(line.iter().map(|value| (value - mean) / std));
                    }
                }
            }
        }

        let count = grid[0] * grid[1];
        Ok(ImagePatches {
            patches: Tensor::from_vec(
                device,
                Shape::create([count, vision.patch_features()])?,
                values,
            )?,
            grid,
        })
    }

    // Encodes an image's patches into the rows its prompt's image tokens read, `[N, W]`, and returns them with its merged grid.
    pub fn encode(&self, image: &ImagePatches) -> Result<(Tensor, [i64; 2])> {
        let merge = self.config.vision_config.spatial_merge_size;
        let grid = Tensor::from_vec(self.device, Shape::create([2])?, image.grid.to_vec())?;
        let rows = self
            .encoder
            .layer()
            .encode(&image.patches, &grid, self.encoder.state())?;
        Ok((rows, [image.grid[0] / merge, image.grid[1] / merge]))
    }

    // Generates after each prompt in `options.rows` rows, each prompt joining as a row frees, with its image's rows written to its row's sources and its grid to its row's rotary offsets first.
    // Returns each prompt's generated tokens, in order.
    pub fn generate(
        &self,
        prompts: &[ImagePrompt],
        options: &BatchOptions,
    ) -> Result<Vec<Vec<i64>>> {
        if prompts.is_empty() || options.rows == 0 {
            return Ok(vec![Vec::new(); prompts.len()]);
        }

        let mut decoder = options.decoder(
            &self.decoder,
            prompts.iter().map(|prompt| prompt.tokens.len()),
        )?;

        let image = self.config.image_token_id;
        let device = self.device;
        let tokens = prompts
            .iter()
            .map(|prompt| prompt.tokens.as_slice())
            .collect::<Vec<_>>();
        Ok(generate_batch(
            &mut decoder,
            &tokens,
            &options.generation,
            None,
            |decoder, row, index| {
                let prompt = &prompts[index];
                let first = prompt
                    .tokens
                    .iter()
                    .position(|&token| token == image)
                    .ok_or_else(|| SingeError::invalid_argument("a prompt without an image"))?;
                let first = i64::try_from(first).map_err(|_| SingeError::out_of_range("prompt"))?;
                let offsets = rotary_offsets(
                    device,
                    decoder.capacity(),
                    &Tensor::from_vec(device, Shape::create([1])?, vec![first])?,
                    &Tensor::from_vec(device, Shape::create([2])?, prompt.grid.to_vec())?,
                )?;
                Qwen35ImageDecoder::write_sources(decoder.caches(), row, first, &prompt.rows)?;
                Qwen35ImageDecoder::write_positions(decoder.caches(), row, &offsets)
            },
            |_, _| Ok(()),
        )?)
    }
}

impl Qwen35VisionLanguage {
    #[must_use]
    pub const fn decoder(&self) -> &Model<Qwen35ImageDecoder> {
        &self.decoder
    }

    // The step that encodes one image of up to `patches` patches into `decoder`'s first sequence's sources and rotary offsets, as a graph whose inputs are `patches`, `[patches, C · T · P · P]` in `F32` as `patches` cuts them, the image's first and the rest zero, `source_grid`, the image's grid of patches, `I64` `[2]`, `source_position`, where its prompt's image tokens start, `I64` `[1]`, and the caches, named as the decoder's steps name them.
    // Its output, `source_rows`, is how many image tokens the prompt holds.
    // Exported with the decoder's steps as `encode`, it lets a generation loop describe an image of any size up to `patches` with nothing of Singe, reading the grid and the patches from a file.
    pub fn encode_step_graph(
        &self,
        decoder: &CachedDecoder<Qwen35ImageDecoder>,
        patches: i64,
    ) -> Result<(Graph, HashMap<String, Tensor>)> {
        let vision = &self.config.vision_config;
        let merge = vision.spatial_merge_size;
        let block = merge * merge;
        if patches <= 0 || patches % block != 0 {
            return Err(SingeError::invalid_argument(format!(
                "an encode step reads a positive multiple of {block} patches"
            ))
            .into());
        }

        let device = self.device;
        let capacity = decoder.capacity();
        let inputs = vec![
            (
                "patches".to_owned(),
                Tensor::zeroed_on(
                    device,
                    DataType::F32,
                    Shape::create([patches, vision.patch_features()])?,
                )?,
            ),
            // A grid the patches fill, so the capture's own run divides by no zero.
            (
                "source_grid".to_owned(),
                Tensor::from_vec(device, Shape::create([2])?, vec![merge, patches / merge])?,
            ),
            (
                "source_position".to_owned(),
                Tensor::from_vec(device, Shape::create([1])?, vec![0_i64])?,
            ),
        ];
        let (graph, inputs) =
            decoder.encode_step_graph(self.encoder.state(), inputs, |inputs, caches| {
                let [patches_input, grid, position] = inputs else {
                    return Err(SingeError::missing_input("the encode step's inputs"));
                };
                let rows =
                    self.encoder
                        .layer()
                        .encode(patches_input, grid, self.encoder.state())?;
                let merged = quotient(grid, &integers(device, &[merge])?)?;
                let count = axis(device, &merged, 0)?.mul(&axis(device, &merged, 1)?)?;
                // The padding's rows write the image's last row again where it already goes, so every write lands on an image token.
                let places = integers(device, &(0..patches / block).collect::<Vec<_>>())?
                    .minimum(&count.sub(&integers(device, &[1])?)?)?;
                Qwen35ImageDecoder::write_sources_at(
                    caches,
                    0,
                    &places.add(position)?,
                    &rows.index_select(0, &places)?,
                )?;
                let offsets = rotary_offsets(device, capacity, position, &merged)?;
                let every = integers(device, &(0..capacity).collect::<Vec<_>>())?;
                Qwen35ImageDecoder::write_positions_at(caches, 0, &every, &offsets)?;
                Ok(vec![count])
            })?;
        Ok((graph.with_output_names(["source_rows"])?, inputs))
    }
}

// How far each of `capacity` cache positions' rotary positions lie from the position, `[capacity, 3]` of `I64`, on the axes of time, height, and width, for a sequence whose image's tokens start at `first`, `[1]`, on a merged grid of `merged`, `[2]`: an image token takes the image's start on time and its row and column after it on height and width, and the text after it continues from the start plus the grid's longer side, as `transformers`' `get_rope_index` places them.
fn rotary_offsets(
    device: Device,
    capacity: i64,
    first: &Tensor,
    merged: &Tensor,
) -> SingeResult<Tensor> {
    let (rows, columns) = (axis(device, merged, 0)?, axis(device, merged, 1)?);
    let count = rows.mul(&columns)?;
    let after = rows.maximum(&columns)?.sub(&count)?;
    let place = integers(device, &(0..capacity).collect::<Vec<_>>())?.sub(first)?;
    let zero = integers(device, &[0])?;
    let before = place.less(&zero)?;
    let within = place.less(&count)?;
    let axes = [
        place.neg()?,
        quotient(&place, &columns)?.sub(&place)?,
        place.remainder(&columns)?.sub(&place)?,
    ]
    .iter()
    .map(|offset| {
        Tensor::where_select(
            &before,
            &zero,
            &Tensor::where_select(&within, offset, &after)?,
        )
    })
    .collect::<SingeResult<Vec<_>>>()?;
    Tensor::stack(&axes, 1)
}

fn to_usize(value: i64) -> Result<usize> {
    usize::try_from(value).map_err(|_| SingeError::out_of_range("image size").into())
}

fn encoder_schema(
    config: &Qwen35VisionConfig,
    data_type: DataType,
) -> Result<Model<Qwen35VisionEncoder>> {
    if !config.deepstack_visual_indexes.is_empty() {
        return Err(unsupported("deepstack visual features"));
    }

    let width = config.hidden_size;
    let head_dim = width / config.num_heads;
    let mut builder = StateBuilder::new();
    let layers = {
        let mut visual = Schema::new(builder.scope("model.visual"), data_type);
        let patch_weight = visual.parameter(
            "patch_embed.proj.weight",
            [
                width,
                config.in_channels,
                config.temporal_patch_size,
                config.patch_size,
                config.patch_size,
            ],
        )?;
        let patch_bias = visual.parameter("patch_embed.proj.bias", [width])?;
        let positions = Embedding::new(
            visual.parameter("pos_embed.weight", [config.num_position_embeddings, width])?,
        );
        let blocks = (0..config.depth)
            .map(|index| {
                let mut block = visual.scope(format!("blocks.{index}"));
                let norm1 = block.layer_norm("norm1", width, LAYER_NORM_EPS)?;
                let attention = {
                    let mut attention = block.scope("attn");
                    let projections = attention.linear("qkv", width, 3 * width, true)?;
                    let head = |index: i64| {
                        HeadProjection::new(
                            projections.clone().with_outputs(index * width, width),
                            config.num_heads,
                            head_dim,
                        )
                    };
                    GroupedQueryAttention::create(
                        GroupedQueryLayers {
                            query: head(0),
                            key: head(1),
                            value: head(2),
                            output: attention.linear("proj", width, width, true)?,
                        },
                        AttentionOptions::default(),
                    )?
                };
                let norm2 = block.layer_norm("norm2", width, LAYER_NORM_EPS)?;
                let mlp = {
                    let mut mlp = block.scope("mlp");
                    Mlp::new(
                        mlp.linear("linear_fc1", width, config.intermediate_size, true)?,
                        mlp.linear("linear_fc2", config.intermediate_size, width, true)?,
                        config.hidden_act,
                    )
                };
                Ok(PreNormBlock::new(norm1, attention, norm2, mlp))
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let block_width = width * config.spatial_merge_size * config.spatial_merge_size;
        let mut merger = visual.scope("merger");
        Qwen35VisionEncoder {
            config: config.clone(),
            side: config.position_side()?,
            patch_weight,
            patch_bias,
            positions,
            blocks,
            merger_norm: merger.layer_norm("norm", width, LAYER_NORM_EPS)?,
            merger: [
                merger.linear("linear_fc1", block_width, block_width, true)?,
                merger.linear("linear_fc2", block_width, config.out_hidden_size, true)?,
            ],
        }
    };
    Ok(Model::new(layers, builder.finish()?))
}

// Reads `text_config.rope_parameters.mrope_section`, the frequencies each of M-RoPE's axes takes.
fn sections(text: &Value) -> Result<[i64; 3]> {
    let parameters = text
        .get("rope_parameters")
        .or_else(|| text.get("rope_scaling"))
        .ok_or_else(|| unsupported("a text config without rope parameters"))?;
    if parameters.get("mrope_interleaved").and_then(Value::as_bool) != Some(true) {
        return Err(unsupported("M-RoPE whose sections are not interleaved"));
    }

    serde_json::from_value(
        parameters
            .get("mrope_section")
            .cloned()
            .ok_or_else(|| unsupported("rope parameters without `mrope_section`"))?,
    )
    .map_err(Error::ConfigFormat)
}

// Loads a Hugging Face snapshot directory of Qwen3.5 that reads images, converting the weights to `data_type`.
pub fn load(
    snapshot: impl AsRef<Path>,
    device: Device,
    data_type: DataType,
) -> Result<Qwen35VisionLanguage> {
    let snapshot = snapshot.as_ref();
    let files = Snapshot::new(snapshot);
    let config: Qwen35VisionLanguageConfig = files.config()?;
    let processor: Qwen35ImageProcessorConfig = files.json("preprocessor_config.json")?;
    let text: Qwen35TextConfig =
        serde_json::from_value(config.text_config.clone()).map_err(Error::ConfigFormat)?;
    text.validate()?;
    let (layers, state) = decoder(&text, "model.language_model", data_type, false)?;
    let decoder = Model::new(
        SourcedCausalLm::new(
            layers,
            config.image_token_id,
            config.vision_config.out_hidden_size,
        )
        .with_position_sections(sections(&config.text_config)?),
        state,
    );
    let encoder_names = |name: &str| name.starts_with("model.visual.").then(|| name.to_owned());
    let decoder_names = |name: &str| {
        (name.starts_with("model.language_model.") || name == "lm_head.weight")
            .then(|| name.to_owned())
    };
    Ok(Qwen35VisionLanguage {
        encoder: load_weights(
            &encoder_schema(&config.vision_config, data_type)?,
            snapshot,
            device,
            Some(&encoder_names),
        )?,
        decoder: load_expert_weights(
            &decoder,
            snapshot,
            device,
            &decoder_names,
            text.moe_intermediate_size,
        )?,
        config,
        processor,
        device,
    })
}
