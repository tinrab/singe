//! Qwen3.5 reading images: the vision encoder, and the text decoder, whose prompts read the encoder's rows in place of their image tokens on `M-RoPE`'s grid of positions.
//!
//! The encoder's state is `embeddings.patches`, a linear map of each patch's pixels stored as a convolution's `[W, C, T, P, P]`, `embeddings.positions`, `blocks.N` (`attention_norm`, `attention` with `query`, `key`, `value`, and `output`, `feed_forward_norm`, and `feed_forward` with `up` and `down`), and `merger` (`norm`, `up`, and `down`).

use std::{collections::HashMap, iter};

use singe::{
    DataType, Device, Error as SingeError, Result as SingeResult, Shape, Tensor,
    generation::{CachedDecoder, GenerationOptions, generate_batch},
    graph::{Graph, capture},
    nn::{
        Activation, DenseAttentionInput, Embedding, Layer, LayerNorm, Mlp, Model, ModelState,
        OffsetRmsNorm, RotaryAngles, RotaryLayout, SourcedCausalLm, StateKey,
    },
};

use crate::{
    BatchOptions,
    assembly::{PreNormEncoderBlock, PreNormEncoderConfig, pre_norm_encoder_block},
    decoder_cache::{DecoderKey, Decoders},
    error::{Error, Result},
    head::{ImageTextToText, LoopLimits, LoopSteps},
    model::text::qwen3_5::{
        Qwen35Block, Qwen35VisionConfig, Qwen35VisionLanguageConfig, processor::ImagePatches,
    },
    processor::{image::Image, text::TextTokenizer},
    schema::Schema,
};

/// The epsilon of every vision `LayerNorm`.
const LAYER_NORM_EPS: f32 = 1e-6;

/// Qwen3.5's vision encoder: patches embedded by a linear map of their pixels plus learned positions resampled to the image's grid, blocks of `LayerNorm`, attention rotated by each patch's row and column, and an MLP, and a merger of each `merge_size` × `merge_size` block of patches into one row of the decoder's width.
#[derive(Debug, Clone)]
pub struct Qwen35VisionEncoder {
    config: Qwen35VisionConfig,
    /// The side of the square table of learned positions.
    side: i64,
    patch_weight: StateKey,
    patch_bias: StateKey,
    positions: Embedding,
    blocks: Vec<PreNormEncoderBlock>,
    merger_norm: LayerNorm,
    merger: Mlp,
}

impl Qwen35VisionEncoder {
    pub fn create(schema: &mut Schema<'_>, config: &Qwen35VisionConfig) -> Result<Self> {
        let width = config.width;
        let spec = PreNormEncoderConfig {
            width,
            heads: config.heads,
            inner: config.inner,
            activation: config.activation,
            eps: LAYER_NORM_EPS,
            projection_biases: [true; 4],
            causal: false,
            layer_scale: false,
        };
        let patch_weight = schema.parameter(
            "embeddings.patches.weight",
            [
                width,
                config.channels,
                config.frames,
                config.patch_size,
                config.patch_size,
            ],
        )?;
        let patch_bias = schema.parameter("embeddings.patches.bias", [width])?;
        let positions = Embedding::new(
            schema.parameter("embeddings.positions.weight", [config.max_positions, width])?,
        );
        let blocks = (0..config.layers)
            .map(|index| {
                pre_norm_encoder_block(&mut schema.scope(format!("blocks.{index}")), &spec)
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let block_width = width * config.merge_size * config.merge_size;
        Ok(Self {
            config: *config,
            side: config.position_side()?,
            patch_weight,
            patch_bias,
            positions,
            blocks,
            merger_norm: schema.layer_norm("merger.norm", width, LAYER_NORM_EPS)?,
            merger: schema.mlp(
                "merger",
                [block_width, block_width, config.out_width],
                Activation::GeluExact,
                true,
            )?,
        })
    }

    /// Encodes one image's patches, `[N, C · T · P · P]` in `F32` with `N` a multiple of a merged block's patches, into a row per merged block, `[N / merge_size², out]`.
    /// `grid`, `I64` `[2]`, is the image's grid of patches, rows then columns, whose patches come first; the rest are padding, which no real patch attends to.
    ///
    /// # Errors
    ///
    /// Returns an error if `patches` is not two-dimensional.
    pub fn encode(
        &self,
        patches: &Tensor,
        grid: &Tensor,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let config = &self.config;
        let device = patches.device();
        let &[count, _] = patches.shape().dimensions() else {
            return Err(SingeError::invalid_argument("an image's patches"));
        };
        let width = config.width;
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
                    mask: None,
                },
                state,
            )?;
        }

        let merge = config.merge_size * config.merge_size;
        hidden
            .squeeze(0)?
            .apply(&self.merger_norm, state)?
            .reshape([count / merge, width * merge])?
            .apply(&self.merger, state)
    }

    /// Returns the four rows of the learned position table each patch reads, `[N · 4]`, and their weights, `[N, 4, 1]` in `F32`, resampling the square table to the image's grid bilinearly with its corners aligned, as `transformers`' `get_vision_interpolation_indices_and_weights` does.
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
            let source = places
                .cast(DataType::F32)?
                .mul_scalar((side - 1) as f64)?
                .div(&extent)?;
            let floor = source.floor()?;
            let mut indices = Vec::new();
            let mut weights = Vec::new();
            for offset in [0.0, 1.0] {
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
        let &[count] = rows.shape().dimensions() else {
            return Err(SingeError::invalid_argument("the patches' rows"));
        };
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
    let merge = integers(device, &[config.merge_size])?;
    let block = integers(device, &[config.merge_size * config.merge_size])?;
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

/// Returns each patch's rotary cosines and sines, `[N, D / 2]` in `F32`, as `transformers`' axial vision rotary computes them with base 10000: the first `D / 4` angles follow the patch's row and the next `D / 4` its column, so each axis turns half of a head's features.
fn patch_angles(
    config: &Qwen35VisionConfig,
    device: Device,
    rows: &Tensor,
    columns: &Tensor,
) -> SingeResult<(Tensor, Tensor)> {
    let head_dim = config.width / config.heads;
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

/// Qwen3.5's text decoder reading images: its prompts take the encoder's rows in place of their image tokens, and rotary positions from `M-RoPE`'s grid.
pub type Qwen35ImageDecoder = SourcedCausalLm<Qwen35Block, OffsetRmsNorm>;

/// Qwen3.5 reading images and text: the image front end's settings, the vision encoder, and the decoder, the last two with state of their own.
#[derive(Debug, Clone)]
pub struct Qwen35VisionLanguage {
    pub(crate) config: Qwen35VisionLanguageConfig,
    pub(crate) encoder: Model<Qwen35VisionEncoder>,
    pub(crate) decoder: Model<Qwen35ImageDecoder>,
    pub(crate) device: Device,
    pub(crate) decoders: Decoders<DecoderKey, CachedDecoder<Qwen35ImageDecoder>>,
}

/// A prompt that reads one image: its tokens, with one image token per row of `rows`, and the encoder's rows, `[N, W]`, of an image whose merged grid is `grid`.
#[derive(Debug, Clone)]
struct ImagePrompt {
    tokens: Vec<i64>,
    rows: Tensor,
    grid: [i64; 2],
}

impl Qwen35VisionLanguage {
    /// Encodes an image's patches into the rows its prompt's image tokens read, `[N, W]`, and returns them with its merged grid.
    fn encode(&self, image: &ImagePatches) -> Result<(Tensor, [i64; 2])> {
        let merge = self.config.encoder.merge_size;
        let grid = Tensor::from_vec(self.device, Shape::create([2])?, image.grid.to_vec())?;
        let rows = self
            .encoder
            .layer()
            .encode(&image.patches, &grid, self.encoder.state())?;
        Ok((rows, [image.grid[0] / merge, image.grid[1] / merge]))
    }

    /// Generates after each prompt in `options.rows` rows, each prompt joining as a row frees, with its image's rows written to its row's sources and its grid to its row's rotary offsets first.
    /// Returns each prompt's generated tokens, in order.
    fn generate_tokens(
        &self,
        prompts: &[ImagePrompt],
        options: &BatchOptions,
    ) -> Result<Vec<Vec<i64>>> {
        if prompts.is_empty() || options.rows == 0 {
            return Ok(vec![Vec::new(); prompts.len()]);
        }

        options.run_decoder(
            &self.decoders,
            &self.decoder,
            options.decoder_options(prompts.iter().map(|prompt| prompt.tokens.len())),
            |decoder| {
                let image = self.config.image_token_id;
                let device = self.device;
                let tokens = prompts
                    .iter()
                    .map(|prompt| prompt.tokens.as_slice())
                    .collect::<Vec<_>>();
                Ok(generate_batch(
                    decoder,
                    &tokens,
                    &options.generation,
                    None,
                    |decoder, row, index| {
                        let prompt = &prompts[index];
                        let first = prompt
                            .tokens
                            .iter()
                            .position(|&token| token == image)
                            .ok_or_else(|| {
                                SingeError::invalid_argument("a prompt without an image")
                            })?;
                        let first =
                            i64::try_from(first).map_err(|_| SingeError::out_of_range("prompt"))?;
                        let offsets = rotary_offsets(
                            device,
                            decoder.capacity(),
                            &Tensor::from_vec(device, Shape::create([1])?, vec![first])?,
                            &Tensor::from_vec(device, Shape::create([2])?, prompt.grid.to_vec())?,
                        )?;
                        Qwen35ImageDecoder::write_sources(
                            decoder.caches(),
                            row,
                            first,
                            &prompt.rows,
                        )?;
                        Qwen35ImageDecoder::write_positions(decoder.caches(), row, &offsets)
                    },
                    |_, _| Ok(()),
                )?)
            },
        )
    }

    /// Returns the step that encodes one image of up to `patches` patches into `decoder`'s first sequence's sources and rotary offsets, and its example inputs.
    /// The step is a graph whose inputs are `patches`, `[patches, C · T · P · P]` in `F32` as [`Qwen35ImageProcessor::patches`](crate::model::text::qwen3_5::Qwen35ImageProcessor::patches) cuts them, the image's first and the rest zero, `source_grid`, the image's grid of patches, `I64` `[2]`, `source_position`, where its prompt's image tokens start, `I64` `[1]`, and the caches, named as the decoder's steps name them.
    /// Its output, `source_rows`, is how many image tokens the prompt holds.
    /// Exported with the decoder's steps as `encode`, it lets a generation loop describe an image of any size up to `patches` without Singe, reading the grid and the patches from a file.
    fn encode_step_graph(
        &self,
        decoder: &CachedDecoder<Qwen35ImageDecoder>,
        patches: i64,
    ) -> Result<(Graph, HashMap<String, Tensor>)> {
        let vision = &self.config.encoder;
        let merge = vision.merge_size;
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

/// Returns how far each of `capacity` cache positions' rotary positions lie from the position, `[capacity, 3]` of `I64`, on the axes of time, height, and width, for a sequence whose image's tokens start at `first`, `[1]`, on a merged grid of `merged`, `[2]`: an image token takes the image's start on time and its row and column after it on height and width, and the text after it continues from the start plus the grid's longer side, as `transformers`' `get_rope_index` places them.
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

impl ImageTextToText for Qwen35VisionLanguage {
    fn loop_steps(&self, limits: &LoopLimits, generation: &GenerationOptions) -> Result<LoopSteps> {
        let patches = limits
            .image_patches
            .ok_or_else(|| SingeError::missing_input("the most patches of an image"))?;
        LoopSteps::one_sequence(&self.decoder, limits, generation, |decoder| {
            let (encode, _) = self.encode_step_graph(decoder, patches)?;
            Ok(Some(encode))
        })
    }

    fn image_token(&self) -> i64 {
        self.config.image_token_id
    }

    fn generate(
        &self,
        tokenizer: &TextTokenizer,
        prompt: &str,
        images: &[Image],
        options: &BatchOptions,
    ) -> Result<Vec<i64>> {
        let [image] = images else {
            return Err(Error::invalid("a prompt of other than one image"));
        };
        let patches = self
            .config
            .processor
            .patches(image, &self.config.encoder, self.device)?;
        let (rows, grid) = self.encode(&patches)?;
        let image_token = self.config.image_token_id;
        let count = usize::try_from(grid[0] * grid[1])
            .map_err(|_| SingeError::out_of_range("image tokens"))?;
        let mut tokens = Vec::new();
        for token in tokenizer.encode(prompt, false)? {
            if token == image_token {
                tokens.extend(iter::repeat_n(token, count));
            } else {
                tokens.push(token);
            }
        }

        let mut generated = self.generate_tokens(&[ImagePrompt { tokens, rows, grid }], options)?;
        Ok(generated.pop().unwrap_or_default())
    }
}
