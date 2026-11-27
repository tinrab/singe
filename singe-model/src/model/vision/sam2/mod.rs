//! SAM 2 for images: a Hiera encoder whose stages pool their queries and attend within windows, an FPN neck, and SAM's prompt encoder and two-way transformer, whose mask decoder adds an object-score token and upscales through the encoder's two finer maps.
//! For video, a memory of earlier frames' masks and object pointers conditions each frame's coarsest map before the decoder reads it.

mod tracking;
mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Error as SingeError, Result as SingeResult, Shape, Tensor,
    nn::{
        Activation, Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig, LayerNorm,
        Linear, Mlp, ModelState, StateKey,
    },
    shape::spatial::{Padding, Padding2d, Spatial2d},
    tensor::{AttentionOptions, InterpolateMode, Interpolation},
};

use crate::{
    assembly::{
        Attention, LinearStack, channel_norm, merge_heads, merge_windows, partition_windows,
        split_heads, to_map, to_tokens,
    },
    error::Result,
    format::Loader,
    head::{Boxed, HeadLoader, MaskGenerator, PromptableLayers},
    model::{
        ModelEntry,
        vision::sam::{PromptPositions, TwoWayBlock, TwoWaySizes, fourier, grid_positions},
    },
    schema::Schema,
    source::Format,
};

/// One Hiera block's sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HieraBlock {
    pub input: i64,
    pub output: i64,
    pub heads: i64,
    /// The side of the windows it attends within, or zero to attend globally.
    pub window: i64,
    /// Whether it pools its queries, and its residual, by the query stride.
    pub pools: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sam2Config {
    pub blocks: Vec<HieraBlock>,
    /// The index of each stage's last block.
    pub stage_ends: Vec<usize>,
    pub hidden: i64,
    pub patch_kernel: i64,
    pub patch_stride: i64,
    pub patch_padding: i64,
    /// The side of the learned background positions, resized to the patches.
    pub background: i64,
    /// The side of the learned window positions, tiled over the patches.
    pub window_positions: i64,
    pub query_stride: i64,
    pub mlp_ratio: f64,
    pub activation: Activation,
    pub eps: f32,
    pub fpn: i64,
    /// The levels, counted from the finest, that take the coarser level's features upsampled.
    pub top_down: Vec<usize>,
    /// The finest levels the decoder reads.
    pub levels: usize,
    pub image_size: i64,
    pub decoder: TwoWaySizes,
    pub decoder_layers: i64,
    pub masks: i64,
    pub iou_depth: i64,
    pub iou_hidden: i64,
    /// The video memory, which image-only checkpoints lack.
    pub memory: Option<Sam2MemoryConfig>,
}

/// SAM 2's video memory: the attention that conditions a frame on earlier frames, the encoder of each frame's mask into its memory, and how a frame's mask is picked.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sam2MemoryConfig {
    pub layers: i64,
    pub heads: i64,
    pub inner: i64,
    pub activation: Activation,
    pub rope_theta: f32,
    /// The channels of a frame's memory, and of each piece of an object pointer.
    pub channels: i64,
    /// The frames whose memories a tracked frame reads: the prompted frame and up to one fewer before it.
    pub memories: usize,
    /// The most object pointers a tracked frame reads.
    pub pointers: usize,
    /// The strided convolutions that shrink a mask to the coarsest map.
    pub downsampler_layers: usize,
    pub downsampler_activation: Activation,
    /// The `ConvNeXt` blocks that fuse a mask with its frame.
    pub fuser_layers: i64,
    pub fuser_activation: Activation,
    /// The scale a mask's probabilities take before the memory encoder reads them.
    pub mask_scale: f32,
    /// The bias a mask's scaled probabilities take before the memory encoder reads them.
    pub mask_bias: f32,
    /// How far either way of zero a single mask's scores are cut to measure its stability.
    pub stability_delta: f32,
    /// The stability a single mask needs to be kept over the best of the several masks.
    pub stability_threshold: f32,
}

/// Returns `map`, `[B, H, W, C]`, max-pooled over `stride × stride` points.
fn pooled(map: &Tensor, stride: i64) -> SingeResult<Tensor> {
    map.permute(&[0, 3, 1, 2])?
        .max_pool2d(
            (stride, stride),
            (stride, stride),
            Padding2d::symmetric(0, 0),
            (1, 1),
            false,
        )?
        .permute(&[0, 2, 3, 1])
}

/// Returns `map`, `[B, H, W, C]`, padded with zeros at its bottom and right to whole windows of `window`.
fn padded(map: &Tensor, window: i64) -> SingeResult<Tensor> {
    let &[_, height, width, _] = map.shape().dimensions() else {
        return Err(SingeError::invalid_argument("a Hiera map, [B, H, W, C]"));
    };
    let (bottom, right) = (
        (window - height % window) % window,
        (window - width % window) % window,
    );
    if bottom == 0 && right == 0 {
        return Ok(map.clone());
    }
    map.pad(
        &[
            Padding::ZERO,
            Padding::new(0, bottom),
            Padding::new(0, right),
            Padding::ZERO,
        ],
        0.0,
    )
}

/// A Hiera block: attention within windows, pooling its queries when it starts a stage, and a feed-forward, each normed before and added back.
#[derive(Debug, Clone)]
struct Block {
    attention_norm: LayerNorm,
    qkv: Linear,
    output: Linear,
    feed_forward_norm: LayerNorm,
    feed_forward: Mlp,
    projection: Option<Linear>,
    config: HieraBlock,
}

impl Block {
    /// Attends within each of `windows`, `[N, h, w, C]`, pooling the queries when the block pools.
    fn attend(&self, windows: &Tensor, stride: i64, state: &ModelState) -> SingeResult<Tensor> {
        let &[count, height, width, _] = windows.shape().dimensions() else {
            return Err(SingeError::invalid_argument("Hiera windows, [N, h, w, C]"));
        };
        let out = self.config.output;
        let qkv = windows
            .apply(&self.qkv, state)?
            .reshape([count, height * width, 3, out])?;
        let mut query = qkv.select(2, 0)?;
        let (mut rows, mut columns) = (height, width);
        if self.config.pools {
            query = pooled(&query.reshape([count, height, width, out])?, stride)?;
            (rows, columns) = (height / stride, width / stride);
            query = query.reshape([count, rows * columns, out])?;
        }

        let heads = self.config.heads;
        let attended = split_heads(&query, heads)?.attend(
            &split_heads(&qkv.select(2, 1)?, heads)?,
            &split_heads(&qkv.select(2, 2)?, heads)?,
            None,
            None,
            AttentionOptions::default(),
        )?;
        merge_heads(&attended)?
            .apply(&self.output, state)?
            .reshape([count, rows, columns, out])
    }

    fn forward(&self, hidden: &Tensor, stride: i64, state: &ModelState) -> SingeResult<Tensor> {
        let batch = hidden.shape().dimensions()[0];
        let normed = hidden.apply(&self.attention_norm, state)?;
        let residual = match &self.projection {
            Some(projection) => pooled(&normed.apply(projection, state)?, stride)?,
            None => hidden.clone(),
        };
        let &[_, out_height, out_width, channels] = residual.shape().dimensions() else {
            return Err(SingeError::invalid_argument("a Hiera residual"));
        };
        let window = self.config.window;
        let attended = if window > 0 {
            let map = padded(&normed, window)?;
            let (padded_height, padded_width) =
                (map.shape().dimensions()[1], map.shape().dimensions()[2]);
            let windows = partition_windows(&map, window)?;
            let count = windows.shape().dimensions()[0];
            let windows =
                windows.reshape([count, window, window, normed.shape().dimensions()[3]])?;
            let attended = self.attend(&windows, stride, state)?;
            // A pooling block's windows shrink with its queries.
            let (window, padded_height, padded_width) = if self.config.pools {
                let window = window / stride;
                let whole = |side: i64| side + (window - side % window) % window;
                (window, whole(out_height), whole(out_width))
            } else {
                (window, padded_height, padded_width)
            };
            merge_windows(
                &attended.reshape([count, window * window, channels])?,
                window,
                batch,
                (padded_height, padded_width),
            )?
            .narrow(1, 0, out_height)?
            .narrow(2, 0, out_width)?
        } else {
            self.attend(&normed, stride, state)?
        };
        let hidden = residual.add(&attended)?;
        hidden.add(
            &hidden
                .apply(&self.feed_forward_norm, state)?
                .apply(&self.feed_forward, state)?,
        )
    }
}

/// SAM 2's mask decoder's parts past SAM's two-way transformer.
#[derive(Debug, Clone)]
struct Decoder {
    gaussian: StateKey,
    point_gaussian: StateKey,
    point_embeddings: StateKey,
    not_a_point: StateKey,
    no_mask: StateKey,
    tokens: [StateKey; 3],
    two_way: Vec<TwoWayBlock>,
    final_attention: Attention,
    final_norm: LayerNorm,
    finest: Conv2d,
    finer: Conv2d,
    upscale: [ConvTranspose2d; 2],
    upscale_norm: LayerNorm,
    hypernetworks: Vec<LinearStack>,
    iou_head: LinearStack,
    object_score: LinearStack,
}

/// What the decoder gives for a batch of prompts: every mask's scores, `[B, M, S / 4, S / 4]`, their predicted quality, `[B, M]`, their tokens, `[B, M, C]`, and the object's score, `[B, 1]`, where the first of the `M` masks is the single mask and the rest are the several.
struct Decoded {
    masks: Tensor,
    quality: Tensor,
    tokens: Tensor,
    object_scores: Tensor,
}

#[derive(Debug, Clone)]
pub struct Sam2Layers {
    patches: Conv2d,
    background: StateKey,
    window_positions: StateKey,
    blocks: Vec<Block>,
    lateral: Vec<Conv2d>,
    no_memory: StateKey,
    decoder: Decoder,
    config: Sam2Config,
}

impl Sam2Layers {
    pub fn create(schema: &mut Schema<'_>, config: &Sam2Config) -> SingeResult<Self> {
        let hidden = config.hidden;
        let patches = schema.conv2d(
            "patches",
            3,
            hidden,
            config.patch_kernel,
            Conv2dConfig {
                stride: Spatial2d::square(config.patch_stride),
                padding: Padding2d::symmetric(config.patch_padding, config.patch_padding),
                ..Conv2dConfig::default()
            },
            true,
        )?;
        let blocks = config
            .blocks
            .iter()
            .enumerate()
            .map(|(index, &block)| {
                let mut scope = schema.scope(format!("blocks.{index}"));
                let inner = (block.output as f64 * config.mlp_ratio) as i64;
                Ok(Block {
                    attention_norm: scope.layer_norm("attention_norm", block.input, config.eps)?,
                    qkv: scope.linear("qkv", block.input, 3 * block.output, true)?,
                    output: scope.linear("output", block.output, block.output, true)?,
                    feed_forward_norm: scope.layer_norm(
                        "feed_forward_norm",
                        block.output,
                        config.eps,
                    )?,
                    feed_forward: scope.mlp(
                        "feed_forward",
                        [block.output, inner, block.output],
                        config.activation,
                        true,
                    )?,
                    projection: (block.input != block.output)
                        .then(|| scope.linear("projection", block.input, block.output, true))
                        .transpose()?,
                    config: block,
                })
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let stage_widths = config
            .stage_ends
            .iter()
            .map(|&end| config.blocks[end].output)
            .collect::<Vec<_>>();
        let fpn = config.fpn;
        // Numbered from the coarsest stage, as the neck's `convs` are.
        let lateral = stage_widths
            .iter()
            .rev()
            .enumerate()
            .map(|(index, &width)| {
                schema.conv2d(
                    &format!("lateral.{index}"),
                    width,
                    fpn,
                    1,
                    Conv2dConfig::default(),
                    true,
                )
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let sizes = config.decoder;
        let embedding = sizes.embedding;
        let doubling = ConvTranspose2dConfig {
            stride: Spatial2d::square(2),
            ..ConvTranspose2dConfig::default()
        };
        let decoder = Decoder {
            gaussian: schema.parameter("gaussian", [2, embedding / 2])?,
            point_gaussian: schema.parameter("point_gaussian", [2, embedding / 2])?,
            point_embeddings: schema.parameter("point_embeddings", [4, embedding])?,
            not_a_point: schema.parameter("not_a_point", [1, embedding])?,
            no_mask: schema.parameter("no_mask", [1, embedding])?,
            tokens: [
                schema.parameter("object_token", [1, embedding])?,
                schema.parameter("iou_token", [1, embedding])?,
                schema.parameter("mask_tokens", [config.masks + 1, embedding])?,
            ],
            two_way: (0..config.decoder_layers)
                .map(|index| {
                    let positions = if index == 0 {
                        PromptPositions::Skipped
                    } else {
                        PromptPositions::Added
                    };
                    TwoWayBlock::create(
                        &mut schema.scope(format!("two_way.{index}")),
                        sizes,
                        positions,
                    )
                })
                .collect::<SingeResult<Vec<_>>>()?,
            final_attention: Attention::create(
                &mut schema.scope("final_attention"),
                sizes.cross_attention(),
            )?,
            final_norm: schema.layer_norm("final_norm", embedding, sizes.eps)?,
            finest: schema.conv2d(
                "finest",
                fpn,
                embedding / 8,
                1,
                Conv2dConfig::default(),
                true,
            )?,
            finer: schema.conv2d(
                "finer",
                fpn,
                embedding / 4,
                1,
                Conv2dConfig::default(),
                true,
            )?,
            upscale: [
                schema.conv_transpose2d(
                    "upscale.0",
                    embedding,
                    embedding / 4,
                    2,
                    doubling,
                    true,
                )?,
                schema.conv_transpose2d(
                    "upscale.1",
                    embedding / 4,
                    embedding / 8,
                    2,
                    doubling,
                    true,
                )?,
            ],
            upscale_norm: schema.layer_norm("upscale_norm", embedding / 4, 1e-6)?,
            hypernetworks: (0..=config.masks)
                .map(|index| {
                    LinearStack::create(
                        &mut schema.scope(format!("hypernetworks.{index}")),
                        [embedding, embedding, embedding / 8],
                        3,
                        Activation::Relu,
                    )
                })
                .collect::<SingeResult<Vec<_>>>()?,
            iou_head: LinearStack::create(
                &mut schema.scope("iou_head"),
                [embedding, config.iou_hidden, config.masks + 1],
                config.iou_depth,
                Activation::Relu,
            )?,
            object_score: LinearStack::create(
                &mut schema.scope("object_score"),
                [embedding, embedding, 1],
                3,
                Activation::Relu,
            )?,
        };
        Ok(Self {
            patches,
            background: schema.parameter(
                "background",
                [1, hidden, config.background, config.background],
            )?,
            window_positions: schema.parameter(
                "window_positions",
                [1, hidden, config.window_positions, config.window_positions],
            )?,
            blocks,
            lateral,
            no_memory: schema.parameter("no_memory", [1, 1, fpn])?,
            decoder,
            config: config.clone(),
        })
    }

    /// Returns the patches' positions, `[1, H, W, C]`: the background grid resized bicubically, and the window grid tiled over it.
    fn positions(&self, side: i64, state: &ModelState) -> SingeResult<Tensor> {
        let background = state.tensor(&self.background)?.interpolate(
            Spatial2d::square(side),
            Interpolation::new(InterpolateMode::Bicubic),
        )?;
        let window = self.config.window_positions;
        let tiles = side / window;
        let channels = self.config.hidden;
        let tiled = state
            .tensor(&self.window_positions)?
            .reshape([1, channels, 1, window, 1, window])?
            .expand(Shape::create([1, channels, tiles, window, tiles, window])?)?
            .reshape([1, channels, side, side])?;
        background.add(&tiled)?.permute(&[0, 2, 3, 1])
    }

    /// Returns the decoder's two finer maps, projected, and the coarsest map as the neck gives it, `[1, C, S / 16, S / 16]`.
    fn levels(&self, pixels: &Tensor, state: &ModelState) -> SingeResult<[Tensor; 3]> {
        let config = &self.config;
        let patches = pixels.apply(&self.patches, state)?.permute(&[0, 2, 3, 1])?;
        let side = patches.shape().dimensions()[1];
        let mut hidden = patches.add(&self.positions(side, state)?)?;
        let mut stages = Vec::with_capacity(config.stage_ends.len());
        for (index, block) in self.blocks.iter().enumerate() {
            hidden = block.forward(&hidden, config.query_stride, state)?;
            if config.stage_ends.contains(&index) {
                stages.push(hidden.permute(&[0, 3, 1, 2])?);
            }
        }

        // The neck, from the coarsest level down, each finer level taking the coarser one's features upsampled where `top_down` names it.
        let mut levels: Vec<Tensor> = Vec::with_capacity(stages.len());
        for (level, stage) in stages.iter().enumerate().rev() {
            let lateral = stage.apply(&self.lateral[stages.len() - 1 - level], state)?;
            let features = match levels.last() {
                Some(coarser) if config.top_down.contains(&level) => {
                    let &[_, _, height, width] = lateral.shape().dimensions() else {
                        return Err(SingeError::invalid_argument("an FPN level"));
                    };
                    lateral.add(&coarser.interpolate(
                        Spatial2d::new(height, width),
                        Interpolation::new(InterpolateMode::Nearest),
                    )?)?
                }
                _ => lateral,
            };
            levels.push(features);
        }

        levels.reverse();
        levels.truncate(config.levels);
        let decoder = &self.decoder;
        let [finest, finer, coarsest] = levels.as_slice() else {
            return Err(SingeError::invalid_argument("SAM 2's three decoder levels"));
        };
        Ok([
            finest.apply(&decoder.finest, state)?,
            finer.apply(&decoder.finer, state)?,
            coarsest.clone(),
        ])
    }

    /// Returns `map`, `[1, C, H, W]`, with the embedding of a frame read without memory added.
    fn without_memory(&self, map: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let channels = map.shape().dimensions()[1];
        map.add(
            &state
                .tensor(&self.no_memory)?
                .reshape([1, channels, 1, 1])?,
        )
    }

    /// Returns the embeddings of `points`, `[x, y]` in input pixels, each shifted to its pixel's center and marked by its label, `[1, N, C]`: 1 inside the object, 0 outside it, and 2 and 3 a box's corners.
    fn point_embeddings(
        &self,
        points: &[[f32; 2]],
        labels: &[i64],
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let decoder = &self.decoder;
        let device = state.tensor(&decoder.not_a_point)?.device();
        let count = i64::try_from(points.len()).map_err(|_| SingeError::out_of_range("points"))?;
        let input = self.config.image_size as f32;
        let coordinates = points
            .iter()
            .flat_map(|&[x, y]| [(x + 0.5) / input, (y + 0.5) / input])
            .collect::<Vec<_>>();
        let coordinates = Tensor::from_vec(device, Shape::create([1, count, 2])?, coordinates)?;
        let embeddings = state.tensor(&decoder.point_embeddings)?;
        let labels = labels
            .iter()
            .map(|&label| embeddings.narrow(0, label, 1))
            .collect::<SingeResult<Vec<_>>>()?;
        fourier(&coordinates, state.tensor(&decoder.point_gaussian)?)?
            .add(&Tensor::concat(&labels, 0)?.unsqueeze(0)?)
    }

    /// Returns `count` padding points' embeddings, `[1, count, C]`, as the prompt encoder gives a point labelled `-1`.
    fn padding(&self, count: i64, state: &ModelState) -> SingeResult<Tensor> {
        let padding = state.tensor(&self.decoder.not_a_point)?;
        let channels = padding.shape().dimensions()[1];
        padding
            .unsqueeze(0)?
            .expand(Shape::create([1, count, channels])?)
    }

    /// Decodes a batch of prompts, `[B, N, C]`, against an image's three maps.
    fn decode_prompts(
        &self,
        embedding: &[Tensor],
        prompts: &Tensor,
        state: &ModelState,
    ) -> SingeResult<Decoded> {
        let [finest, finer, image] = embedding else {
            return Err(SingeError::invalid_argument("SAM 2's three image maps"));
        };
        let decoder = &self.decoder;
        let device = image.device();
        let &[_, channels, side, _] = image.shape().dimensions() else {
            return Err(SingeError::invalid_argument("SAM 2's image embedding"));
        };
        let count = prompts.shape().dimensions()[0];
        let gaussian = state.tensor(&decoder.gaussian)?;
        let tokens = decoder
            .tokens
            .iter()
            .map(|token| state.tensor(token).cloned())
            .collect::<SingeResult<Vec<_>>>()?;
        let tokens = Tensor::concat(&tokens, 0)?;
        let outputs = tokens.shape().dimensions()[0];
        let tokens = tokens
            .unsqueeze(0)?
            .expand(Shape::create([count, outputs, channels])?)?;
        let prompts = Tensor::concat(&[tokens, prompts.clone()], 1)?;
        let keys = to_tokens(
            &image.add(
                &state
                    .tensor(&decoder.no_mask)?
                    .reshape([1, channels, 1, 1])?,
            )?,
        )?
        .expand(Shape::create([count, side * side, channels])?)?;
        let positions = grid_positions(side, gaussian, device)?.expand(Shape::create([
            count,
            side * side,
            channels,
        ])?)?;
        let (mut queries, mut keys) = (prompts.clone(), keys);
        for block in &decoder.two_way {
            (queries, keys) = block.forward((&queries, &keys), (&prompts, &positions), state)?;
        }

        let attended = decoder.final_attention.forward(
            &queries.add(&prompts)?,
            &keys.add(&positions)?,
            &keys,
            state,
        )?;
        let queries = queries.add(&attended)?.apply(&decoder.final_norm, state)?;

        // Upscaled through the finer maps, each added before its activation.
        let map = to_map(&keys, (side, side))?;
        let upscaled = channel_norm(
            &map.apply(&decoder.upscale[0], state)?.add(finer)?,
            &decoder.upscale_norm,
            state,
        )?
        .apply(&Activation::GeluExact, state)?
        .apply(&decoder.upscale[1], state)?
        .add(finest)?
        .apply(&Activation::GeluExact, state)?;
        let &[_, features, height, width] = upscaled.shape().dimensions() else {
            return Err(SingeError::invalid_argument("SAM 2's upscaled embedding"));
        };
        let mask_count = self.config.masks + 1;
        // The object-score token and the quality token come first, so the mask tokens start at 2.
        let mask_weights = decoder
            .hypernetworks
            .iter()
            .zip(2..)
            .map(|(network, index)| queries.narrow(1, index, 1)?.apply(network, state))
            .collect::<SingeResult<Vec<_>>>()?;
        let masks = Tensor::concat(&mask_weights, 1)?
            .matmul(&upscaled.reshape([count, features, height * width])?)?
            .reshape([count, mask_count, height, width])?;
        let quality = queries
            .narrow(1, 1, 1)?
            .squeeze(1)?
            .apply(&decoder.iou_head, state)?
            .apply(&Activation::Sigmoid, state)?;
        Ok(Decoded {
            masks,
            quality,
            tokens: queries.narrow(1, 2, mask_count)?,
            object_scores: queries
                .narrow(1, 0, 1)?
                .squeeze(1)?
                .apply(&decoder.object_score, state)?,
        })
    }
}

impl PromptableLayers for Sam2Layers {
    fn embed(&self, pixels: &Tensor, state: &ModelState) -> SingeResult<Vec<Tensor>> {
        let [finest, finer, coarsest] = self.levels(pixels, state)?;
        Ok(vec![finest, finer, self.without_memory(&coarsest, state)?])
    }

    fn decode(
        &self,
        embedding: &[Tensor],
        points: &[[f32; 2]],
        state: &ModelState,
    ) -> SingeResult<(Tensor, Tensor)> {
        // Each point its own prompt: a positive point and a padding point after it.
        let prompts = points
            .iter()
            .map(|&point| {
                Tensor::concat(
                    &[
                        self.point_embeddings(&[point], &[1], state)?,
                        self.padding(1, state)?,
                    ],
                    1,
                )
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let decoded = self.decode_prompts(embedding, &Tensor::concat(&prompts, 0)?, state)?;
        // The several masks' outputs, leaving out the single mask's.
        let masks = decoded.quality.shape().dimensions()[1];
        Ok((
            decoded.masks.narrow(1, 1, masks - 1)?,
            decoded.quality.narrow(1, 1, masks - 1)?,
        ))
    }

    fn image_size(&self) -> i64 {
        self.config.image_size
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "sam2",
    formats: &[(Format::Transformers, &["sam2", "sam2_video"])],
    loaders: &[
        HeadLoader::MaskGeneration(load_generator),
        HeadLoader::MaskTracking(tracking::load_tracker),
    ],
};

pub fn load_generator(loader: &mut Loader<'_>) -> Result<MaskGenerator> {
    let config: Sam2Config = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        Sam2Layers::create(schema, &config)
            .map(|layers| Boxed::<dyn PromptableLayers>(Box::new(layers)))
    })?;
    let model = loader.load_whole(&model, &transformers::rules(&config, false))?;
    Ok(MaskGenerator::new(model))
}
