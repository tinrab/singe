//! SAM 2's video tracking, as `Sam2VideoModel` runs it from a prompted frame, forward or in reverse: each frame's coarsest map attends to the memories of the prompted frame and of up to six frames tracked before it, and to the object pointers of up to sixteen, before SAM 2's decoder reads it.
//! A frame's memory is its mask, shrunk and fused with its coarsest map, rounded through `BF16` as `transformers` stores it; its object pointer is its chosen mask's token, projected.
//! The memory bank and the choice of each frame's memories live on the host, so a frame's attention reads as many keys as its memories hold.

use singe::{
    DataType, Error as SingeError, Result as SingeResult, Shape, Tensor,
    nn::{Activation, Conv2d, Conv2dConfig, LayerNorm, Linear, Mlp, ModelState, StateKey},
    shape::spatial::{Padding2d, Spatial2d},
    tensor::{AttentionOptions, InterpolateMode},
};

use crate::{
    assembly::{
        AxialRotary, ConvNextBlock, ConvNextScaling, LinearStack, SinePositions, channel_norm,
        merge_heads, sine_positions, split_heads, to_map, to_tokens,
    },
    error::{Error, Result},
    format::Loader,
    head::{Boxed, MaskTracker, TrackedFrame, TrackingLayers, TrackingPrompt},
    model::vision::sam2::{Decoded, Sam2Config, Sam2Layers, Sam2MemoryConfig, transformers},
    schema::Schema,
};

/// The score a mask takes everywhere in a frame the object is not in, as `NO_OBJ_SCORE` sets it.
const NO_OBJECT_SCORE: f64 = -1024.0;

/// The share of the coarsest map's positions that the memory attention's input adds.
const POSITION_SHARE: f64 = 0.1;

/// The base of the object pointers' temporal sine positions' periods.
const POINTER_TEMPERATURE: f32 = 10000.0;

/// `Sam2VideoRoPEAttention`: queries and keys turned by the coarsest map's rotary tables, each memorized frame's keys by the same tables, and the object pointers' keys left unturned.
#[derive(Debug, Clone)]
struct RopeAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
}

impl RopeAttention {
    fn create(schema: &mut Schema<'_>, width: i64, keys: i64) -> SingeResult<Self> {
        Ok(Self {
            query: schema.linear("query", width, width, true)?,
            key: schema.linear("key", keys, width, true)?,
            value: schema.linear("value", keys, width, true)?,
            output: schema.linear("output", width, width, true)?,
        })
    }

    /// Attends from `queries`, `[1, N, C]`, to `keys` and `values`, `[1, K, C']`, whose last `unturned` keys the tables leave alone.
    fn forward(
        &self,
        queries: &Tensor,
        keys: &Tensor,
        values: &Tensor,
        tables: &AxialRotary,
        unturned: i64,
        heads: i64,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let query = tables.rotated(&split_heads(&queries.apply(&self.query, state)?, heads)?)?;
        let key = split_heads(&keys.apply(&self.key, state)?, heads)?;
        let &[_, _, count, head] = key.shape().dimensions() else {
            return Err(SingeError::invalid_argument("memory attention keys"));
        };
        let points = tables.points();
        let turned = count - unturned;
        let frames = turned / points;
        let mut pieces = vec![
            tables
                .rotated(
                    &key.narrow(2, 0, turned)?
                        .reshape([1, heads, frames, points, head])?,
                )?
                .reshape([1, heads, turned, head])?,
        ];
        if unturned > 0 {
            pieces.push(key.narrow(2, turned, unturned)?);
        }

        let key = Tensor::concat(&pieces, 2)?;
        let value = split_heads(&values.apply(&self.value, state)?, heads)?;
        merge_heads(&query.attend(&key, &value, None, None, AttentionOptions::default())?)?
            .apply(&self.output, state)
    }
}

/// `Sam2VideoMemoryAttentionLayer`: self-attention, attention to the memory, and a feed-forward, each normed before and added back.
#[derive(Debug, Clone)]
struct MemoryLayer {
    self_norm: LayerNorm,
    self_attention: RopeAttention,
    cross_norm: LayerNorm,
    cross_attention: RopeAttention,
    feed_forward_norm: LayerNorm,
    feed_forward: Mlp,
}

/// The video memory's layers.
#[derive(Debug, Clone)]
struct Memory {
    layers: Vec<MemoryLayer>,
    norm: LayerNorm,
    downsampler: Vec<(Conv2d, LayerNorm)>,
    downsampler_output: Conv2d,
    feature_projection: Conv2d,
    fuser: Vec<ConvNextBlock>,
    projection: Conv2d,
    /// Each memorized frame's temporal position, `[memories, 1, 1, channels]`: the last for prompted frames, the one before the `k`-th for the frame `k` before.
    temporal: StateKey,
    no_object_pointer: StateKey,
    occlusion: StateKey,
    pointer_projection: LinearStack,
    pointer_positions: Linear,
    config: Sam2MemoryConfig,
}

impl Memory {
    fn create(schema: &mut Schema<'_>, width: i64, config: &Sam2MemoryConfig) -> SingeResult<Self> {
        let channels = config.channels;
        let layers = (0..config.layers)
            .map(|index| {
                let mut scope = schema.scope(format!("layers.{index}"));
                Ok(MemoryLayer {
                    self_norm: scope.layer_norm("self_norm", width, 1e-5)?,
                    self_attention: RopeAttention::create(
                        &mut scope.scope("self_attention"),
                        width,
                        width,
                    )?,
                    cross_norm: scope.layer_norm("cross_norm", width, 1e-5)?,
                    cross_attention: RopeAttention::create(
                        &mut scope.scope("cross_attention"),
                        width,
                        channels,
                    )?,
                    feed_forward_norm: scope.layer_norm("feed_forward_norm", width, 1e-5)?,
                    feed_forward: scope.mlp(
                        "feed_forward",
                        [width, config.inner, width],
                        config.activation,
                        true,
                    )?,
                })
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let halving = Conv2dConfig {
            stride: Spatial2d::square(2),
            padding: Padding2d::symmetric(1, 1),
            ..Conv2dConfig::default()
        };
        let mut input = 1;
        let mut downsampler = Vec::with_capacity(config.downsampler_layers);
        for index in 0..config.downsampler_layers {
            let output = input * 4;
            let mut scope = schema.scope(format!("downsampler.{index}"));
            downsampler.push((
                scope.conv2d("conv", input, output, 3, halving, true)?,
                scope.layer_norm("layer_norm", output, 1e-6)?,
            ));
            input = output;
        }

        let point = |schema: &mut Schema<'_>, name: &str, from: i64, to: i64| {
            schema.conv2d(name, from, to, 1, Conv2dConfig::default(), true)
        };
        let memories =
            i64::try_from(config.memories).map_err(|_| SingeError::out_of_range("memories"))?;
        Ok(Self {
            layers,
            norm: schema.layer_norm("norm", width, 1e-5)?,
            downsampler,
            downsampler_output: point(schema, "downsampler_output", input, width)?,
            feature_projection: point(schema, "feature_projection", width, width)?,
            fuser: (0..config.fuser_layers)
                .map(|index| {
                    ConvNextBlock::create(
                        &mut schema.scope(format!("fuser.{index}")),
                        width,
                        config.fuser_activation,
                        ConvNextScaling::LayerScale,
                    )
                })
                .collect::<SingeResult<Vec<_>>>()?,
            projection: point(schema, "projection", width, channels)?,
            temporal: schema.parameter("temporal", [memories, 1, 1, channels])?,
            no_object_pointer: schema.parameter("no_object_pointer", [1, width])?,
            occlusion: schema.parameter("occlusion", [1, channels])?,
            pointer_projection: LinearStack::create(
                &mut schema.scope("pointer_projection"),
                [width; 3],
                3,
                Activation::Relu,
            )?,
            pointer_positions: schema.linear("pointer_positions", width, channels, true)?,
            config: config.clone(),
        })
    }

    /// Conditions the coarsest map, `[1, C, H, W]`, whose positions are `positions`, `[1, H · W, C]`, on `memories`, `[K, c]`, at `memory_positions`, of which the last `pointers` are object pointers' pieces.
    fn attend(
        &self,
        map: &Tensor,
        positions: &Tensor,
        memories: &Tensor,
        memory_positions: &Tensor,
        pointers: i64,
        tables: &AxialRotary,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let &[_, _, height, width] = map.shape().dimensions() else {
            return Err(SingeError::invalid_argument("SAM 2's coarsest map"));
        };
        let heads = self.config.heads;
        let keys = memories.add(memory_positions)?.unsqueeze(0)?;
        let values = memories.unsqueeze(0)?;
        let mut hidden = to_tokens(map)?.add(&positions.mul_scalar(POSITION_SHARE)?)?;
        for layer in &self.layers {
            let normed = hidden.apply(&layer.self_norm, state)?;
            hidden = hidden.add(
                &layer
                    .self_attention
                    .forward(&normed, &normed, &normed, tables, 0, heads, state)?,
            )?;
            let normed = hidden.apply(&layer.cross_norm, state)?;
            hidden = hidden.add(
                &layer
                    .cross_attention
                    .forward(&normed, &keys, &values, tables, pointers, heads, state)?,
            )?;
            hidden = hidden.add(
                &hidden
                    .apply(&layer.feed_forward_norm, state)?
                    .apply(&layer.feed_forward, state)?,
            )?;
        }
        to_map(&hidden.apply(&self.norm, state)?, (height, width))
    }

    /// Returns a frame's memory, `[H · W, c]`, from its coarsest map as the neck gives it and its mask's scores for the memory encoder, `[1, 1, S, S]`, with the occlusion embedding added when the object is not in the frame, rounded through `BF16`.
    fn encode(
        &self,
        map: &Tensor,
        mask: &Tensor,
        is_present: bool,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let mut shrunk = mask.clone();
        for (convolution, norm) in &self.downsampler {
            shrunk = channel_norm(&shrunk.apply(convolution, state)?, norm, state)?
                .apply(&self.config.downsampler_activation, state)?;
        }

        let mut fused = map
            .apply(&self.feature_projection, state)?
            .add(&shrunk.apply(&self.downsampler_output, state)?)?;
        for block in &self.fuser {
            fused = fused.apply(block, state)?;
        }

        let mut memory = to_tokens(&fused.apply(&self.projection, state)?)?.squeeze(0)?;
        if !is_present {
            memory = memory.add(state.tensor(&self.occlusion)?)?;
        }

        let data_type = memory.data_type();
        memory.cast(DataType::Bf16)?.cast(data_type)
    }

    /// Returns the temporal position of a memory `offset` frames before the tracked frame, or of a prompted frame's when `offset` is zero, `[1, c]`.
    fn temporal(&self, offset: usize, state: &ModelState) -> SingeResult<Tensor> {
        let index = offset.checked_sub(1).unwrap_or(self.config.memories - 1);
        let index = i64::try_from(index).map_err(|_| SingeError::out_of_range("memory offset"))?;
        state
            .tensor(&self.temporal)?
            .narrow(0, index, 1)?
            .reshape([1, self.config.channels])
    }

    /// Returns the object pointers' pieces, `[4 · P, c]` for pointers of width `4c`, and their positions: `get_1d_sine_pe` of each pointer's distance in frames over `reach − 1`, projected, and repeated for each piece.
    fn pointer_tokens(
        &self,
        pointers: &[(usize, &Tensor)],
        reach: usize,
        state: &ModelState,
    ) -> SingeResult<(Tensor, Tensor)> {
        let channels = self.config.channels;
        let first = pointers
            .first()
            .ok_or_else(|| SingeError::invalid_argument("object pointers"))?
            .1;
        let width = first.shape().dimensions()[1];
        let pieces = width / channels;
        let count =
            i64::try_from(pointers.len()).map_err(|_| SingeError::out_of_range("pointers"))?;
        let half = width / 2;
        let periods = (0..half)
            .map(|feature| POINTER_TEMPERATURE.powf((2 * (feature / 2)) as f32 / half as f32))
            .collect::<Vec<_>>();
        let span = (reach - 1) as f32;
        let mut sines = Vec::with_capacity(pointers.len() * width as usize);
        for &(offset, _) in pointers {
            let distance = offset as f32 / span;
            let angles = periods.iter().map(|period| distance / period);
            sines.extend(angles.clone().map(f32::sin));
            sines.extend(angles.map(f32::cos));
        }

        let positions = Tensor::from_vec(first.device(), Shape::create([count, width])?, sines)?
            .cast(first.data_type())?
            .apply(&self.pointer_positions, state)?
            .unsqueeze(1)?
            .expand(Shape::create([count, pieces, channels])?)?
            .reshape([count * pieces, channels])?;
        let tokens = Tensor::concat(
            &pointers
                .iter()
                .map(|&(_, pointer)| pointer.clone())
                .collect::<Vec<_>>(),
            0,
        )?
        .reshape([count * pieces, channels])?;
        Ok((tokens, positions))
    }
}

/// What a tracked frame leaves for the frames after it.
struct Remembered {
    frame: usize,
    /// Its memory, `[H · W, c]`.
    memory: Tensor,
    /// Its object pointer, `[1, C]`.
    pointer: Tensor,
}

/// SAM 2 with its video memory.
#[derive(Debug, Clone)]
pub struct Sam2TrackerLayers {
    image: Sam2Layers,
    memory: Memory,
}

impl Sam2TrackerLayers {
    pub fn create(schema: &mut Schema<'_>, config: &Sam2Config) -> SingeResult<Self> {
        let memory = config
            .memory
            .as_ref()
            .ok_or_else(|| SingeError::invalid_argument("a SAM 2 config without video memory"))?;
        Ok(Self {
            image: Sam2Layers::create(schema, config)?,
            memory: Memory::create(&mut schema.scope("memory"), config.fpn, memory)?,
        })
    }

    /// Returns the index of the mask to keep of a frame's, from their predicted quality: the best of the several masks, or with several points on a prompted frame, the single mask when its scores are stable, as `_dynamic_multimask_via_stability` picks it; and the index of the token its pointer comes from.
    fn chosen(&self, decoded: &Decoded, several: bool) -> SingeResult<(i64, i64)> {
        let quality = decoded.quality.cast(DataType::F32)?.to_vec::<f32>()?;
        let best = quality
            .iter()
            .enumerate()
            .skip(1)
            .fold((1, f32::NEG_INFINITY), |best, (index, &score)| {
                if score > best.1 { (index, score) } else { best }
            })
            .0;
        let best = i64::try_from(best).map_err(|_| SingeError::out_of_range("mask index"))?;
        if several {
            return Ok((best, best));
        }

        let config = &self.memory.config;
        let single = decoded
            .masks
            .narrow(1, 0, 1)?
            .cast(DataType::F32)?
            .contiguous()?
            .to_vec::<f32>()?;
        let above = |cut: f32| single.iter().filter(|&&score| score > cut).count();
        let (inside, outside) = (
            above(config.stability_delta),
            above(-config.stability_delta),
        );
        let stability = if outside > 0 {
            inside as f32 / outside as f32
        } else {
            1.0
        };
        Ok((
            if stability >= config.stability_threshold {
                0
            } else {
                best
            },
            0,
        ))
    }
}

impl TrackingLayers for Sam2TrackerLayers {
    fn track(
        &self,
        frames: usize,
        pixels: &dyn Fn(usize) -> Result<Tensor>,
        prompt: &TrackingPrompt,
        state: &ModelState,
    ) -> Result<Vec<TrackedFrame>> {
        let (image, memory) = (&self.image, &self.memory);
        let config = &memory.config;
        let start = prompt.frame;
        if start >= frames {
            return Err(Error::invalid(format!(
                "a prompt on frame {start} of a clip of {frames} frames"
            )));
        }

        if prompt.points.is_empty() || prompt.points.len() != prompt.labels.len() {
            return Err(Error::invalid(
                "a tracking prompt without points, or without a label for each",
            ));
        }

        let reach = frames.min(config.pointers);
        let positions = SinePositions::DETR;
        let size = image.config.image_size;
        let mut prompted: Option<Remembered> = None;
        let mut tracked: Vec<Remembered> = Vec::new();
        let order: Vec<usize> = if prompt.reverse {
            (0..=start).rev().collect()
        } else {
            (start..frames).collect()
        };
        // The frame `offset` steps back along the tracking's direction from `frame`.
        let behind = |frame: usize, offset: usize| {
            if prompt.reverse {
                frame.checked_add(offset)
            } else {
                frame.checked_sub(offset)
            }
        };
        let mut outputs = Vec::with_capacity(order.len());
        let mut tables: Option<(AxialRotary, Tensor, Tensor)> = None;
        for frame in order {
            let [finest, finer, coarsest] = image.levels(&pixels(frame)?, state)?;
            let channels = coarsest.shape().dimensions()[1];
            let side = coarsest.shape().dimensions()[2];
            if tables.is_none() {
                tables = Some((
                    AxialRotary::create(
                        side,
                        channels / config.heads,
                        config.rope_theta,
                        1.0,
                        &coarsest,
                    )?,
                    sine_positions(&coarsest, channels, positions)?.cast(coarsest.data_type())?,
                    sine_positions(&coarsest, config.channels, positions)?
                        .cast(coarsest.data_type())?
                        .squeeze(0)?,
                ));
            }

            let Some((rotary, map_positions, memory_positions)) = tables.as_ref() else {
                return Err(SingeError::invalid_argument("rotary tables").into());
            };
            let is_prompted = frame == start;
            let (conditioned, prompts) = if is_prompted {
                (
                    image.without_memory(&coarsest, state)?,
                    Tensor::concat(
                        &[
                            image.point_embeddings(&prompt.points, &prompt.labels, state)?,
                            image.padding(1, state)?,
                        ],
                        1,
                    )?,
                )
            } else {
                let first = prompted
                    .as_ref()
                    .ok_or_else(|| SingeError::invalid_argument("a prompted frame's memory"))?;
                let remembered_behind = |offset: usize| {
                    tracked
                        .iter()
                        .find(|earlier| Some(earlier.frame) == behind(frame, offset))
                };
                // The prompted frame's memory, then the frames tracked before this one, farthest first.
                let mut memories = vec![(0, first)];
                for offset in (1..config.memories).rev() {
                    if let Some(earlier) = remembered_behind(offset) {
                        memories.push((offset, earlier));
                    }
                }

                // The prompted frame's pointer, then the frames tracked before this one, nearest first.
                let mut pointers = vec![(frame.abs_diff(start), &first.pointer)];
                for offset in 1..reach {
                    if let Some(earlier) = remembered_behind(offset) {
                        pointers.push((offset, &earlier.pointer));
                    }
                }

                let (pointer_tokens, pointer_positions) =
                    memory.pointer_tokens(&pointers, reach, state)?;
                let pointer_count = pointer_tokens.shape().dimensions()[0];
                let mut keys = memories
                    .iter()
                    .map(|(_, remembered)| remembered.memory.clone())
                    .collect::<Vec<_>>();
                keys.push(pointer_tokens);
                let mut key_positions = memories
                    .iter()
                    .map(|&(offset, _)| memory_positions.add(&memory.temporal(offset, state)?))
                    .collect::<SingeResult<Vec<_>>>()?;
                key_positions.push(pointer_positions);
                (
                    memory.attend(
                        &coarsest,
                        map_positions,
                        &Tensor::concat(&keys, 0)?,
                        &Tensor::concat(&key_positions, 0)?,
                        pointer_count,
                        rotary,
                        state,
                    )?,
                    image.padding(2, state)?,
                )
            };

            let decoded = image.decode_prompts(&[finest, finer, conditioned], &prompts, state)?;
            let object_score = decoded.object_scores.cast(DataType::F32)?.to_vec::<f32>()?[0];
            let is_present = object_score > 0.0;
            let several = !is_prompted || prompt.points.len() <= 1;
            let (mask, token) = self.chosen(&decoded, several)?;
            let mut scores = decoded.masks.narrow(1, mask, 1)?;
            let pointer = if is_present {
                decoded
                    .tokens
                    .narrow(1, token, 1)?
                    .squeeze(1)?
                    .apply(&memory.pointer_projection, state)?
            } else {
                scores = scores.mul_scalar(0.0)?.add_scalar(NO_OBJECT_SCORE)?;
                state.tensor(&memory.no_object_pointer)?.clone()
            };

            // A prompted frame's mask enters its memory binarized, a tracked frame's as probabilities.
            let upscaled =
                scores.interpolate(Spatial2d::square(size), InterpolateMode::Bilinear)?;
            let probabilities = if is_prompted {
                upscaled
                    .greater(&upscaled.mul_scalar(0.0)?)?
                    .cast(upscaled.data_type())?
            } else {
                upscaled.sigmoid()?
            };
            let remembered = Remembered {
                frame,
                memory: memory.encode(
                    &coarsest,
                    &probabilities
                        .mul_scalar(f64::from(config.mask_scale))?
                        .add_scalar(f64::from(config.mask_bias))?,
                    is_present,
                    state,
                )?,
                pointer,
            };
            if is_prompted {
                prompted = Some(remembered);
            } else {
                tracked.push(remembered);
                let kept = config.memories.max(reach);
                if tracked.len() > kept {
                    tracked.remove(0);
                }
            }

            outputs.push(TrackedFrame {
                scores,
                object_score,
            });
        }
        Ok(outputs)
    }

    fn image_size(&self) -> i64 {
        self.image.config.image_size
    }
}

pub fn load_tracker(loader: &mut Loader<'_>) -> Result<MaskTracker> {
    let config: Sam2Config = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        Sam2TrackerLayers::create(schema, &config)
            .map(|layers| Boxed::<dyn TrackingLayers>(Box::new(layers)))
    })?;
    let model = loader.load_whole(&model, &transformers::rules(&config, true))?;
    Ok(MaskTracker::new(model))
}
