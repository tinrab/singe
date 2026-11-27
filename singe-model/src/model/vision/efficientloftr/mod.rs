//! `EfficientLoFTR`: a `RepVGG` backbone's coarse maps of both images refined by attention over aggregated cells, within each image with 2-D rotary positions and across to the other, matched cell to cell by a dual softmax, and refined in each matched cell's window of the full-resolution features.
//! It reproduces `transformers`' port as it runs: the first fine stage's dual softmax runs over the keypoints and the first window's points rather than over both windows, its offsets are always the window's first, since its `gather` reads the keypoint axis, and the post-processing pairs the first image's matched cells, in the second image's order, with the second image's, in the first image's order.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Error as SingeError, Result as SingeResult, Shape, Tensor,
    graph::{ComputePrecision, with_convolution_precision},
    nn::{BatchNorm, Conv2d, Conv2dConfig, Layer, LayerNorm, Linear, ModelState},
    shape::spatial::{Padding, Padding2d, Spatial2d},
    tensor::{AttentionOptions, InterpolateMode, Interpolation},
};

use crate::{
    assembly::{ConvNorm, PYTORCH_NORM_EPS, merge_heads, split_heads, swapped_pairs},
    error::Result,
    format::Loader,
    head::{Boxed, HeadLoader, KeypointMatcher, MatchingLayers, PairMatching},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

/// One `RepVGG` block's channels and stride.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepVggBlock {
    pub input: i64,
    pub output: i64,
    pub stride: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EfficientLoftrConfig {
    pub stages: Vec<Vec<RepVggBlock>>,
    pub width: i64,
    pub heads: i64,
    pub layers: i64,
    pub inner: i64,
    pub aggregation: i64,
    pub theta: f64,
    /// The features of a head the rotary positions cover, as a multiple of its width.
    pub rotary_factor: f64,
    pub temperature: f32,
    pub threshold: f32,
    pub border: i64,
    pub fine_dims: Vec<i64>,
    pub fine_kernel: i64,
    pub slice: i64,
    pub regress_temperature: f32,
}

/// A `RepVGG` block: a 3 × 3 and a 1 × 1 convolution with batch norms, and a batch-normed identity where the shape allows, summed, then `ReLU`.
#[derive(Debug, Clone)]
struct RepVgg {
    wide: ConvNorm,
    narrow: ConvNorm,
    identity: Option<BatchNorm>,
}

impl Layer for RepVgg {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let mut sum = input
            .apply(&self.wide, state)?
            .add(&input.apply(&self.narrow, state)?)?;
        if let Some(identity) = &self.identity {
            sum = sum.add(&input.apply(identity, state)?)?;
        }
        sum.relu()
    }
}

/// Attention over aggregated cells: queries from a strided depthwise convolution, keys and values from a max pool, one norm on both, attention, upsampled back, and an MLP of each point and its message added to it.
#[derive(Debug, Clone)]
struct AggregatedAttention {
    queries: Conv2d,
    norm: LayerNorm,
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    up: Linear,
    down: Linear,
    output_norm: LayerNorm,
}

impl AggregatedAttention {
    fn create(schema: &mut Schema<'_>, config: &EfficientLoftrConfig) -> SingeResult<Self> {
        let width = config.width;
        let linear = |schema: &mut Schema<'_>, name: &str, input: i64, output: i64| {
            schema.linear(name, input, output, false)
        };
        Ok(Self {
            queries: schema.conv2d(
                "aggregation.q_aggregation",
                width,
                width,
                config.aggregation,
                Conv2dConfig {
                    stride: Spatial2d::square(config.aggregation),
                    groups: width,
                    ..Conv2dConfig::default()
                },
                false,
            )?,
            norm: schema.layer_norm("aggregation.norm", width, PYTORCH_NORM_EPS)?,
            query: linear(schema, "attention.q_proj", width, width)?,
            key: linear(schema, "attention.k_proj", width, width)?,
            value: linear(schema, "attention.v_proj", width, width)?,
            output: linear(schema, "attention.o_proj", width, width)?,
            up: linear(schema, "mlp.fc1", 2 * width, config.inner)?,
            down: linear(schema, "mlp.fc2", config.inner, width)?,
            output_norm: schema.layer_norm("mlp.layer_norm", width, PYTORCH_NORM_EPS)?,
        })
    }

    /// Returns `hidden`, `[B, C, H, W]`, updated by attention to `context`'s cells, or its own with rotary `angles`.
    fn forward(
        &self,
        hidden: &Tensor,
        context: Option<&Tensor>,
        angles: Option<&Rotation>,
        [heads, aggregation]: [i64; 2],
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let &[batch, width, height, columns] = hidden.shape().dimensions() else {
            return Err(SingeError::invalid_argument(
                "an EfficientLoFTR map, [B, C, H, W]",
            ));
        };
        let cells = |map: Tensor| -> SingeResult<Tensor> {
            let &[_, _, rows, across] = map.shape().dimensions() else {
                return Err(SingeError::invalid_argument("aggregated cells"));
            };
            map.permute(&[0, 2, 3, 1])?
                .apply(&self.norm, state)?
                .reshape([batch, rows * across, width])
        };
        let queries = cells(hidden.apply(&self.queries, state)?)?;
        let pooled = context.unwrap_or(hidden).max_pool2d(
            (aggregation, aggregation),
            (aggregation, aggregation),
            Padding2d::symmetric(0, 0),
            (1, 1),
            false,
        )?;
        let (rows, across) = (
            pooled.shape().dimensions()[2],
            pooled.shape().dimensions()[3],
        );
        let sources = cells(pooled)?;
        let rotated = |input: Tensor| -> SingeResult<Tensor> {
            match angles {
                Some(angles) => angles.rotate(&input),
                None => Ok(input),
            }
        };
        let query = rotated(queries.apply(&self.query, state)?)?;
        let key = rotated(sources.apply(&self.key, state)?)?;
        let value = sources.apply(&self.value, state)?;
        let attended = split_heads(&query, heads)?.attend(
            &split_heads(&key, heads)?,
            &split_heads(&value, heads)?,
            None,
            None,
            AttentionOptions::default(),
        )?;
        let message = merge_heads(&attended)?
            .apply(&self.output, state)?
            .permute(&[0, 2, 1])?
            .reshape([batch, width, rows, across])?
            .interpolate(
                Spatial2d::new(height, columns),
                Interpolation::new(InterpolateMode::Bilinear),
            )?;
        let update = Tensor::concat(&[hidden.clone(), message], 1)?
            .permute(&[0, 2, 3, 1])?
            .apply(&self.up, state)?
            .leaky_relu(0.01)?
            .apply(&self.down, state)?
            .apply(&self.output_norm, state)?
            .permute(&[0, 3, 1, 2])?;
        hidden.add(&update)
    }
}

/// The cosines and sines of each feature's angle, `[N, W]`, for rotating adjacent features in place across the whole projection, before it is split into heads, as `apply_rotary_pos_emb` with `rotate_half` over interleaved pairs turns it.
#[derive(Debug, Clone)]
struct Rotation {
    cos: Tensor,
    sin: Tensor,
}

impl Rotation {
    /// Returns `input`, `[B, N, W]`, each pair `(x₂ₚ, x₂ₚ₊₁)` turned by its angle and left in place.
    fn rotate(&self, input: &Tensor) -> SingeResult<Tensor> {
        input
            .mul(&self.cos)?
            .add(&swapped_pairs(input)?.mul(&self.sin)?)
    }
}

/// A refinement block: the finer residual map projected and added, two 3 × 3 convolutions around a batch norm and leaky `ReLU`, then a doubling.
#[derive(Debug, Clone)]
struct RefinementBlock {
    residual: Conv2d,
    first: Conv2d,
    norm: BatchNorm,
    second: Conv2d,
}

impl RefinementBlock {
    fn forward(
        &self,
        hidden: &Tensor,
        residual: &Tensor,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let mixed = residual
            .apply(&self.residual, state)?
            .add(hidden)?
            .apply(&self.first, state)?
            .apply(&self.norm, state)?
            .leaky_relu(0.01)?
            .apply(&self.second, state)?;
        doubled(&mixed)
    }
}

/// Returns `map` upsampled to twice its size, bilinearly without aligned corners.
fn doubled(map: &Tensor) -> SingeResult<Tensor> {
    let &[_, _, height, width] = map.shape().dimensions() else {
        return Err(SingeError::invalid_argument("a map, [B, C, H, W]"));
    };
    map.interpolate(
        Spatial2d::new(2 * height, 2 * width),
        Interpolation::new(InterpolateMode::Bilinear),
    )
}

/// Returns `map`'s windows of `kernel × kernel` points every `stride` points, `[cells, kernel², C]`, after `padding` zeros on each side, as `F.unfold` and the reference's reshape lay them out.
fn windows(map: &Tensor, kernel: i64, stride: i64, padding: i64) -> SingeResult<Tensor> {
    let map = if padding > 0 {
        map.pad(
            &[
                Padding::ZERO,
                Padding::symmetric(padding),
                Padding::symmetric(padding),
            ],
            0.0,
        )?
    } else {
        map.clone()
    };
    let channels = map.shape().dimensions()[0];
    let unfolded = map
        .unfold(1, kernel, stride)?
        .contiguous()?
        .unfold(2, kernel, stride)?;
    let &[_, rows, columns, _, _] = unfolded.shape().dimensions() else {
        return Err(SingeError::invalid_argument("unfolded windows"));
    };
    unfolded
        .permute(&[1, 2, 3, 4, 0])?
        .reshape([rows * columns, kernel * kernel, channels])
}

#[derive(Debug, Clone)]
pub struct EfficientLoftrLayers {
    stages: Vec<Vec<RepVgg>>,
    layers: Vec<[AggregatedAttention; 2]>,
    projection: Conv2d,
    refinement: Vec<RefinementBlock>,
    config: EfficientLoftrConfig,
}

impl EfficientLoftrLayers {
    pub fn create(schema: &mut Schema<'_>, config: &EfficientLoftrConfig) -> SingeResult<Self> {
        let stages = config
            .stages
            .iter()
            .enumerate()
            .map(|(stage, blocks)| {
                blocks
                    .iter()
                    .enumerate()
                    .map(|(index, block)| {
                        let mut scope =
                            schema.scope(format!("backbone.stages.{stage}.blocks.{index}"));
                        Ok(RepVgg {
                            wide: ConvNorm::create(
                                &mut scope,
                                ("conv1.conv", "conv1.norm"),
                                block.input,
                                block.output,
                                3,
                                block.stride,
                            )?,
                            narrow: ConvNorm::create(
                                &mut scope,
                                ("conv2.conv", "conv2.norm"),
                                block.input,
                                block.output,
                                1,
                                block.stride,
                            )?,
                            identity: (block.input == block.output && block.stride == 1)
                                .then(|| {
                                    scope.batch_norm("identity", block.input, PYTORCH_NORM_EPS)
                                })
                                .transpose()?,
                        })
                    })
                    .collect::<SingeResult<Vec<_>>>()
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let layers = (0..config.layers)
            .map(|index| {
                let mut scope = schema.scope(format!("transformer.layers.{index}"));
                Ok([
                    AggregatedAttention::create(&mut scope.scope("self_attention"), config)?,
                    AggregatedAttention::create(&mut scope.scope("cross_attention"), config)?,
                ])
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let dims = &config.fine_dims;
        let pointwise = Conv2dConfig::default();
        let padded = Conv2dConfig {
            padding: Padding2d::symmetric(1, 1),
            ..Conv2dConfig::default()
        };
        let refinement = (1..dims.len())
            .map(|index| {
                let (hidden, inner) = (dims[index], dims[index - 1]);
                let mut scope = schema.scope(format!("refinement.out_conv_layers.{}", index - 1));
                Ok(RefinementBlock {
                    residual: scope.conv2d("out_conv1", hidden, inner, 1, pointwise, false)?,
                    first: scope.conv2d("out_conv2", inner, inner, 3, padded, false)?,
                    norm: scope.batch_norm("batch_norm", inner, PYTORCH_NORM_EPS)?,
                    second: scope.conv2d("out_conv3", inner, hidden, 3, padded, false)?,
                })
            })
            .collect::<SingeResult<Vec<_>>>()?;
        Ok(Self {
            stages,
            layers,
            projection: schema.conv2d(
                "refinement.out_conv",
                dims[0],
                dims[0],
                1,
                pointwise,
                false,
            )?,
            refinement,
            config: config.clone(),
        })
    }

    /// Returns the rotary angles of an `rows × columns` grid of aggregated cells, interleaved pairs turning by the cell's row and column alternately, both counted from one, as `compute_embeddings` lays them out.
    fn angles(&self, rows: i64, columns: i64, like: &Tensor) -> SingeResult<Rotation> {
        let head = self.config.width / self.config.heads;
        let rotary = (head as f64 * self.config.rotary_factor) as i64;
        let frequencies = (0..rotary / 2)
            .map(|index| 1.0 / (self.config.theta as f32).powf((2 * index) as f32 / rotary as f32))
            .collect::<Vec<f32>>();
        let half = self.config.width / 2;
        let pairs = usize::try_from(half).map_err(|_| SingeError::out_of_range("rotary pairs"))?;
        let angles = (0..rows)
            .flat_map(|row| (0..columns).map(move |column| ((row + 1) as f32, (column + 1) as f32)))
            .flat_map(|(row, column)| {
                frequencies
                    .iter()
                    .flat_map(move |&frequency| [row * frequency, column * frequency])
                    .take(pairs)
            })
            .collect::<Vec<f32>>();
        // Each pair's angle for both of its features.
        let table = |values: Vec<f32>| -> SingeResult<Tensor> {
            let values = values
                .into_iter()
                .flat_map(|value| [value, value])
                .collect();
            Tensor::from_vec(
                like.device(),
                Shape::create([rows * columns, 2 * half])?,
                values,
            )?
            .cast(like.data_type())
        };
        Ok(Rotation {
            cos: table(angles.iter().map(|angle| angle.cos()).collect())?,
            sin: table(angles.iter().map(|angle| angle.sin()).collect())?,
        })
    }

    /// Returns each image's coarse features after the transformer, `[2, C, h, w]`, and the backbone's finer maps, coarsest first.
    fn features(&self, pixels: &Tensor, state: &ModelState) -> SingeResult<(Tensor, Vec<Tensor>)> {
        let mut hidden = pixels.narrow(1, 0, 1)?;
        let mut outputs = Vec::with_capacity(self.stages.len());
        for stage in &self.stages {
            for block in stage {
                hidden = hidden.apply(block, state)?;
            }

            outputs.push(hidden.clone());
        }

        let mut coarse = outputs
            .pop()
            .ok_or_else(|| SingeError::invalid_argument("a backbone without stages"))?;
        // The reference's backbone leaves the first stage's map out of the residuals.
        let finer = outputs.into_iter().skip(1).rev().collect::<Vec<_>>();
        let &[_, _, height, width] = coarse.shape().dimensions() else {
            return Err(SingeError::invalid_argument("coarse maps"));
        };
        let aggregation = self.config.aggregation;
        let angles = self.angles(
            (height - aggregation) / aggregation + 1,
            (width - aggregation) / aggregation + 1,
            &coarse,
        )?;
        let sizes = [self.config.heads, aggregation];
        for [within, across] in &self.layers {
            coarse = within.forward(&coarse, None, Some(&angles), sizes, state)?;
            let first = across.forward(
                &coarse.narrow(0, 0, 1)?,
                Some(&coarse.narrow(0, 1, 1)?),
                None,
                sizes,
                state,
            )?;
            let second =
                across.forward(&coarse.narrow(0, 1, 1)?, Some(&first), None, sizes, state)?;
            coarse = Tensor::concat(&[first, second], 0)?;
        }
        Ok((coarse, finer))
    }

    fn matched(&self, pixels: &Tensor, state: &ModelState) -> SingeResult<PairMatching> {
        let config = &self.config;
        let &[_, _, height, _] = pixels.shape().dimensions() else {
            return Err(SingeError::invalid_argument("image pair"));
        };
        let (coarse, finer) = self.features(pixels, state)?;
        let &[_, width, rows, columns] = coarse.shape().dimensions() else {
            return Err(SingeError::invalid_argument("coarse maps"));
        };
        let cells = rows * columns;
        let coarse_scale = (height / rows) as f32;
        // Coarse matching: a dual softmax of the scaled similarities, kept above the threshold, off the border, and at its row's and column's maximum.
        let flat = |image: i64| -> SingeResult<Tensor> {
            coarse
                .select(0, image)?
                .permute(&[1, 2, 0])?
                .reshape([cells, width])?
                .div_scalar((width as f64).sqrt())
        };
        let similarity = flat(0)?
            .matmul(&flat(1)?.transpose(0, 1)?)?
            .div_scalar(f64::from(config.temperature))?
            .cast(DataType::F32)?;
        let confidence = similarity.softmax(0)?.mul(&similarity.softmax(1)?)?;
        let inside = (0..cells)
            .map(|cell| {
                let (row, column) = (cell / columns, cell % columns);
                let border = config.border;
                f32::from(u8::from(
                    row >= border
                        && row < rows - border
                        && column >= border
                        && column < columns - border,
                ))
            })
            .collect::<Vec<f32>>();
        let device = pixels.device();
        let inside_rows = Tensor::from_vec(device, Shape::create([cells, 1])?, inside.clone())?;
        let inside_columns = Tensor::from_vec(device, Shape::create([1, cells])?, inside)?;
        let threshold = Tensor::from_vec(device, Shape::scalar(), vec![config.threshold])?;
        let kept = confidence
            .greater(&threshold)?
            .cast(DataType::F32)?
            .mul(&inside_rows)?
            .mul(&inside_columns)?
            .mul(
                &confidence
                    .equal(&confidence.max_reduce(&[1], true)?)?
                    .cast(DataType::F32)?,
            )?
            .mul(
                &confidence
                    .equal(&confidence.max_reduce(&[0], true)?)?
                    .cast(DataType::F32)?,
            )?;
        let masked = confidence.mul(&kept)?;
        // Each second-image cell's best first-image cell, and each first-image cell's best second-image cell.
        let (scores_by_second, firsts) = masked.topk(1, 0, true, true)?;
        let (scores_by_first, seconds) = masked.topk(1, 1, true, true)?;
        let (scores_by_second, firsts) =
            (scores_by_second.to_vec::<f32>()?, firsts.to_vec::<i64>()?);
        let (scores_by_first, seconds) =
            (scores_by_first.to_vec::<f32>()?, seconds.to_vec::<i64>()?);
        let cell_at = |index: i64| {
            [
                (index % columns) as f32 * coarse_scale,
                (index / columns) as f32 * coarse_scale,
            ]
        };
        let fine_scale = height as f32 / (rows as f32 * coarse_scale);
        // The first fine stage's offset, always the window's first point, as the reference's `gather` gives it.
        let offset = -(config.fine_kernel as f32) / 2.0 + 0.5;
        let last = cells - 1;
        let mut first_keypoints = Vec::new();
        let mut scores = Vec::new();
        for (slot, &score) in scores_by_second.iter().enumerate() {
            if score > 0.0 {
                let [x, y] = cell_at(firsts[slot]);
                first_keypoints.push([x + offset * fine_scale, y + offset * fine_scale]);
                scores.push(score);
            }
        }

        let refined = (0..scores_by_first.len())
            .filter(|&slot| scores_by_first[slot] > 0.0)
            .collect::<Vec<_>>();
        let mut second_keypoints = Vec::with_capacity(refined.len());
        if !refined.is_empty() {
            // The windows the reference reads at every slot, the last cell's where a slot has no match.
            let window = |scores: &[f32], cells: &[i64]| -> Vec<i64> {
                scores
                    .iter()
                    .zip(cells)
                    .map(|(&score, &cell)| if score > 0.0 { cell } else { last })
                    .collect()
            };
            let window_firsts = window(&scores_by_second, &firsts);
            let window_seconds = window(&scores_by_first, &seconds);
            let expectations = self.refine(
                &coarse,
                &finer,
                [&window_firsts, &window_seconds],
                &refined,
                state,
            )?;
            for (index, &slot) in refined.iter().enumerate() {
                let [x, y] = cell_at(seconds[slot]);
                let [dx, dy] = expectations[index];
                second_keypoints.push([
                    x + offset * fine_scale + dx * fine_scale,
                    y + offset * fine_scale + dy * fine_scale,
                ]);
            }
        }

        let count = first_keypoints.len().min(second_keypoints.len());
        Ok(PairMatching {
            keypoints: [first_keypoints, second_keypoints],
            matches: (0..).take(count).collect(),
            scores: scores.into_iter().take(count).collect(),
        })
    }

    /// Returns the second fine stage's offset at each of the `wanted` slots, whose windows are the first image's cells `firsts` and the second's `seconds`, given at every slot: the expectation over the 3 × 3 neighbourhood of the first stage's best point, by the softmax of the sliced features' scores over the regression temperature.
    fn refine(
        &self,
        coarse: &Tensor,
        finer: &[Tensor],
        [firsts, seconds]: [&[i64]; 2],
        wanted: &[usize],
        state: &ModelState,
    ) -> SingeResult<Vec<[f32; 2]>> {
        let config = &self.config;
        let width = coarse.shape().dimensions()[1];
        let mut fine = doubled(
            &coarse
                .div_scalar((width as f64).sqrt())?
                .apply(&self.projection, state)?,
        )?;
        for (block, residual) in self.refinement.iter().zip(finer) {
            fine = block.forward(&fine, residual, state)?;
        }

        let kernel = config.fine_kernel;
        let stride = fine.shape().dimensions()[2] / coarse.shape().dimensions()[2];
        let device = coarse.device();
        let picked = |windows: Tensor, cells: &[i64]| -> SingeResult<Tensor> {
            let count =
                i64::try_from(cells.len()).map_err(|_| SingeError::out_of_range("matches"))?;
            windows.index_select(
                0,
                &Tensor::from_vec(device, Shape::create([count])?, cells.to_vec())?,
            )
        };
        let first = picked(windows(&fine.select(0, 0)?, kernel, stride, 0)?, firsts)?
            .cast(DataType::F32)?;
        let second = picked(
            windows(&fine.select(0, 1)?, kernel + 2, stride, 1)?,
            seconds,
        )?
        .cast(DataType::F32)?;
        let channels = first.shape().dimensions()[2];
        let wide = channels - config.slice;
        let count = first.shape().dimensions()[0];
        let outer = kernel + 2;
        let first_wide = first.narrow(2, 0, wide)?.div_scalar((wide as f64).sqrt())?;
        let second_wide = second
            .narrow(2, 0, wide)?
            .div_scalar((wide as f64).sqrt())?;
        let confidence = first_wide.matmul(&second_wide.transpose(1, 2)?)?;
        // Over the slots and the first window's points, as the reference's dims 1 and 2 of `[B, K, 64, 100]` are.
        let dual = confidence.softmax(0)?.mul(&confidence.softmax(1)?)?;
        let inner = dual
            .reshape([count, kernel * kernel, outer, outer])?
            .narrow(2, 1, kernel)?
            .narrow(3, 1, kernel)?
            .reshape([count, kernel.pow(4)])?;
        let wanted_slots =
            i64::try_from(wanted.len()).map_err(|_| SingeError::out_of_range("matches"))?;
        let wanted_index = Tensor::from_vec(
            device,
            Shape::create([wanted_slots])?,
            wanted
                .iter()
                .map(|&slot| i64::try_from(slot).map_err(|_| SingeError::out_of_range("matches")))
                .collect::<SingeResult<Vec<_>>>()?,
        )?;
        let (_, best) = inner
            .index_select(0, &wanted_index)?
            .topk(1, 1, true, true)?;
        let best = best.to_vec::<i64>()?;
        let (first, second) = (
            first.index_select(0, &wanted_index)?,
            second.index_select(0, &wanted_index)?,
        );
        let window = kernel * kernel;
        let scores = first
            .narrow(2, wide, config.slice)?
            .matmul(
                &second
                    .narrow(2, wide, config.slice)?
                    .div_scalar((config.slice as f64).sqrt())?
                    .transpose(1, 2)?,
            )?
            .reshape([wanted_slots * window * outer * outer])?;
        // Each match's 3 × 3 neighbourhood in the second window, its indices wrapping past the window's edge as Python's do.
        let picks = best
            .iter()
            .zip(0_i64..)
            .flat_map(|(&best, index)| {
                let (point, inner_point) = (best / window, best % window);
                let (row, column) = (inner_point / kernel, inner_point % kernel);
                let base = index * window * outer * outer + point * outer * outer;
                (0..3).flat_map(move |down| {
                    (0..3).map(move |across| {
                        base + (row + down - 1).rem_euclid(outer) * outer
                            + (column + across - 1).rem_euclid(outer)
                    })
                })
            })
            .collect::<Vec<i64>>();
        let picked_count =
            i64::try_from(picks.len()).map_err(|_| SingeError::out_of_range("matches"))?;
        let values = scores
            .index_select(
                0,
                &Tensor::from_vec(device, Shape::create([picked_count])?, picks)?,
            )?
            .to_vec::<f32>()?;
        Ok(values
            .as_chunks::<9>()
            .0
            .iter()
            .map(|neighbourhood| {
                let scaled = neighbourhood
                    .iter()
                    .map(|value| value / config.regress_temperature)
                    .collect::<Vec<_>>();
                let most = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exponents = scaled
                    .iter()
                    .map(|value| (value - most).exp())
                    .collect::<Vec<_>>();
                let total = exponents.iter().sum::<f32>();
                // `spatial_expectation2d`: columns give the horizontal offset, rows the vertical, each from -1 to 1.
                let (mut x, mut y) = (0.0, 0.0);
                for (index, exponent) in exponents.iter().enumerate() {
                    let weight = exponent / total;
                    x += ((index % 3) as f32 - 1.0) * weight;
                    y += ((index / 3) as f32 - 1.0) * weight;
                }
                [x, y]
            })
            .collect())
    }
}

impl MatchingLayers for EfficientLoftrLayers {
    fn match_pair(&self, pixels: &Tensor, state: &ModelState) -> SingeResult<PairMatching> {
        // Matches are thresholded and argmaxed through a deep backbone, which TF32 convolutions round apart from PyTorch's: full precision.
        with_convolution_precision(ComputePrecision::Full, || self.matched(pixels, state))
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "efficientloftr",
    formats: &[(Format::Transformers, &["efficientloftr"])],
    loaders: &[HeadLoader::KeypointMatching(load_matcher)],
};

pub fn load_matcher(loader: &mut Loader<'_>) -> Result<KeypointMatcher> {
    let config: EfficientLoftrConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        EfficientLoftrLayers::create(schema, &config)
            .map(|layers| Boxed::<dyn MatchingLayers>(Box::new(layers)))
    })?;
    let model = loader.load_whole(&model, &transformers::rules())?;
    Ok(KeypointMatcher::new(model))
}
