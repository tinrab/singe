// LightGlue: SuperPoint's keypoints in two images, each described anew by layers of self-attention with rotary keypoint positions and cross-attention to the other image, pruned of keypoints confidently unmatchable, until enough are confident, then matched by a dual softmax over their similarities.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult, Shape, Tensor,
    nn::{Activation, LayerNorm, Linear, Model, ModelState, StateBuilder},
};

use crate::{
    error::Result,
    format::Loader,
    head::{
        HeadLoader, KeypointLayers, KeypointMatcher, MatchingLayers, PairMatches, extract_keypoints,
    },
    model::{
        ModelEntry,
        vision::{
            host,
            superpoint::{self, SuperPointConfig, SuperPointLayers},
        },
    },
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LightGlueConfig {
    pub detector: SuperPointConfig,
    pub width: i64,
    pub heads: i64,
    pub inner: i64,
    pub layers: usize,
    pub activation: Activation,
    // The least score a match keeps.
    pub filter_threshold: f32,
    // The share of confident keypoints at which matching stops early.
    pub depth_confidence: f32,
    // One less the least matchability a confident keypoint keeps.
    pub width_confidence: f32,
}

#[derive(Debug, Clone)]
struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    heads: i64,
}

// Rotary angles' cosines and sines, `[1, 1, N, head]`, each repeated for a pair of features.
struct Positions {
    cos: Tensor,
    sin: Tensor,
}

// Each pair of features `(a, b)` as `(-b, a)`, as LightGlue's `rotate_half` interleaves them.
fn rotate_pairs(input: &Tensor) -> SingeResult<Tensor> {
    let dimensions = input.shape().dimensions().to_vec();
    let last = dimensions[dimensions.len() - 1];
    let mut pairs = dimensions.clone();
    pairs.pop();
    pairs.extend([last / 2, 2]);
    let pairs = input.reshape(pairs)?;
    let (even, odd) = (pairs.narrow(-1, 0, 1)?, pairs.narrow(-1, 1, 1)?);
    Tensor::concat(&[odd.neg()?, even], -1)?.reshape(dimensions)
}

impl Attention {
    fn forward(
        &self,
        hidden: &Tensor,
        context: &Tensor,
        positions: Option<&Positions>,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let dimensions = hidden.shape().dimensions();
        let (length, width) = (dimensions[1], dimensions[2]);
        let head = width / self.heads;
        let split = |tensor: Tensor| -> SingeResult<Tensor> {
            let length = tensor.shape().dimensions()[1];
            tensor
                .reshape([1, length, self.heads, head])?
                .transpose(1, 2)
        };
        let mut query = split(hidden.apply(&self.query, state)?)?;
        let mut key = split(context.apply(&self.key, state)?)?;
        let value = split(context.apply(&self.value, state)?)?;
        if let Some(positions) = positions {
            let rotate = |tensor: &Tensor| -> SingeResult<Tensor> {
                tensor
                    .mul(&positions.cos)?
                    .add(&rotate_pairs(tensor)?.mul(&positions.sin)?)
            };
            query = rotate(&query)?;
            key = rotate(&key)?;
        }
        query
            .scaled_dot_product_attention(&key, &value, None, false)?
            .transpose(1, 2)?
            .reshape([1, length, width])?
            .apply(&self.output, state)
    }
}

// The feed-forward over a keypoint's descriptor beside what attention gave it.
#[derive(Debug, Clone)]
struct Update {
    up: Linear,
    norm: LayerNorm,
    down: Linear,
    activation: Activation,
}

impl Update {
    fn forward(
        &self,
        hidden: &Tensor,
        message: &Tensor,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        Tensor::concat(&[hidden.clone(), message.clone()], 2)?
            .apply(&self.up, state)?
            .apply(&self.norm, state)?
            .apply(&self.activation, state)?
            .apply(&self.down, state)?
            .add(hidden)
    }
}

#[derive(Debug, Clone)]
struct Block {
    self_attention: Attention,
    self_update: Update,
    cross_attention: Attention,
    cross_update: Update,
}

#[derive(Debug, Clone)]
struct Assignment {
    projection: Linear,
    matchability: Linear,
}

#[derive(Debug, Clone)]
pub struct LightGlueLayers {
    detector: SuperPointLayers,
    positions: Linear,
    blocks: Vec<Block>,
    assignments: Vec<Assignment>,
    confidences: Vec<Linear>,
    config: LightGlueConfig,
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

// `log(sigmoid(value))`, computed without overflow.
fn log_sigmoid(value: f32) -> f32 {
    value.min(0.0) - (-value.abs()).exp().ln_1p()
}

// Each row's log-softmax, of a row-major `rows × columns` matrix.
fn log_softmax_rows(values: &[f32], columns: usize) -> Vec<f32> {
    values
        .chunks_exact(columns)
        .flat_map(|row| {
            let most = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let total = row
                .iter()
                .map(|value| (value - most).exp())
                .sum::<f32>()
                .ln();
            row.iter().map(move |value| value - most - total)
        })
        .collect()
}

// The index and value of a slice's greatest value, the first of equals, as `max` gives them.
fn best(values: impl Iterator<Item = f32>) -> (usize, f32) {
    values
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |found, (index, value)| {
            if value > found.1 {
                (index, value)
            } else {
                found
            }
        })
}

// One image's keypoints as LightGlue carries them through its layers.
struct Side {
    descriptors: Tensor,
    positions: Positions,
    // Each kept keypoint's index among the image's keypoints.
    indices: Vec<usize>,
}

impl Side {
    fn keep(&mut self, kept: &[usize]) -> SingeResult<()> {
        let device = self.descriptors.device();
        let count =
            i64::try_from(kept.len()).map_err(|_| singe::Error::out_of_range("keypoints"))?;
        let rows = kept
            .iter()
            .map(|&index| i64::try_from(index).map_err(|_| singe::Error::out_of_range("keypoints")))
            .collect::<SingeResult<Vec<_>>>()?;
        let rows = Tensor::from_vec(device, Shape::create([1, count, 1])?, rows)?;
        let gather = |tensor: &Tensor, dim: i64| -> SingeResult<Tensor> {
            let mut shape = tensor.shape().dimensions().to_vec();
            let axis = usize::try_from(dim).map_err(|_| singe::Error::out_of_range("axis"))?;
            shape[axis] = count;
            let mut index_shape = vec![1; shape.len()];
            index_shape[axis] = count;
            tensor.gather(
                dim,
                &rows.reshape(index_shape)?.expand(Shape::create(shape)?)?,
            )
        };
        self.descriptors = gather(&self.descriptors, 1)?;
        self.positions = Positions {
            cos: gather(&self.positions.cos, 2)?,
            sin: gather(&self.positions.sin, 2)?,
        };
        self.indices = kept.iter().map(|&index| self.indices[index]).collect();
        Ok(())
    }
}

impl LightGlueLayers {
    // The rotary angles of keypoints at `[x, y]` pixels of a `height × width` image, centered and scaled by half the longer side.
    #[expect(clippy::cast_precision_loss, reason = "image sides are far below 2^24")]
    fn positions(
        &self,
        keypoints: &[[f32; 2]],
        (height, width): (i64, i64),
        device: singe::Device,
        state: &ModelState,
    ) -> SingeResult<Positions> {
        let scale = height.max(width) as f32 / 2.0;
        let (shift_x, shift_y) = (width as f32 / 2.0, height as f32 / 2.0);
        let (side_x, side_y) = (width as f32, height as f32);
        // SuperPoint's coordinates relative to the image, back at its pixels, as LightGlue's forward reads them.
        let normalized = keypoints
            .iter()
            .map(|&[x, y]| [x / side_x * side_x, y / side_y * side_y])
            .flat_map(|[x, y]| [(x - shift_x) / scale, (y - shift_y) / scale])
            .collect::<Vec<_>>();
        let count =
            i64::try_from(keypoints.len()).map_err(|_| singe::Error::out_of_range("keypoints"))?;
        let angles = Tensor::from_vec(device, Shape::create([1, count, 2])?, normalized)?
            .apply(&self.positions, state)?;
        let half = angles.shape().dimensions()[2];
        // Each angle twice, as `repeat_interleave(2)` lays them out.
        let angles = angles
            .unsqueeze(3)?
            .expand(Shape::create([1, count, half, 2])?)?
            .reshape([1, 1, count, half * 2])?;
        Ok(Positions {
            cos: angles.cos()?,
            sin: angles.sin()?,
        })
    }

    // Matches the two sides with layer `layer`'s assignment: each first keypoint's match in the second image and its score.
    fn assign(
        &self,
        layer: usize,
        sides: &[Side; 2],
        state: &ModelState,
    ) -> SingeResult<(Vec<i64>, Vec<f32>)> {
        let assignment = &self.assignments[layer];
        let width = self.config.width;
        #[expect(clippy::cast_precision_loss, reason = "descriptor widths are small")]
        let scale = (width as f64).powf(0.25).recip();
        let projected = |side: &Side| -> SingeResult<Tensor> {
            side.descriptors
                .apply(&assignment.projection, state)?
                .mul_scalar(scale)
        };
        let similarity =
            host(&projected(&sides[0])?.matmul(&projected(&sides[1])?.transpose(1, 2)?)?)?;
        let matchability =
            |side: &Side| host(&side.descriptors.apply(&assignment.matchability, state)?);
        let (first, second) = (matchability(&sides[0])?, matchability(&sides[1])?);
        let (rows, columns) = (first.len(), second.len());

        // Each pair's log-probability under both directions' softmax and both keypoints' matchability, as `sigmoid_log_double_softmax` gives it.
        let by_rows = log_softmax_rows(&similarity, columns);
        let transposed = (0..columns)
            .flat_map(|column| (0..rows).map(move |row| (row, column)))
            .map(|(row, column)| similarity[row * columns + column])
            .collect::<Vec<_>>();
        let by_columns = log_softmax_rows(&transposed, rows);
        let score = |row: usize, column: usize| {
            by_rows[row * columns + column]
                + by_columns[column * rows + row]
                + (log_sigmoid(first[row]) + log_sigmoid(second[column]))
        };

        // Mutual best matches whose probability passes the threshold, as `get_matches_from_scores` keeps them.
        let row_best = (0..rows)
            .map(|row| best((0..columns).map(|column| score(row, column))))
            .collect::<Vec<_>>();
        let column_best = (0..columns)
            .map(|column| best((0..rows).map(|row| score(row, column))).0)
            .collect::<Vec<_>>();
        let mut matches = vec![-1; rows];
        let mut scores = vec![0.0; rows];
        for (row, &(column, value)) in row_best.iter().enumerate() {
            if column_best.get(column) == Some(&row) {
                scores[row] = value.exp();
                if scores[row] > self.config.filter_threshold {
                    matches[row] = i64::try_from(column)
                        .map_err(|_| singe::Error::out_of_range("keypoints"))?;
                }
            }
        }
        Ok((matches, scores))
    }
}

impl MatchingLayers for LightGlueLayers {
    fn match_pair(&self, pixels: &Tensor, state: &ModelState) -> SingeResult<PairMatches> {
        let dimensions = pixels.shape().dimensions();
        let image_size = (dimensions[2], dimensions[3]);
        let device = pixels.device();
        let detected = (0..2)
            .map(|image| {
                let (scores, descriptors) =
                    self.detector.maps(&pixels.narrow(0, image, 1)?, state)?;
                Ok((scores, descriptors))
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let found = detected
            .iter()
            .map(|(scores, descriptors)| {
                extract_keypoints(scores, descriptors, &self.config.detector.keypoints)
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let keypoints = [found[0][0].keypoints.clone(), found[1][0].keypoints.clone()];
        let total = keypoints[0].len() + keypoints[1].len();
        if keypoints.iter().any(Vec::is_empty) {
            return Ok(PairMatches {
                matches: vec![-1; keypoints[0].len()],
                scores: vec![0.0; keypoints[0].len()],
                keypoints,
            });
        }

        let side = |index: usize| -> SingeResult<Side> {
            let points = &found[index][0];
            let count = i64::try_from(points.keypoints.len())
                .map_err(|_| singe::Error::out_of_range("keypoints"))?;
            let values = points
                .descriptors
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>();
            let width = self.config.width;
            Ok(Side {
                descriptors: Tensor::from_vec(device, Shape::create([1, count, width])?, values)?
                    .cast(pixels.data_type())?,
                positions: self.positions(&points.keypoints, image_size, device, state)?,
                indices: (0..points.keypoints.len()).collect(),
            })
        };
        let mut sides = [side(0)?, side(1)?];
        let final_layer = self.config.layers - 1;
        for (layer, block) in self.blocks.iter().enumerate() {
            let updated = sides
                .iter()
                .map(|side| {
                    let message = block.self_attention.forward(
                        &side.descriptors,
                        &side.descriptors,
                        Some(&side.positions),
                        state,
                    )?;
                    block
                        .self_update
                        .forward(&side.descriptors, &message, state)
                })
                .collect::<SingeResult<Vec<_>>>()?;
            for (index, side) in sides.iter_mut().enumerate() {
                let other = &updated[1 - index];
                let message = block
                    .cross_attention
                    .forward(&updated[index], other, None, state)?;
                side.descriptors = block
                    .cross_update
                    .forward(&updated[index], &message, state)?;
            }

            if layer == final_layer {
                break;
            }

            // Stops when enough keypoints, counting the pruned as confident, are confident.
            let threshold = confidence_threshold(layer, self.config.layers);
            let confidences = sides
                .iter()
                .map(|side| {
                    Ok(
                        host(&side.descriptors.apply(&self.confidences[layer], state)?)?
                            .into_iter()
                            .map(sigmoid)
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<SingeResult<Vec<_>>>()?;
            let unsure = confidences
                .iter()
                .flatten()
                .filter(|&&value| value < threshold)
                .count();
            #[expect(clippy::cast_precision_loss, reason = "keypoint counts are small")]
            let confident = 1.0 - unsure as f32 / total as f32;
            if confident > self.config.depth_confidence {
                let (matches, scores) = self.assign(layer, &sides, state)?;
                return Ok(unpruned(&sides, &keypoints, &matches, &scores));
            }

            // Keeps the keypoints that may match, and those not yet confident.
            for (side, confidences) in sides.iter_mut().zip(&confidences) {
                let matchability = host(
                    &side
                        .descriptors
                        .apply(&self.assignments[layer].matchability, state)?,
                )?;
                let least = 1.0 - self.config.width_confidence;
                let kept = (0..side.indices.len())
                    .filter(|&index| {
                        sigmoid(matchability[index]) > least || confidences[index] <= threshold
                    })
                    .collect::<Vec<_>>();
                side.keep(&kept)?;
            }
        }

        let (matches, scores) = self.assign(final_layer, &sides, state)?;
        Ok(unpruned(&sides, &keypoints, &matches, &scores))
    }
}

// The confidence a keypoint needs after layer `layer` of `layers`, falling from 0.9 toward 0.8, as `_get_confidence_threshold` gives it.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "layer counts are small, and the comparison is in `F32`"
)]
#[expect(
    clippy::suboptimal_flops,
    reason = "rounds each product as the reference does, which the verifications compare with"
)]
fn confidence_threshold(layer: usize, layers: usize) -> f32 {
    (0.8 + 0.1 * (-4.0 * layer as f64 / layers as f64).exp()).clamp(0.0, 1.0) as f32
}

// The kept keypoints' matches and scores at every first keypoint, through the kept indices, with no match and no score for the pruned.
fn unpruned(
    sides: &[Side; 2],
    keypoints: &[Vec<[f32; 2]>; 2],
    matches: &[i64],
    scores: &[f32],
) -> PairMatches {
    let count = keypoints[0].len();
    let mut all_matches = vec![-1; count];
    let mut all_scores = vec![0.0; count];
    for (kept, (&found, &score)) in matches.iter().zip(scores).enumerate() {
        let index = sides[0].indices[kept];
        all_scores[index] = score;
        all_matches[index] = usize::try_from(found)
            .ok()
            .and_then(|found| sides[1].indices.get(found))
            .and_then(|&found| i64::try_from(found).ok())
            .unwrap_or(-1);
    }
    PairMatches {
        keypoints: keypoints.clone(),
        matches: all_matches,
        scores: all_scores,
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "lightglue",
    formats: &[(Format::Transformers, &["lightglue"])],
    heads: &[HeadLoader::KeypointMatching(load_matcher)],
};

fn attention(schema: &mut Schema<'_>, config: &LightGlueConfig) -> SingeResult<Attention> {
    let width = config.width;
    Ok(Attention {
        query: schema.linear("query", width, width, true)?,
        key: schema.linear("key", width, width, true)?,
        value: schema.linear("value", width, width, true)?,
        output: schema.linear("output", width, width, true)?,
        heads: config.heads,
    })
}

fn update(schema: &mut Schema<'_>, config: &LightGlueConfig) -> SingeResult<Update> {
    Ok(Update {
        up: schema.linear("up", config.inner, config.inner, true)?,
        norm: schema.layer_norm("norm", config.inner, 1e-5)?,
        down: schema.linear("down", config.inner, config.width, true)?,
        activation: config.activation,
    })
}

pub fn load_matcher(loader: &mut Loader<'_>) -> Result<KeypointMatcher> {
    let config: LightGlueConfig = loader.config(transformers::config)?;
    let mut builder = StateBuilder::new();
    let layers = {
        let mut schema = Schema::root(&mut builder, loader.options().data_type);
        let detector = superpoint::layers(&mut schema.scope("detector"), &config.detector)?;
        let head = config.width / config.heads;
        let positions = schema.linear("positions", 2, head / 2, false)?;
        let blocks = (0..config.layers)
            .map(|index| {
                let mut schema = schema.scope(format!("blocks.{index}"));
                Ok(Block {
                    self_attention: attention(&mut schema.scope("self_attention"), &config)?,
                    self_update: update(&mut schema.scope("self_update"), &config)?,
                    cross_attention: attention(&mut schema.scope("cross_attention"), &config)?,
                    cross_update: update(&mut schema.scope("cross_update"), &config)?,
                })
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let assignments = (0..config.layers)
            .map(|index| {
                let mut schema = schema.scope(format!("assignments.{index}"));
                Ok(Assignment {
                    projection: schema.linear("projection", config.width, config.width, true)?,
                    matchability: schema.linear("matchability", config.width, 1, true)?,
                })
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let confidences = (0..config.layers - 1)
            .map(|index| schema.linear(&format!("confidences.{index}"), config.width, 1, true))
            .collect::<SingeResult<Vec<_>>>()?;
        LightGlueLayers {
            detector,
            positions,
            blocks,
            assignments,
            confidences,
            config,
        }
    };
    let layers: Box<dyn MatchingLayers> = Box::new(layers);
    let schema = Model::new(layers, builder.finish()?);
    let model = loader.load("model", &schema, &transformers::rules())?;
    loader.finish()?;
    Ok(KeypointMatcher::new(model))
}
