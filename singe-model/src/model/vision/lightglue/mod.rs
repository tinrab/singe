//! `LightGlue`: `SuperPoint`'s keypoints in two images, each described anew by layers of self-attention with rotary keypoint positions and cross-attention to the other image, pruned of keypoints confidently unmatchable, until enough are confident, then matched by a dual softmax over their similarities.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Device, Error as SingeError, Result as SingeResult, Shape, Tensor,
    nn::{Activation, LayerNorm, Linear, ModelState, RotaryAngles, RotaryLayout},
    shape::spatial::{Spatial2d, Spatial2dI64},
};

use crate::{
    assembly::{Attention, AttentionSpec, PYTORCH_NORM_EPS, merge_heads, split_heads},
    error::Result,
    format::Loader,
    head::{
        Boxed, HeadLoader, KeypointLayers, KeypointMatcher, KeypointSet, MatchingLayers,
        PairMatching, extract_keypoints,
    },
    model::{
        ModelEntry,
        vision::superpoint::{SuperPointConfig, SuperPointLayers},
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
    pub layers: i64,
    pub activation: Activation,
    /// The score a match must exceed to be kept.
    pub filter_threshold: f32,
    /// The share of confident keypoints above which matching stops early.
    pub depth_confidence: f32,
    /// One minus the matchability a confident keypoint must exceed to escape pruning.
    pub width_confidence: f32,
}

#[derive(Debug, Clone)]
struct SelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    heads: i64,
}

impl SelfAttention {
    fn create(schema: &mut Schema<'_>, config: &LightGlueConfig) -> SingeResult<Self> {
        let [query, key, value, output] = schema.attention_projections(config.width, [true; 4])?;
        Ok(Self {
            query,
            key,
            value,
            output,
            heads: config.heads,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        angles: &RotaryAngles,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let &[batch, length, width] = hidden.shape().dimensions() else {
            return Err(SingeError::invalid_argument("keypoint descriptors"));
        };
        let rotated = |projection: &Linear| -> SingeResult<Tensor> {
            let heads = hidden.apply(projection, state)?.reshape([
                batch,
                length,
                self.heads,
                width / self.heads,
            ])?;
            angles.rotate(&heads)?.transpose(1, 2)
        };
        let value = split_heads(&hidden.apply(&self.value, state)?, self.heads)?;
        let attended = rotated(&self.query)?.scaled_dot_product_attention(
            &rotated(&self.key)?,
            &value,
            None,
            false,
        )?;
        merge_heads(&attended)?.apply(&self.output, state)
    }
}

/// The feed-forward over a keypoint's descriptor beside what attention gave it.
#[derive(Debug, Clone)]
struct Update {
    up: Linear,
    norm: LayerNorm,
    down: Linear,
    activation: Activation,
}

impl Update {
    fn create(schema: &mut Schema<'_>, config: &LightGlueConfig) -> SingeResult<Self> {
        Ok(Self {
            up: schema.linear("up", config.inner, config.inner, true)?,
            norm: schema.layer_norm("norm", config.inner, PYTORCH_NORM_EPS)?,
            down: schema.linear("down", config.inner, config.width, true)?,
            activation: config.activation,
        })
    }

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
    self_attention: SelfAttention,
    self_update: Update,
    cross_attention: Attention,
    cross_update: Update,
}

impl Block {
    fn create(schema: &mut Schema<'_>, config: &LightGlueConfig) -> SingeResult<Self> {
        Ok(Self {
            self_attention: SelfAttention::create(&mut schema.scope("self_attention"), config)?,
            self_update: Update::create(&mut schema.scope("self_update"), config)?,
            cross_attention: Attention::create(
                &mut schema.scope("cross_attention"),
                AttentionSpec::square(config.width, config.heads),
            )?,
            cross_update: Update::create(&mut schema.scope("cross_update"), config)?,
        })
    }

    fn forward(&self, sides: &mut [Side; 2], state: &ModelState) -> SingeResult<()> {
        let updated = sides
            .iter()
            .map(|side| {
                let message =
                    self.self_attention
                        .forward(&side.descriptors, &side.angles()?, state)?;
                self.self_update.forward(&side.descriptors, &message, state)
            })
            .collect::<SingeResult<Vec<_>>>()?;
        for (index, side) in sides.iter_mut().enumerate() {
            let other = &updated[1 - index];
            let message = self
                .cross_attention
                .forward(&updated[index], other, other, state)?;
            side.descriptors = self
                .cross_update
                .forward(&updated[index], &message, state)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct Assignment {
    projection: Linear,
    matchability: Linear,
}

impl Assignment {
    fn create(schema: &mut Schema<'_>, config: &LightGlueConfig) -> SingeResult<Self> {
        Ok(Self {
            projection: schema.linear("projection", config.width, config.width, true)?,
            matchability: schema.linear("matchability", config.width, 1, true)?,
        })
    }
}

/// One image's keypoints as `LightGlue` carries them through its layers.
struct Side {
    descriptors: Tensor,
    cos: Tensor,
    sin: Tensor,
    /// Each kept keypoint's index among the image's keypoints.
    indices: Vec<usize>,
}

impl Side {
    fn angles(&self) -> SingeResult<RotaryAngles> {
        RotaryAngles::create(&self.cos, &self.sin, RotaryLayout::Interleaved)
    }

    fn keep(&mut self, kept: &[usize]) -> SingeResult<()> {
        let device = self.descriptors.device();
        let count = i64::try_from(kept.len()).map_err(|_| SingeError::out_of_range("keypoints"))?;
        let rows = kept
            .iter()
            .map(|&index| i64::try_from(index).map_err(|_| SingeError::out_of_range("keypoints")))
            .collect::<SingeResult<Vec<_>>>()?;
        let rows = Tensor::from_vec(device, Shape::create([count])?, rows)?;
        let gather = |tensor: &Tensor, axis: usize| -> SingeResult<Tensor> {
            let dim = i64::try_from(axis).map_err(|_| SingeError::out_of_range("axis"))?;
            let mut shape = tensor.shape().dimensions().to_vec();
            shape[axis] = count;
            let mut index_shape = vec![1; shape.len()];
            index_shape[axis] = count;
            tensor.gather(
                dim,
                &rows.reshape(index_shape)?.expand(Shape::create(shape)?)?,
            )
        };
        self.descriptors = gather(&self.descriptors, 1)?;
        self.cos = gather(&self.cos, 0)?;
        self.sin = gather(&self.sin, 0)?;
        self.indices = kept.iter().map(|&index| self.indices[index]).collect();
        Ok(())
    }
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

impl LightGlueLayers {
    pub fn create(schema: &mut Schema<'_>, config: &LightGlueConfig) -> SingeResult<Self> {
        let detector = SuperPointLayers::create(&mut schema.scope("detector"), &config.detector)?;
        let head = config.width / config.heads;
        let positions = schema.linear("positions", 2, head / 2, false)?;
        let blocks = (0..config.layers)
            .map(|index| Block::create(&mut schema.scope(format!("blocks.{index}")), config))
            .collect::<SingeResult<Vec<_>>>()?;
        let assignments = (0..config.layers)
            .map(|index| {
                Assignment::create(&mut schema.scope(format!("assignments.{index}")), config)
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let confidences = (0..config.layers - 1)
            .map(|index| schema.linear(&format!("confidences.{index}"), config.width, 1, true))
            .collect::<SingeResult<Vec<_>>>()?;
        Ok(Self {
            detector,
            positions,
            blocks,
            assignments,
            confidences,
            config: config.clone(),
        })
    }

    fn detect(&self, pixels: &Tensor, image: i64, state: &ModelState) -> SingeResult<KeypointSet> {
        let (scores, descriptors) = self.detector.maps(&pixels.narrow(0, image, 1)?, state)?;
        extract_keypoints(&scores, &descriptors, &self.config.detector.keypoints)?
            .into_iter()
            .next()
            .ok_or_else(|| SingeError::invalid_argument("keypoint maps of no image"))
    }

    /// Returns the cosines and sines of the rotary angles of keypoints at `[x, y]` pixels of a `height × width` image, centered and scaled by half the longer side.
    fn positions(
        &self,
        keypoints: &[[f32; 2]],
        Spatial2d { height, width }: Spatial2dI64,
        device: Device,
        state: &ModelState,
    ) -> SingeResult<(Tensor, Tensor)> {
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
            i64::try_from(keypoints.len()).map_err(|_| SingeError::out_of_range("keypoints"))?;
        let angles = Tensor::from_vec(device, Shape::create([1, count, 2])?, normalized)?
            .apply(&self.positions, state)?
            .select(0, 0)?;
        Ok((angles.cos()?, angles.sin()?))
    }

    fn side(
        &self,
        points: &KeypointSet,
        image_size: Spatial2dI64,
        pixels: &Tensor,
        state: &ModelState,
    ) -> SingeResult<Side> {
        let device = pixels.device();
        let count = i64::try_from(points.keypoints.len())
            .map_err(|_| SingeError::out_of_range("keypoints"))?;
        let values = points
            .descriptors
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let (cos, sin) = self.positions(&points.keypoints, image_size, device, state)?;
        Ok(Side {
            descriptors: Tensor::from_vec(
                device,
                Shape::create([1, count, self.config.width])?,
                values,
            )?
            .cast(pixels.data_type())?,
            cos,
            sin,
            indices: (0..points.keypoints.len()).collect(),
        })
    }

    fn match_sides(
        &self,
        sides: &mut [Side; 2],
        total: usize,
        state: &ModelState,
    ) -> SingeResult<(Vec<i64>, Vec<f32>)> {
        let layers = self.blocks.len();
        for (layer, block) in self.blocks.iter().enumerate() {
            block.forward(sides, state)?;
            if layer + 1 == layers {
                break;
            }

            // Stops when enough keypoints, counting the pruned as confident, are confident.
            let threshold = confidence_threshold(layer, layers);
            let confidences = self.confidences(layer, sides, state)?;
            let unsure = confidences
                .iter()
                .flatten()
                .filter(|&&value| value < threshold)
                .count();
            let confident = 1.0 - unsure as f32 / total as f32;
            if confident > self.config.depth_confidence {
                return self.assign(layer, sides, state);
            }

            self.prune(layer, sides, &confidences, threshold, state)?;
        }

        self.assign(layers - 1, sides, state)
    }

    fn confidences(
        &self,
        layer: usize,
        sides: &[Side; 2],
        state: &ModelState,
    ) -> SingeResult<Vec<Vec<f32>>> {
        sides
            .iter()
            .map(|side| {
                Ok(side
                    .descriptors
                    .apply(&self.confidences[layer], state)?
                    .cast(DataType::F32)?
                    .to_vec::<f32>()?
                    .into_iter()
                    .map(sigmoid)
                    .collect::<Vec<_>>())
            })
            .collect()
    }

    /// Keeps the keypoints that may match, and those not yet confident.
    fn prune(
        &self,
        layer: usize,
        sides: &mut [Side; 2],
        confidences: &[Vec<f32>],
        threshold: f32,
        state: &ModelState,
    ) -> SingeResult<()> {
        for (side, confidences) in sides.iter_mut().zip(confidences) {
            let matchability = side
                .descriptors
                .apply(&self.assignments[layer].matchability, state)?
                .cast(DataType::F32)?
                .to_vec::<f32>()?;
            let least = 1.0 - self.config.width_confidence;
            let kept = (0..side.indices.len())
                .filter(|&index| {
                    sigmoid(matchability[index]) > least || confidences[index] <= threshold
                })
                .collect::<Vec<_>>();
            side.keep(&kept)?;
        }
        Ok(())
    }

    /// Matches the two sides with layer `layer`'s assignment.
    ///
    /// Returns each kept first keypoint's match among the second side's kept keypoints, or -1 for none, and its score.
    fn assign(
        &self,
        layer: usize,
        sides: &[Side; 2],
        state: &ModelState,
    ) -> SingeResult<(Vec<i64>, Vec<f32>)> {
        let assignment = &self.assignments[layer];
        let width = self.config.width;
        let scale = (width as f64).powf(0.25).recip();
        let projected = |side: &Side| -> SingeResult<Tensor> {
            side.descriptors
                .apply(&assignment.projection, state)?
                .mul_scalar(scale)
        };
        let similarity = projected(&sides[0])?
            .matmul(&projected(&sides[1])?.transpose(1, 2)?)?
            .cast(DataType::F32)?
            .to_vec::<f32>()?;
        let matchability = |side: &Side| {
            side.descriptors
                .apply(&assignment.matchability, state)?
                .cast(DataType::F32)?
                .to_vec::<f32>()
        };
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
                    matches[row] =
                        i64::try_from(column).map_err(|_| SingeError::out_of_range("keypoints"))?;
                }
            }
        }
        Ok((matches, scores))
    }
}

impl MatchingLayers for LightGlueLayers {
    fn match_pair(&self, pixels: &Tensor, state: &ModelState) -> SingeResult<PairMatching> {
        let &[_, _, height, width] = pixels.shape().dimensions() else {
            return Err(SingeError::invalid_argument("image pair"));
        };
        let found = [
            self.detect(pixels, 0, state)?,
            self.detect(pixels, 1, state)?,
        ];
        let keypoints = [found[0].keypoints.clone(), found[1].keypoints.clone()];
        if keypoints.iter().any(Vec::is_empty) {
            return Ok(PairMatching {
                matches: vec![-1; keypoints[0].len()],
                scores: vec![0.0; keypoints[0].len()],
                keypoints,
            });
        }

        let mut sides = [
            self.side(&found[0], Spatial2d::new(height, width), pixels, state)?,
            self.side(&found[1], Spatial2d::new(height, width), pixels, state)?,
        ];
        let total = keypoints[0].len() + keypoints[1].len();
        let (matches, scores) = self.match_sides(&mut sides, total, state)?;
        Ok(unpruned(&sides, &keypoints, &matches, &scores))
    }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

/// Returns `log(sigmoid(value))`, computed without overflow.
fn log_sigmoid(value: f32) -> f32 {
    value.min(0.0) - (-value.abs()).exp().ln_1p()
}

/// Returns each row's log-softmax, of a row-major `rows × columns` matrix.
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

/// Returns the index and value of the greatest of `values`, the first of equals, as `max` gives them.
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

/// Returns the confidence a keypoint needs after layer `layer` of `layers`, falling from 0.9 toward 0.8, as `_get_confidence_threshold` gives it.
fn confidence_threshold(layer: usize, layers: usize) -> f32 {
    (0.8 + 0.1 * (-4.0 * layer as f64 / layers as f64).exp()).clamp(0.0, 1.0) as f32
}

/// Spreads the kept keypoints' matches and scores back over every keypoint of the first image, through the kept indices, with -1 and a score of zero for the pruned.
fn unpruned(
    sides: &[Side; 2],
    keypoints: &[Vec<[f32; 2]>; 2],
    matches: &[i64],
    scores: &[f32],
) -> PairMatching {
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
    PairMatching {
        keypoints: keypoints.clone(),
        matches: all_matches,
        scores: all_scores,
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "lightglue",
    formats: &[(Format::Transformers, &["lightglue"])],
    loaders: &[HeadLoader::KeypointMatching(load_matcher)],
};

pub fn load_matcher(loader: &mut Loader<'_>) -> Result<KeypointMatcher> {
    let config: LightGlueConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        LightGlueLayers::create(schema, &config)
            .map(|layers| Boxed::<dyn MatchingLayers>(Box::new(layers)))
    })?;
    let model = loader.load_whole(&model, &transformers::rules())?;
    Ok(KeypointMatcher::new(model))
}
