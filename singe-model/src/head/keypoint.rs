// Keypoint detection, as SuperPoint gives it: each pixel's keypoint probability and a map of descriptors on the device, then on the host the local maxima above a threshold away from the border, each described by the map sampled at it.

use std::fmt::Debug;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Error as SingeError, Result as SingeResult, Tensor,
    nn::{Model, ModelState},
};

use crate::{error::Result, head::Boxed, host};

// What a keypoint detector's layers give: each pixel's keypoint probability, `[B, H, W]`, and unit descriptors over a coarser map, `[B, D, h, w]`.
pub trait KeypointLayers: Debug + Send + Sync {
    fn maps(&self, pixels: &Tensor, state: &ModelState) -> SingeResult<(Tensor, Tensor)>;
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct KeypointConfig {
    // The least probability a keypoint has.
    pub threshold: f32,
    // The most keypoints kept, the most probable, or every one.
    pub max_keypoints: Option<usize>,
    // The radius of the square within which a keypoint is the most probable.
    pub radius: usize,
    // The pixels from each edge within which no keypoint is kept.
    pub border: usize,
}

// An image's keypoints at pixels of the image the model read, `[x, y]`, their probabilities, and their unit descriptors.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Keypoints {
    pub keypoints: Vec<[f32; 2]>,
    pub scores: Vec<f32>,
    pub descriptors: Vec<Vec<f32>>,
}

#[derive(Debug)]
pub struct KeypointDetector {
    model: Model<Boxed<dyn KeypointLayers>>,
    config: KeypointConfig,
}

// A row-major `height × width` map.
struct Plane<'a> {
    values: &'a [f32],
    height: usize,
    width: usize,
}

impl Plane<'_> {
    // The maximum over each point's `(2 · radius + 1)`-sided square, separably, as `max_pool2d` with stride one and padding `radius` takes it.
    fn max_pool(&self, radius: usize) -> Vec<f32> {
        let (height, width) = (self.height, self.width);
        let mut rows = vec![f32::NEG_INFINITY; height * width];
        for row in 0..height {
            for column in 0..width {
                let (start, end) = (
                    column.saturating_sub(radius),
                    (column + radius + 1).min(width),
                );
                rows[row * width + column] = self.values[row * width + start..row * width + end]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
            }
        }

        let mut pooled = vec![f32::NEG_INFINITY; height * width];
        for row in 0..height {
            let (start, end) = (row.saturating_sub(radius), (row + radius + 1).min(height));
            for column in 0..width {
                pooled[row * width + column] = (start..end)
                    .map(|other| rows[other * width + column])
                    .fold(f32::NEG_INFINITY, f32::max);
            }
        }
        pooled
    }
}

// The scores but where a point is no local maximum, zero, as SuperPoint's `simple_nms` keeps them: maxima first, then twice, maxima among the points no kept maximum suppresses.
fn suppressed(scores: &[f32], height: usize, width: usize, radius: usize) -> Vec<f32> {
    let pool = |values: &[f32]| {
        Plane {
            values,
            height,
            width,
        }
        .max_pool(radius)
    };
    let pooled = pool(scores);
    let mut maxima = scores
        .iter()
        .zip(&pooled)
        .map(|(score, pooled)| score == pooled)
        .collect::<Vec<_>>();
    for _ in 0..2 {
        let near = pool(
            &maxima
                .iter()
                .map(|&kept| f32::from(u8::from(kept)))
                .collect::<Vec<_>>(),
        );
        let near = near.iter().map(|&value| value > 0.0).collect::<Vec<_>>();
        let rest = scores
            .iter()
            .zip(&near)
            .map(|(&score, &near)| if near { 0.0 } else { score })
            .collect::<Vec<_>>();
        let rest_pooled = pool(&rest);
        for index in 0..maxima.len() {
            maxima[index] |= rest[index] == rest_pooled[index] && !near[index];
        }
    }
    scores
        .iter()
        .zip(&maxima)
        .map(|(&score, &kept)| if kept { score } else { 0.0 })
        .collect()
}

// Samples `descriptors`, `[D, h, w]` row-major, bilinearly at a keypoint, as SuperPoint's `_sample_descriptors` does through `grid_sample` with aligned corners and zeros outside, and scales the result to unit length.
fn sample(
    descriptors: &[f32],
    depth: usize,
    height: usize,
    width: usize,
    [x, y]: [f32; 2],
) -> Vec<f32> {
    const SCALE: f32 = 8.0;
    let normalized = |value: f32, side: usize| {
        let divisor = side as f32 * SCALE - SCALE / 2.0 - 0.5;
        ((value - SCALE / 2.0 + 0.5) / divisor) * 2.0 - 1.0
    };
    #[expect(
        clippy::manual_midpoint,
        reason = "`transformers` computes `(value + 1) / 2`"
    )]
    let unnormalized = |value: f32, side: usize| (value + 1.0) / 2.0 * (side as f32 - 1.0);
    let column = unnormalized(normalized(x, width), width);
    let row = unnormalized(normalized(y, height), height);
    let (left, top) = (column.floor(), row.floor());
    let (right_weight, bottom_weight) = (column - left, row - top);
    let corners = [
        (top, left, (1.0 - bottom_weight) * (1.0 - right_weight)),
        (top, left + 1.0, (1.0 - bottom_weight) * right_weight),
        (top + 1.0, left, bottom_weight * (1.0 - right_weight)),
        (top + 1.0, left + 1.0, bottom_weight * right_weight),
    ];
    let mut sampled = vec![0.0_f32; depth];
    for (row, column, weight) in corners {
        if row < 0.0 || column < 0.0 || row >= height as f32 || column >= width as f32 {
            continue;
        }

        let at = row as usize * width + column as usize;
        for (channel, value) in sampled.iter_mut().enumerate() {
            *value += descriptors[channel * height * width + at] * weight;
        }
    }

    let norm = sampled
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(1e-12);
    sampled.iter().map(|value| value / norm).collect()
}

impl KeypointDetector {
    #[must_use]
    pub const fn new(model: Model<Boxed<dyn KeypointLayers>>, config: KeypointConfig) -> Self {
        Self { model, config }
    }

    #[must_use]
    pub const fn model(&self) -> &Model<Boxed<dyn KeypointLayers>> {
        &self.model
    }

    #[must_use]
    pub const fn config(&self) -> &KeypointConfig {
        &self.config
    }

    // Returns each image's keypoints.
    pub fn detect(&self, pixels: &Tensor) -> Result<Vec<Keypoints>> {
        let (scores, descriptors) = self.model.layer().maps(pixels, self.model.state())?;
        Ok(extract_keypoints(&scores, &descriptors, &self.config)?)
    }
}

// Each image's keypoints from its probabilities and descriptors, as `KeypointLayers` give them, in the order of their rows and then columns, or the most probable first when `config` bounds their count.
pub fn extract_keypoints(
    scores: &Tensor,
    descriptors: &Tensor,
    config: &KeypointConfig,
) -> SingeResult<Vec<Keypoints>> {
    let side = |dimension: i64| {
        usize::try_from(dimension).map_err(|_| SingeError::out_of_range("map size"))
    };
    let &[batch, height, width] = scores.shape().dimensions() else {
        return Err(SingeError::invalid_argument(
            "keypoint probabilities, [B, H, W]",
        ));
    };
    let &[_, depth, rows, columns] = descriptors.shape().dimensions() else {
        return Err(SingeError::invalid_argument("descriptors, [B, D, h, w]"));
    };
    let (batch, height, width) = (side(batch)?, side(height)?, side(width)?);
    let (depth, rows, columns) = (side(depth)?, side(rows)?, side(columns)?);
    let scores = scores.cast(DataType::F32)?.to_vec::<f32>()?;
    let descriptors = descriptors.cast(DataType::F32)?.to_vec::<f32>()?;
    Ok((0..batch)
        .map(|image| {
            let scores = suppressed(
                &scores[image * height * width..(image + 1) * height * width],
                height,
                width,
                config.radius,
            );
            let border = config.border;
            let mut found = (0..height * width)
                .filter(|&index| scores[index] > config.threshold)
                .map(|index| (index / width, index % width))
                .filter(|&(row, column)| {
                    row >= border
                        && row + border < height
                        && column >= border
                        && column + border < width
                })
                .map(|(row, column)| ((row, column), scores[row * width + column]))
                .collect::<Vec<_>>();
            if let Some(most) = config.max_keypoints.filter(|&most| most < found.len()) {
                host::best_first(&mut found, most, |&(_, score)| score);
            }

            let map =
                &descriptors[image * depth * rows * columns..(image + 1) * depth * rows * columns];
            let keypoints = found
                .iter()
                .map(|&((row, column), _)| [column as f32, row as f32])
                .collect::<Vec<_>>();
            Keypoints {
                descriptors: keypoints
                    .iter()
                    .map(|&point| sample(map, depth, rows, columns, point))
                    .collect(),
                scores: found.iter().map(|&(_, score)| score).collect(),
                keypoints,
            }
        })
        .collect())
}

// What a keypoint matcher's layers give for a pair of images, `[2, C, H, W]`: each image's keypoints at its pixels, `[x, y]`, and each first keypoint's match among the second's, or -1, with its score.
pub trait MatchingLayers: Debug + Send + Sync {
    fn match_pair(&self, pixels: &Tensor, state: &ModelState) -> SingeResult<PairMatches>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct PairMatches {
    pub keypoints: [Vec<[f32; 2]>; 2],
    pub matches: Vec<i64>,
    pub scores: Vec<f32>,
}

#[derive(Debug)]
pub struct KeypointMatcher {
    model: Model<Boxed<dyn MatchingLayers>>,
}

impl KeypointMatcher {
    #[must_use]
    pub const fn new(model: Model<Boxed<dyn MatchingLayers>>) -> Self {
        Self { model }
    }

    #[must_use]
    pub const fn model(&self) -> &Model<Boxed<dyn MatchingLayers>> {
        &self.model
    }

    pub fn match_pair(&self, pixels: &Tensor) -> Result<PairMatches> {
        Ok(self.model.layer().match_pair(pixels, self.model.state())?)
    }
}
