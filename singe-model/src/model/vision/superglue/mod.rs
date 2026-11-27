//! `SuperGlue`: `SuperPoint`'s keypoints in two images, each encoded from its position and score and added to its descriptor, refined by layers of attention within an image and across to the other, then matched by optimal transport, with a dustbin for keypoints that match nothing, and mutual best matches.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Error as SingeError, Result as SingeResult, Shape, Tensor,
    nn::{Activation, BatchNorm, Linear, ModelState, StateKey},
};

use crate::{
    assembly::{Attention, AttentionSpec, PYTORCH_NORM_EPS},
    error::Result,
    format::Loader,
    head::{
        Boxed, HeadLoader, KeypointLayers, KeypointMatcher, Keypoints, MatchingLayers, PairMatches,
        extract_keypoints,
    },
    model::{
        ModelEntry,
        vision::superpoint::{SuperPointConfig, SuperPointLayers},
    },
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SuperGlueConfig {
    pub detector: SuperPointConfig,
    pub width: i64,
    pub heads: i64,
    /// The widths of the keypoint encoder's hidden layers.
    pub encoder: Vec<i64>,
    /// Whether each layer attends across to the other image rather than within its own.
    pub crosses: Vec<bool>,
    pub sinkhorn_iterations: usize,
    /// The score a mutual match must exceed to be kept.
    pub matching_threshold: f32,
}

/// A linear layer, then a batch norm over its features and `ReLU`.
#[derive(Debug, Clone)]
struct NormedLinear {
    linear: Linear,
    norm: BatchNorm,
}

impl NormedLinear {
    fn create(schema: &mut Schema<'_>, input: i64, output: i64) -> SingeResult<Self> {
        Ok(Self {
            linear: schema.linear("linear", input, output, true)?,
            norm: schema.batch_norm("norm", output, PYTORCH_NORM_EPS)?,
        })
    }

    /// Returns `input`, `[1, N, C]`, through the layer.
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        input
            .apply(&self.linear, state)?
            .transpose(1, 2)?
            .apply(&self.norm, state)?
            .transpose(1, 2)?
            .apply(&Activation::Relu, state)
    }
}

/// An attentional propagation layer: attention to its image's or the other's keypoints, and an MLP of each descriptor and its message, added back.
#[derive(Debug, Clone)]
struct Propagation {
    attention: Attention,
    hidden: NormedLinear,
    output: Linear,
    cross: bool,
}

#[derive(Debug, Clone)]
pub struct SuperGlueLayers {
    detector: SuperPointLayers,
    encoder: Vec<NormedLinear>,
    encoder_output: Linear,
    layers: Vec<Propagation>,
    projection: Linear,
    bin_score: StateKey,
    config: SuperGlueConfig,
}

impl SuperGlueLayers {
    pub fn create(schema: &mut Schema<'_>, config: &SuperGlueConfig) -> SingeResult<Self> {
        let width = config.width;
        let mut input = 3;
        let mut encoder = Vec::with_capacity(config.encoder.len());
        for (index, &output) in config.encoder.iter().enumerate() {
            encoder.push(NormedLinear::create(
                &mut schema.scope(format!("encoder.{index}")),
                input,
                output,
            )?);
            input = output;
        }

        let encoder_output = schema.linear("encoder_output", input, width, true)?;
        let layers = config
            .crosses
            .iter()
            .enumerate()
            .map(|(index, &cross)| {
                let mut layer = schema.scope(format!("layers.{index}"));
                Ok(Propagation {
                    attention: Attention::create(
                        &mut layer.scope("attention"),
                        AttentionSpec::square(width, config.heads),
                    )?,
                    hidden: NormedLinear::create(&mut layer.scope("hidden"), 2 * width, 2 * width)?,
                    output: layer.linear("output", 2 * width, width, true)?,
                    cross,
                })
            })
            .collect::<SingeResult<Vec<_>>>()?;
        Ok(Self {
            detector: SuperPointLayers::create(&mut schema.scope("detector"), &config.detector)?,
            encoder,
            encoder_output,
            layers,
            projection: schema.linear("projection", width, width, true)?,
            bin_score: schema.parameter("bin_score", [])?,
            config: config.clone(),
        })
    }

    fn detect(&self, pixels: &Tensor, image: i64, state: &ModelState) -> SingeResult<Keypoints> {
        let (scores, descriptors) = self.detector.maps(&pixels.narrow(0, image, 1)?, state)?;
        extract_keypoints(&scores, &descriptors, &self.config.detector.keypoints)?
            .into_iter()
            .next()
            .ok_or_else(|| SingeError::invalid_argument("keypoint maps of no image"))
    }

    /// Returns one image's descriptors, `[1, n, W]`, with its keypoints' encoding added: each keypoint's position centered and scaled by 0.7 of the longer side, and its score, as `normalize_keypoints` and the keypoint encoder read them.
    fn encoded(
        &self,
        found: &Keypoints,
        [height, width]: [i64; 2],
        like: &Tensor,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let device = like.device();
        let count = i64::try_from(found.keypoints.len())
            .map_err(|_| SingeError::out_of_range("keypoints"))?;
        let scale = height.max(width) as f32 * 0.7;
        let (side_x, side_y) = (width as f32, height as f32);
        let (center_x, center_y) = (side_x / 2.0, side_y / 2.0);
        // SuperPoint's coordinates relative to the image, back at its pixels.
        let positions = found
            .keypoints
            .iter()
            .zip(&found.scores)
            .flat_map(|(&[x, y], &score)| {
                [
                    ((x / side_x) * side_x - center_x) / scale,
                    ((y / side_y) * side_y - center_y) / scale,
                    score,
                ]
            })
            .collect::<Vec<_>>();
        let mut hidden = Tensor::from_vec(device, Shape::create([1, count, 3])?, positions)?
            .cast(like.data_type())?;
        for layer in &self.encoder {
            hidden = layer.forward(&hidden, state)?;
        }

        let descriptors = Tensor::from_vec(
            device,
            Shape::create([1, count, self.config.width])?,
            found.descriptors.iter().flatten().copied().collect(),
        )?
        .cast(like.data_type())?;
        descriptors.add(&hidden.apply(&self.encoder_output, state)?)
    }

    /// Returns the log of the optimal transport of `scores`, `[n₀, n₁]`, padded with the lowest score to `count × count`, as the reference pads the image with fewer keypoints, with a dustbin row and column of the bin score, after the configured Sinkhorn iterations.
    fn transport(&self, scores: &Tensor, count: i64, state: &ModelState) -> SingeResult<Tensor> {
        let &[rows, columns] = scores.shape().dimensions() else {
            return Err(SingeError::invalid_argument("matching scores, [n₀, n₁]"));
        };
        let device = scores.device();
        let lowest = |shape: [i64; 2]| {
            let len = usize::try_from(shape[0] * shape[1])
                .map_err(|_| SingeError::out_of_range("matching padding"))?;
            Tensor::from_vec(device, Shape::create(shape)?, vec![f32::MIN; len])
        };
        let mut padded = scores.clone();
        if columns < count {
            padded = Tensor::concat(&[padded, lowest([rows, count - columns])?], 1)?;
        }

        if rows < count {
            padded = Tensor::concat(&[padded, lowest([count - rows, count])?], 0)?;
        }

        let bin = state
            .tensor(&self.bin_score)?
            .cast(DataType::F32)?
            .reshape([1, 1])?;
        let couplings = Tensor::concat(
            &[
                Tensor::concat(&[padded, bin.expand(Shape::create([count, 1])?)?], 1)?,
                bin.expand(Shape::create([1, count + 1])?)?,
            ],
            0,
        )?;
        // Each keypoint's mass, and each dustbin's, the other image's count of them, all over both counts.
        let normalization = -(2.0 * count as f32).ln();
        let mut values = vec![
            normalization;
            usize::try_from(count)
                .map_err(|_| SingeError::out_of_range("keypoints"))?
        ];
        values.push((count as f32).ln() + normalization);
        let marginal = Tensor::from_vec(device, Shape::create([count + 1])?, values)?;
        let zeros = marginal.mul_scalar(0.0)?;
        let (mut rows_scaling, mut columns_scaling) = (zeros.clone(), zeros);
        for _ in 0..self.config.sinkhorn_iterations {
            rows_scaling = marginal.sub(
                &couplings
                    .add(&columns_scaling.unsqueeze(0)?)?
                    .logsumexp(&[1], false)?,
            )?;
            columns_scaling = marginal.sub(
                &couplings
                    .add(&rows_scaling.unsqueeze(1)?)?
                    .logsumexp(&[0], false)?,
            )?;
        }
        couplings
            .add(&rows_scaling.unsqueeze(1)?)?
            .add(&columns_scaling.unsqueeze(0)?)?
            .sub_scalar(f64::from(normalization))
    }
}

impl MatchingLayers for SuperGlueLayers {
    fn match_pair(&self, pixels: &Tensor, state: &ModelState) -> SingeResult<PairMatches> {
        let &[_, _, height, width] = pixels.shape().dimensions() else {
            return Err(SingeError::invalid_argument("image pair"));
        };
        let found = [
            self.detect(pixels, 0, state)?,
            self.detect(pixels, 1, state)?,
        ];
        let (first, second) = (found[0].keypoints.len(), found[1].keypoints.len());
        if first == 0 || second == 0 {
            return Ok(PairMatches {
                keypoints: found.map(|found| found.keypoints),
                matches: vec![-1; first],
                scores: vec![0.0; first],
            });
        }

        let mut descriptors = [
            self.encoded(&found[0], [height, width], pixels, state)?,
            self.encoded(&found[1], [height, width], pixels, state)?,
        ];
        for layer in &self.layers {
            let propagated = |own: &Tensor, other: &Tensor| -> SingeResult<Tensor> {
                let source = if layer.cross { other } else { own };
                let message = layer.attention.forward(own, source, source, state)?;
                let delta = layer
                    .hidden
                    .forward(&Tensor::concat(&[own.clone(), message], 2)?, state)?
                    .apply(&layer.output, state)?;
                own.add(&delta)
            };
            let [first_side, second_side] = &descriptors;
            descriptors = [
                propagated(first_side, second_side)?,
                propagated(second_side, first_side)?,
            ];
        }

        let [first_final, second_final] =
            descriptors.map(|side| side.apply(&self.projection, state));
        let scores = first_final?
            .select(0, 0)?
            .matmul(&second_final?.select(0, 0)?.transpose(0, 1)?)?
            .div_scalar((self.config.width as f64).sqrt())?
            .cast(DataType::F32)?;
        let count =
            i64::try_from(first.max(second)).map_err(|_| SingeError::out_of_range("keypoints"))?;
        let transported = self.transport(&scores, count, state)?;
        let stride =
            usize::try_from(count + 1).map_err(|_| SingeError::out_of_range("keypoints"))?;
        let values = transported.to_vec::<f32>()?;
        let entry = |row: usize, column: usize| values[row * stride + column];
        let size = stride - 1;
        let rows_best = (0..size)
            .map(|row| first_best((0..size).map(|column| (column, entry(row, column)))))
            .collect::<Vec<_>>();
        let columns_best = (0..size)
            .map(|column| first_best((0..size).map(|row| (row, entry(row, column)))).0)
            .collect::<Vec<_>>();
        let mut matches = Vec::with_capacity(first);
        let mut scores = Vec::with_capacity(first);
        for &(column, value) in rows_best.iter().take(first) {
            let row = matches.len();
            let mutual = columns_best[column] == row;
            let score = if mutual { value.exp() } else { 0.0 };
            let score = if score > self.config.matching_threshold {
                score
            } else {
                0.0
            };
            let valid = mutual && score > 0.0 && column < second;
            matches.push(if valid {
                i64::try_from(column).map_err(|_| SingeError::out_of_range("match"))?
            } else {
                -1
            });
            scores.push(if valid { score } else { 0.0 });
        }
        Ok(PairMatches {
            keypoints: found.map(|found| found.keypoints),
            matches,
            scores,
        })
    }
}

/// Returns the index and value of the largest of `values`, the first of equals, as `max` gives it.
fn first_best(values: impl Iterator<Item = (usize, f32)>) -> (usize, f32) {
    values.fold((0, f32::NEG_INFINITY), |best, (index, value)| {
        if value > best.1 { (index, value) } else { best }
    })
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "superglue",
    formats: &[(Format::Transformers, &["superglue"])],
    loaders: &[HeadLoader::KeypointMatching(load_matcher)],
};

pub fn load_matcher(loader: &mut Loader<'_>) -> Result<KeypointMatcher> {
    let config: SuperGlueConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        SuperGlueLayers::create(schema, &config)
            .map(|layers| Boxed::<dyn MatchingLayers>(Box::new(layers)))
    })?;
    let model = loader.load_whole(&model, &transformers::rules(&config))?;
    Ok(KeypointMatcher::new(model))
}
