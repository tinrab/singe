// TimesFM 2.5: a decoder over patches of a series, each patch normalized by the running statistics of the series up to it and embedded by a residual block, causal blocks whose queries and keys are normed per head after their rotation, and residual blocks from the last patch to a point forecast and to quantile spreads, read twice, from the series and from its negation, and averaged so a flipped series forecasts the flipped forecast.
//
// Its state is `tokenizer`, `point`, and `quantiles` (each `input`, `output`, and `residual`), and `blocks.N` (`attention_norm`, `attention` with `query`, `key`, `value`, `output`, `query_norm`, `key_norm`, and `query_scale`, `post_attention_norm`, `feed_forward_norm`, `feed_forward` with `up` and `down`, and `post_feed_forward_norm`).

mod transformers;

use std::iter;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Device, Error as SingeError, Result as SingeResult, Shape, Tensor,
    nn::{
        Activation, Layer, Linear, Model, ModelState, RmsNorm, RotaryAngles, RotaryFrequencies,
        RotaryLayout, RotarySpec, RotaryTable, StateBuilder, StateKey,
    },
    tensor::AttentionOptions,
};

use crate::{
    error::{Error, Result},
    format::Loader,
    head::{Forecast, Forecaster, HeadLoader},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

// The smallest scale a normalization divides by; a smaller one divides by one.
const TOLERANCE: f32 = 1e-6;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimesFmConfig {
    pub patch: i64,
    // The steps the point forecast covers, and those the quantile spreads cover.
    pub horizon: i64,
    pub quantile_horizon: i64,
    // The outputs of each step: the point forecast and then each quantile.
    pub outputs: i64,
    // The output the point forecast reads, the median's.
    pub median: usize,
    // The most past values read, and the length a shorter series is padded to before its global statistics.
    pub context: usize,
    pub width: i64,
    pub inner: i64,
    pub layers: i64,
    pub heads: i64,
    pub head_dim: i64,
    pub eps: f32,
    pub activation: Activation,
    pub theta: f64,
    // Whether the forecast averages in the negated series' forecast.
    pub flip_invariant: bool,
    // Whether a series without negative values forecasts none.
    pub positive: bool,
    // Whether the quantiles come from the quantile spreads around the median.
    pub continuous_quantiles: bool,
}

// `output(act(input(x))) + residual(x)`.
#[derive(Debug, Clone)]
struct ResidualBlock {
    input: Linear,
    output: Linear,
    residual: Linear,
    activation: Activation,
}

impl ResidualBlock {
    fn create(
        schema: &mut Schema<'_>,
        name: &str,
        [input, hidden, output]: [i64; 3],
        bias: bool,
        activation: Activation,
    ) -> SingeResult<Self> {
        let mut schema = schema.scope(name);
        Ok(Self {
            input: schema.linear("input", input, hidden, bias)?,
            output: schema.linear("output", hidden, output, bias)?,
            residual: schema.linear("residual", input, output, bias)?,
            activation,
        })
    }
}

impl Layer for ResidualBlock {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        input
            .apply(&self.input, state)?
            .apply(&self.activation, state)?
            .apply(&self.output, state)?
            .add(&input.apply(&self.residual, state)?)
    }
}

#[derive(Debug, Clone)]
struct TimesFmBlock {
    attention_norm: RmsNorm,
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    query_norm: RmsNorm,
    key_norm: RmsNorm,
    // Each query feature's learned scale, the softplus of the stored one over `√D`, times `1 / ln 2`.
    query_scale: StateKey,
    post_attention_norm: RmsNorm,
    feed_forward_norm: RmsNorm,
    up: Linear,
    down: Linear,
    post_feed_forward_norm: RmsNorm,
    activation: Activation,
    heads: i64,
    head_dim: i64,
}

impl TimesFmBlock {
    fn forward(
        &self,
        hidden: &Tensor,
        angles: &RotaryAngles,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let dimensions = hidden.shape().dimensions();
        let (batch, length, width) = (dimensions[0], dimensions[1], dimensions[2]);
        let normed = hidden.apply(&self.attention_norm, state)?;
        let heads = |projection: &Linear| -> SingeResult<Tensor> {
            normed
                .apply(projection, state)?
                .reshape([batch, length, self.heads, self.head_dim])
        };
        let query = angles
            .rotate(&heads(&self.query)?)?
            .apply(&self.query_norm, state)?
            .mul(state.tensor(&self.query_scale)?)?
            .permute(&[0, 2, 1, 3])?;
        let key = angles
            .rotate(&heads(&self.key)?)?
            .apply(&self.key_norm, state)?
            .permute(&[0, 2, 1, 3])?;
        let value = heads(&self.value)?.permute(&[0, 2, 1, 3])?;
        let attended = query
            .attend(
                &key,
                &value,
                None,
                None,
                AttentionOptions {
                    scale: Some(1.0),
                    causal: true,
                    ..AttentionOptions::default()
                },
            )?
            .permute(&[0, 2, 1, 3])?
            .contiguous()?
            .reshape([batch, length, width])?
            .apply(&self.output, state)?
            .apply(&self.post_attention_norm, state)?;
        let hidden = hidden.add(&attended)?;
        let fed = hidden
            .apply(&self.feed_forward_norm, state)?
            .apply(&self.up, state)?
            .apply(&self.activation, state)?
            .apply(&self.down, state)?
            .apply(&self.post_feed_forward_norm, state)?;
        hidden.add(&fed)
    }
}

// From each series' patches, normalized and joined to their masks, `[B, N, 2 · P]`, to the last patch's point outputs and then its quantile spreads, `[B, (H + Q) · outputs]`, before they are denormalized.
#[derive(Debug, Clone)]
pub struct TimesFmLayers {
    config: TimesFmConfig,
    tokenizer: ResidualBlock,
    blocks: Vec<TimesFmBlock>,
    point: ResidualBlock,
    quantiles: ResidualBlock,
}

impl TimesFmLayers {
    pub fn create(schema: &mut Schema<'_>, config: &TimesFmConfig) -> SingeResult<Self> {
        let (width, activation) = (config.width, config.activation);
        let blocks = (0..config.layers)
            .map(|index| {
                let mut block = schema.scope(format!("blocks.{index}"));
                let (query, key, value, output, query_norm, key_norm, query_scale) = {
                    let mut attention = block.scope("attention");
                    let projection = config.heads * config.head_dim;
                    (
                        attention.linear("query", width, projection, false)?,
                        attention.linear("key", width, projection, false)?,
                        attention.linear("value", width, projection, false)?,
                        attention.linear("output", projection, width, false)?,
                        attention.rms_norm("query_norm", config.head_dim, config.eps)?,
                        attention.rms_norm("key_norm", config.head_dim, config.eps)?,
                        attention.parameter("query_scale", [config.head_dim])?,
                    )
                };
                let attention_norm = block.rms_norm("attention_norm", width, config.eps)?;
                let post_attention_norm =
                    block.rms_norm("post_attention_norm", width, config.eps)?;
                let feed_forward_norm = block.rms_norm("feed_forward_norm", width, config.eps)?;
                let post_feed_forward_norm =
                    block.rms_norm("post_feed_forward_norm", width, config.eps)?;
                let mut feed_forward = block.scope("feed_forward");
                Ok(TimesFmBlock {
                    attention_norm,
                    query,
                    key,
                    value,
                    output,
                    query_norm,
                    key_norm,
                    query_scale,
                    post_attention_norm,
                    feed_forward_norm,
                    up: feed_forward.linear("up", width, config.inner, false)?,
                    down: feed_forward.linear("down", config.inner, width, false)?,
                    post_feed_forward_norm,
                    activation,
                    heads: config.heads,
                    head_dim: config.head_dim,
                })
            })
            .collect::<SingeResult<Vec<_>>>()?;
        Ok(Self {
            config: config.clone(),
            tokenizer: ResidualBlock::create(
                schema,
                "tokenizer",
                [2 * config.patch, width, width],
                true,
                activation,
            )?,
            blocks,
            point: ResidualBlock::create(
                schema,
                "point",
                [width, width, config.horizon * config.outputs],
                false,
                activation,
            )?,
            quantiles: ResidualBlock::create(
                schema,
                "quantiles",
                [width, width, config.quantile_horizon * config.outputs],
                false,
                activation,
            )?,
        })
    }
}

impl Layer for TimesFmLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let length = input.shape().dimensions()[1];
        let device = input.device();
        let mut hidden = input.apply(&self.tokenizer, state)?;
        let positions = Tensor::from_vec(
            device,
            Shape::create([length])?,
            (0..length).collect::<Vec<i64>>(),
        )?;
        let angles = RotaryTable::create(
            device,
            hidden.data_type(),
            RotarySpec {
                head_dim: self.config.head_dim,
                frequencies: RotaryFrequencies::new(self.config.theta),
                layout: RotaryLayout::HalfSplit,
            },
            length,
        )?
        .angles(&positions)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden, &angles, state)?;
        }

        let last = hidden.select(1, length - 1)?;
        Tensor::concat(
            &[
                last.apply(&self.point, state)?,
                last.apply(&self.quantiles, state)?,
            ],
            1,
        )
    }
}

pub type TimesFm = Model<TimesFmLayers>;

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "timesfm2_5",
    formats: &[(Format::Transformers, &["timesfm2_5"])],
    heads: &[HeadLoader::TimeSeriesForecasting(|loader| {
        Ok(Box::new(load(loader)?))
    })],
};

pub fn load(loader: &mut Loader<'_>) -> Result<TimesFm> {
    let config: TimesFmConfig = loader.config(transformers::config)?;
    let mut builder = StateBuilder::new();
    let layers = {
        let mut schema = Schema::root(&mut builder, loader.options().data_type);
        TimesFmLayers::create(&mut schema, &config)?
    };
    let model = loader.load(
        "model",
        &Model::new(layers, builder.finish()?),
        &transformers::rules(config.head_dim),
    )?;
    loader.finish()?;
    Ok(model)
}

// A patch's running count, mean, and standard deviation of the valid values up to it, as `_update_running_stats` combines them.
#[derive(Debug, Clone, Copy, Default)]
struct Running {
    count: f32,
    mean: f32,
    std: f32,
}

impl Running {
    #[expect(
        clippy::suboptimal_flops,
        reason = "rounds each product as the reference does, which the verifications compare with"
    )]
    fn update(self, values: &[f32], masked: &[bool]) -> Self {
        let valid = || {
            values
                .iter()
                .zip(masked)
                .filter(|&(_, &masked)| !masked)
                .map(|(&value, _)| value)
        };
        let count = valid().count() as f32;
        let mean = if count == 0.0 {
            0.0
        } else {
            valid().sum::<f32>() / count
        };
        let variance = if count == 0.0 {
            0.0
        } else {
            valid().map(|value| (value - mean).powi(2)).sum::<f32>() / count
        };
        let std = variance.max(0.0).sqrt();
        let total = self.count + count;
        let new_mean = if total == 0.0 {
            0.0
        } else {
            (self.count * self.mean + mean * count) / total
        };
        let new_variance = if total == 0.0 {
            0.0
        } else {
            (self.count * self.std.powi(2)
                + count * std.powi(2)
                + self.count * (self.mean - new_mean).powi(2)
                + count * (mean - new_mean).powi(2))
                / total
        };
        Self {
            count: total,
            mean: new_mean,
            std: new_variance.max(0.0).sqrt(),
        }
    }
}

fn safe_scale(scale: f32) -> f32 {
    if scale < TOLERANCE { 1.0 } else { scale }
}

// Each series' last patch statistics and model inputs, `[B, N, 2 · P]`, for each of `series`, already normalized and of one length, padded at the front to whole patches.
fn patches(model: &TimesFm, series: &[Vec<f32>]) -> Result<(Vec<Running>, Tensor)> {
    let config = &model.layer().config;
    let patch = usize::try_from(config.patch).map_err(|_| Error::unsupported("a patch length"))?;
    let length = series.first().map_or(0, Vec::len);
    let front = (patch - length % patch) % patch;
    let count = (front + length) / patch;
    let mut values = Vec::with_capacity(series.len() * count * 2 * patch);
    let mut last = Vec::with_capacity(series.len());
    for values_of in series {
        let padded = iter::repeat_n(0.0, front)
            .chain(values_of.iter().copied())
            .collect::<Vec<f32>>();
        let masked = (0..front + length)
            .map(|index| index < front)
            .collect::<Vec<_>>();
        let mut running = Running::default();
        for index in 0..count {
            let range = index * patch..(index + 1) * patch;
            running = running.update(&padded[range.clone()], &masked[range.clone()]);
            let scale = safe_scale(running.std);
            values.extend(
                padded[range.clone()]
                    .iter()
                    .zip(&masked[range.clone()])
                    .map(|(&value, &masked)| {
                        if masked {
                            0.0
                        } else {
                            (value - running.mean) / scale
                        }
                    }),
            );
            values.extend(
                masked[range]
                    .iter()
                    .map(|&masked| f32::from(u8::from(masked))),
            );
        }

        last.push(running);
    }

    let shape = Shape::create([
        i64::try_from(series.len()).map_err(|_| SingeError::out_of_range("series"))?,
        i64::try_from(count).map_err(|_| SingeError::out_of_range("series"))?,
        2 * config.patch,
    ])?;
    let device: Device = model.state().device()?;
    let data_type: DataType = model.state().floating_type()?;
    Ok((
        last,
        Tensor::from_vec(device, shape, values)?.cast(data_type)?,
    ))
}

impl Forecaster for TimesFm {
    // As `TimesFm2_5ModelForPrediction.forward` forecasts one series with its defaults.
    #[expect(
        clippy::suboptimal_flops,
        reason = "rounds each product as the reference does, which the verifications compare with"
    )]
    fn forecast(&self, series: &[f32]) -> Result<Forecast> {
        let config = &self.layer().config;
        if series.is_empty() {
            return Err(SingeError::invalid_argument("a series without values").into());
        }

        let inputs = &series[series.len().saturating_sub(config.context)..];
        let smallest = inputs.iter().copied().fold(f32::INFINITY, f32::min);

        // The series' statistics as `_preprocess` pads it: with zeros in front up to the context's length.
        let padded = config.context.max(inputs.len());
        let mean = inputs.iter().map(|&value| f64::from(value)).sum::<f64>() / padded as f64;
        let variance = (inputs
            .iter()
            .map(|&value| (f64::from(value) - mean).powi(2))
            .sum::<f64>()
            + (padded - inputs.len()) as f64 * mean.powi(2))
            / (padded as f64 - 1.0);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the reference computes in F32"
        )]
        let (mean, std) = (mean as f32, variance.sqrt() as f32);
        let scale = safe_scale(std);
        let normalized = inputs
            .iter()
            .map(|value| (value - mean) / scale)
            .collect::<Vec<f32>>();
        let mut batch = vec![normalized.clone()];
        if config.flip_invariant {
            batch.push(normalized.iter().map(|value| -value).collect());
        }

        let (last, patches) = patches(self, &batch)?;
        let outputs = self
            .forward(&patches)?
            .cast(DataType::F32)?
            .to_vec::<f32>()?;
        let per = usize::try_from(config.outputs).map_err(|_| Error::unsupported("outputs"))?;
        let horizon =
            usize::try_from(config.horizon).map_err(|_| Error::unsupported("a horizon"))?;
        let quantile_horizon = usize::try_from(config.quantile_horizon)
            .map_err(|_| Error::unsupported("a horizon"))?;
        let row = (horizon + quantile_horizon) * per;
        // Each series' point outputs and quantile spreads, `[steps][outputs]`, denormalized by its last patch's statistics.
        let read = |index: usize| -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
            let outputs = &outputs[index * row..(index + 1) * row];
            let Running { mean, std, .. } = last[index];
            let steps = |values: &[f32]| {
                values
                    .chunks(per)
                    .map(|step| step.iter().map(|value| value * std + mean).collect())
                    .collect::<Vec<Vec<f32>>>()
            };
            (
                steps(&outputs[..horizon * per]),
                steps(&outputs[horizon * per..]),
            )
        };
        let (mut point, mut spreads) = read(0);
        if config.flip_invariant {
            let (flipped_point, flipped_spreads) = read(1);
            // The negated series' quantiles mirror the series': the lowest becomes the highest.
            let combine = |values: &mut [Vec<f32>], flipped: &[Vec<f32>]| {
                for (step, flipped) in values.iter_mut().zip(flipped) {
                    let mirrored = iter::once(flipped[0])
                        .chain(flipped[1..].iter().rev().copied())
                        .collect::<Vec<_>>();
                    for (value, mirrored) in step.iter_mut().zip(mirrored) {
                        *value = (*value - mirrored) / 2.0;
                    }
                }
            };
            combine(&mut point, &flipped_point);
            combine(&mut spreads, &flipped_spreads);
        }

        let median = config.median;
        if config.continuous_quantiles {
            for (step, spread) in point.iter_mut().zip(&spreads) {
                for quantile in 1..per {
                    if quantile != median {
                        step[quantile] = spread[quantile] - spread[median] + step[median];
                    }
                }
            }
        }

        let clamp = config.positive && smallest >= 0.0;
        let full = point
            .into_iter()
            .map(|step| {
                step.into_iter()
                    .map(|value| {
                        let value = value * std + mean;
                        if clamp { value.max(0.0) } else { value }
                    })
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<_>>();
        Ok(Forecast {
            mean: full.iter().map(|step| step[median]).collect(),
            full,
        })
    }
}
