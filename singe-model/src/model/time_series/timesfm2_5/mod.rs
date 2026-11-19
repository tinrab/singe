// TimesFM 2.5: a decoder over patches of a series, each patch normalized by the running statistics of the series up to it and embedded by a residual block, causal blocks whose queries and keys are normed per head after their rotation, and residual blocks from the last patch to a point forecast and to quantile spreads, read twice, from the series and from its negation, and averaged so a flipped series forecasts the flipped forecast.
//
// Its state is `tokenizer`, `point`, and `quantiles` (each `input`, `output`, and `residual`), and `blocks.N` (`attention_norm`, `attention` with `query`, `key`, `value`, `output`, `query_norm`, `key_norm`, and `query_scale`, `post_attention_norm`, `feed_forward_norm`, `feed_forward` with `up` and `down`, and `post_feed_forward_norm`).

mod transformers;

use std::iter;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Device, Error as SingeError, Result as SingeResult, Shape, Tensor,
    nn::{
        Activation, Layer, Linear, Mlp, Model, ModelState, RmsNorm, RotaryAngles,
        RotaryFrequencies, RotaryLayout, RotarySpec, RotaryTable, StateKey,
    },
    tensor::AttentionOptions,
};

use crate::{
    assembly::{merge_heads, split_heads},
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
    pub patch: usize,
    // The steps the point forecast covers, and those the quantile spreads cover.
    pub horizon: usize,
    pub quantile_horizon: usize,
    // The outputs of each step: the point forecast and then each quantile.
    pub outputs: usize,
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
    pub frequencies: RotaryFrequencies,
    // Whether the forecast averages in the negated series' forecast.
    pub flip_invariant: bool,
    // Whether a series without negative values forecasts none.
    pub positive: bool,
    // Whether the quantiles come from the quantile spreads around the median.
    pub continuous_quantiles: bool,
}

impl TimesFmConfig {
    // Series `index`'s point outputs and quantile spreads among `outputs`, `[steps][outputs]`, denormalized by its last patch's statistics.
    fn denormalized(
        &self,
        outputs: &[f32],
        index: usize,
        running: Running,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let per = self.outputs;
        let row = (self.horizon + self.quantile_horizon) * per;
        let outputs = &outputs[index * row..(index + 1) * row];
        let Running { mean, std, .. } = running;
        let steps = |values: &[f32]| {
            values
                .chunks(per)
                .map(|step| step.iter().map(|value| value * std + mean).collect())
                .collect::<Vec<Vec<f32>>>()
        };
        (
            steps(&outputs[..self.horizon * per]),
            steps(&outputs[self.horizon * per..]),
        )
    }
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
struct TimesFmAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    query_norm: RmsNorm,
    key_norm: RmsNorm,
    // Each query feature's learned scale, the softplus of the stored one over `√D`, times `1 / ln 2`.
    query_scale: StateKey,
    heads: i64,
    head_dim: i64,
}

impl TimesFmAttention {
    fn create(schema: &mut Schema<'_>, config: &TimesFmConfig) -> SingeResult<Self> {
        let (width, projection) = (config.width, config.heads * config.head_dim);
        let attention = Self {
            query: schema.linear("query", width, projection, false)?,
            key: schema.linear("key", width, projection, false)?,
            value: schema.linear("value", width, projection, false)?,
            output: schema.linear("output", projection, width, false)?,
            query_norm: schema.rms_norm("query_norm", config.head_dim, config.eps)?,
            key_norm: schema.rms_norm("key_norm", config.head_dim, config.eps)?,
            query_scale: schema.parameter("query_scale", [config.head_dim])?,
            heads: config.heads,
            head_dim: config.head_dim,
        };
        schema.join_rows(&[&attention.query, &attention.key, &attention.value])?;
        Ok(attention)
    }

    fn forward(
        &self,
        normed: &Tensor,
        angles: &RotaryAngles,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let &[batch, length, _] = normed.shape().dimensions() else {
            return Err(SingeError::invalid_argument("attention input"));
        };
        let heads = |projection: &Linear| -> SingeResult<Tensor> {
            normed
                .apply(projection, state)?
                .reshape([batch, length, self.heads, self.head_dim])
        };
        let query = angles
            .rotate(&heads(&self.query)?)?
            .apply(&self.query_norm, state)?
            .mul(state.tensor(&self.query_scale)?)?
            .transpose(1, 2)?;
        let key = angles
            .rotate(&heads(&self.key)?)?
            .apply(&self.key_norm, state)?
            .transpose(1, 2)?;
        let value = split_heads(&normed.apply(&self.value, state)?, self.heads)?;
        let attended = query.attend(
            &key,
            &value,
            None,
            None,
            AttentionOptions {
                scale: Some(1.0),
                causal: true,
                ..AttentionOptions::default()
            },
        )?;
        merge_heads(&attended)?.apply(&self.output, state)
    }
}

#[derive(Debug, Clone)]
struct TimesFmBlock {
    attention_norm: RmsNorm,
    attention: TimesFmAttention,
    post_attention_norm: RmsNorm,
    feed_forward_norm: RmsNorm,
    feed_forward: Mlp,
    post_feed_forward_norm: RmsNorm,
}

impl TimesFmBlock {
    fn create(schema: &mut Schema<'_>, config: &TimesFmConfig) -> SingeResult<Self> {
        let (width, eps) = (config.width, config.eps);
        Ok(Self {
            attention_norm: schema.rms_norm("attention_norm", width, eps)?,
            attention: TimesFmAttention::create(&mut schema.scope("attention"), config)?,
            post_attention_norm: schema.rms_norm("post_attention_norm", width, eps)?,
            feed_forward_norm: schema.rms_norm("feed_forward_norm", width, eps)?,
            feed_forward: schema.mlp(
                "feed_forward",
                [width, config.inner, width],
                config.activation,
                false,
            )?,
            post_feed_forward_norm: schema.rms_norm("post_feed_forward_norm", width, eps)?,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        angles: &RotaryAngles,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let attended = self
            .attention
            .forward(&hidden.apply(&self.attention_norm, state)?, angles, state)?
            .apply(&self.post_attention_norm, state)?;
        let hidden = hidden.add(&attended)?;
        let fed = hidden
            .apply(&self.feed_forward_norm, state)?
            .apply(&self.feed_forward, state)?
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
        let width_of = |count: usize, what: &'static str| {
            i64::try_from(count).map_err(|_| SingeError::out_of_range(what))
        };
        let blocks = (0..config.layers)
            .map(|index| TimesFmBlock::create(&mut schema.scope(format!("blocks.{index}")), config))
            .collect::<SingeResult<Vec<_>>>()?;
        Ok(Self {
            config: config.clone(),
            tokenizer: ResidualBlock::create(
                schema,
                "tokenizer",
                [width_of(2 * config.patch, "patch length")?, width, width],
                true,
                activation,
            )?,
            blocks,
            point: ResidualBlock::create(
                schema,
                "point",
                [
                    width,
                    width,
                    width_of(config.horizon * config.outputs, "horizon")?,
                ],
                false,
                activation,
            )?,
            quantiles: ResidualBlock::create(
                schema,
                "quantiles",
                [
                    width,
                    width,
                    width_of(config.quantile_horizon * config.outputs, "quantile horizon")?,
                ],
                false,
                activation,
            )?,
        })
    }
}

impl Layer for TimesFmLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let &[_, length, _] = input.shape().dimensions() else {
            return Err(SingeError::invalid_argument("series patches"));
        };
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
                frequencies: self.config.frequencies,
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

impl Forecaster for TimesFm {
    // As `TimesFm2_5ModelForPrediction.forward` forecasts one series with its defaults.
    fn forecast(&self, series: &[f32]) -> Result<Forecast> {
        let config = &self.layer().config;
        if series.is_empty() {
            return Err(Error::invalid("a series without values"));
        }

        let inputs = &series[series.len().saturating_sub(config.context)..];
        let smallest = inputs.iter().copied().fold(f32::INFINITY, f32::min);
        let (mean, std) = statistics(inputs, config.context);
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
        let (mut point, mut spreads) = config.denormalized(&outputs, 0, last[0]);
        if config.flip_invariant {
            let (flipped_point, flipped_spreads) = config.denormalized(&outputs, 1, last[1]);
            unflip(&mut point, &flipped_point);
            unflip(&mut spreads, &flipped_spreads);
        }

        let median = config.median;
        if config.continuous_quantiles {
            for (step, spread) in point.iter_mut().zip(&spreads) {
                for quantile in 1..config.outputs {
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

// A patch's running count, mean, and standard deviation of the valid values up to it, as `_update_running_stats` combines them.
#[derive(Debug, Clone, Copy, Default)]
struct Running {
    count: f32,
    mean: f32,
    std: f32,
}

impl Running {
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

// The series' mean and standard deviation as `_preprocess` pads it: with zeros in front up to the context's length.
fn statistics(inputs: &[f32], context: usize) -> (f32, f32) {
    let padded = context.max(inputs.len());
    let mean = inputs.iter().map(|&value| f64::from(value)).sum::<f64>() / padded as f64;
    let variance = (inputs
        .iter()
        .map(|&value| (f64::from(value) - mean).powi(2))
        .sum::<f64>()
        + (padded - inputs.len()) as f64 * mean.powi(2))
        / (padded as f64 - 1.0);
    (mean as f32, variance.sqrt() as f32)
}

// Averages `values` with the negated series' `flipped`, whose quantiles mirror the series': the lowest becomes the highest.
fn unflip(values: &mut [Vec<f32>], flipped: &[Vec<f32>]) {
    for (step, flipped) in values.iter_mut().zip(flipped) {
        let mirrored = iter::once(flipped[0])
            .chain(flipped[1..].iter().rev().copied())
            .collect::<Vec<_>>();
        for (value, mirrored) in step.iter_mut().zip(mirrored) {
            *value = (*value - mirrored) / 2.0;
        }
    }
}

// Each series' last patch statistics and model inputs, `[B, N, 2 · P]`, for each of `series`, already normalized and of one length, padded at the front to whole patches.
fn patches(model: &TimesFm, series: &[Vec<f32>]) -> Result<(Vec<Running>, Tensor)> {
    let patch = model.layer().config.patch;
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
        i64::try_from(2 * patch).map_err(|_| SingeError::out_of_range("patch length"))?,
    ])?;
    let device: Device = model.state().device()?;
    let data_type: DataType = model.state().floating_type()?;
    Ok((
        last,
        Tensor::from_vec(device, shape, values)?.cast(data_type)?,
    ))
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "timesfm2_5",
    formats: &[(Format::Transformers, &["timesfm2_5"])],
    loaders: &[HeadLoader::TimeSeriesForecasting(|loader| {
        Ok(Box::new(load(loader)?))
    })],
};

pub fn load(loader: &mut Loader<'_>) -> Result<TimesFm> {
    let config: TimesFmConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        TimesFmLayers::create(schema, &config)
    })?;
    loader.load_whole(&model, &transformers::rules(config.head_dim))
}
