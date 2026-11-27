//! Autoformer: an encoder and a decoder whose attention aggregates each sequence's values at the delays its autocorrelation peaks at, and which split their states into seasonal and trend parts by moving averages; the decoder's trends accumulate into a forecast of a Student-T distribution per step.
//! A series is scaled by its mean absolute value over the context, and each step reads the series at a set of lags, the series' static features, and its time features.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Error as SingeError, Result as SingeResult, Shape, Tensor,
    nn::{Activation, Conv1d, Conv1dConfig, Embedding, LayerNorm, Linear, Mlp, Model, ModelState},
    shape::spatial::Padding,
};

use crate::{
    error::{Error, Result},
    format::Loader,
    head::{
        Forecast, Forecaster, HeadLoader, PastSeries, PathSampling, Series, UniformSource, upload,
    },
    model::{
        ModelEntry,
        time_series::lagged::{self, StepEmbedding},
    },
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutoformerConfig {
    pub width: i64,
    pub heads: i64,
    pub encoder_layers: i64,
    pub decoder_layers: i64,
    pub encoder_inner: i64,
    pub decoder_inner: i64,
    pub activation: Activation,
    pub context: i64,
    pub horizon: i64,
    /// The steps of the context the decoder reads before the horizon.
    pub label: i64,
    pub lags: Vec<i64>,
    pub moving_average: i64,
    /// What the logarithm of a sequence's length is multiplied by for the delays attention aggregates.
    pub autocorrelation_factor: f64,
    pub time_features: i64,
    pub cardinalities: Vec<i64>,
    pub embedding_dims: Vec<i64>,
    /// The width of a step's input: its lags, the static features, and the time features.
    pub feature_size: i64,
    pub minimum_scale: f32,
}

impl AutoformerConfig {
    /// Returns the past steps a forecast reads: the context and the longest lag before it.
    #[must_use]
    pub fn past_length(&self) -> i64 {
        self.context + self.lags.iter().copied().max().unwrap_or(0)
    }
}

/// Returns `input`, `[1, L, C]`, as a trend, its moving average over `kernel` steps with its ends repeated, and the seasonal rest, as `AutoformerSeriesDecompositionLayer` splits it.
fn decompose(input: &Tensor, kernel: i64) -> SingeResult<(Tensor, Tensor)> {
    let &[batch, length, channels] = input.shape().dimensions() else {
        return Err(SingeError::invalid_argument(
            "a decomposed series, [B, L, C]",
        ));
    };
    let pads = (kernel - 1) / 2;
    let edge = |start: i64| {
        input
            .narrow(1, start, 1)?
            .expand(Shape::create([batch, pads, channels])?)
    };
    let padded = Tensor::concat(&[edge(0)?, input.clone(), edge(length - 1)?], 1)?;
    let trend = padded
        .transpose(1, 2)?
        .contiguous()?
        .unfold(2, kernel, 1)?
        .mean(&[-1], false)?
        .transpose(1, 2)?;
    Ok((input.sub(&trend)?, trend))
}

/// `AutoformerLayernorm`: a layer norm, less its mean over time.
fn seasonal_norm(input: &Tensor, norm: &LayerNorm, state: &ModelState) -> SingeResult<Tensor> {
    let normed = input.apply(norm, state)?;
    normed.sub(&normed.mean(&[1], true)?)
}

/// Autocorrelation attention: the queries' and keys' cross-correlation over every delay, by FFT, picks the delays it peaks at on average, and the output is the values rolled back by each, weighted by the softmax of the peaks.
#[derive(Debug, Clone)]
struct AutoCorrelation {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    heads: i64,
    factor: f64,
}

impl AutoCorrelation {
    fn create(schema: &mut Schema<'_>, config: &AutoformerConfig) -> SingeResult<Self> {
        let width = config.width;
        Ok(Self {
            query: schema.linear("query", width, width, true)?,
            key: schema.linear("key", width, width, true)?,
            value: schema.linear("value", width, width, true)?,
            output: schema.linear("output", width, width, true)?,
            heads: config.heads,
            factor: config.autocorrelation_factor,
        })
    }

    /// Attends from `hidden`, `[1, L, W]`, to `context`, `[1, S, W]`, of one series, as `AutoformerAttention` does at inference.
    /// A query longer than the context keeps only as many of its first rows as it is longer, and keys and values gain as many rows of zeros, as `transformers` writes it; a shorter one reads as many keys and values as it has rows.
    fn forward(
        &self,
        hidden: &Tensor,
        context: &Tensor,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let &[_, length, width] = hidden.shape().dimensions() else {
            return Err(SingeError::invalid_argument(
                "autocorrelation queries, [1, L, W]",
            ));
        };
        let sources = context.shape().dimensions()[1];
        let head_dim = width / self.heads;
        let heads = |input: &Tensor, linear: &Linear, rows: i64| {
            input
                .apply(linear, state)?
                .reshape([rows, self.heads, head_dim])?
                .transpose(0, 1)
        };
        let query = heads(hidden, &self.query, length)?;
        let key = heads(context, &self.key, sources)?;
        let value = heads(context, &self.value, sources)?;
        let (query, key, value) = if length > sources {
            let rows = length - sources;
            let zeros = Tensor::zeroed_on(
                hidden.device(),
                value.data_type(),
                Shape::create([self.heads, rows, head_dim])?,
            )?;
            (
                query.narrow(1, 0, rows)?,
                Tensor::concat(&[key, zeros.clone()], 1)?,
                Tensor::concat(&[value, zeros], 1)?,
            )
        } else {
            (
                query,
                key.narrow(1, 0, length)?,
                value.narrow(1, 0, length)?,
            )
        };
        // Each signal along time, zero-padded to the queries' length, as `rfft(n=L)` pads it.
        let spectrum = |input: &Tensor| -> SingeResult<Tensor> {
            let rows = input.shape().dimensions()[1];
            let signal = input.transpose(1, 2)?.cast(DataType::F32)?;
            let signal = if rows < length {
                signal.pad(
                    &[Padding::ZERO, Padding::ZERO, Padding::new(0, length - rows)],
                    0.0,
                )?
            } else {
                signal
            };
            signal.contiguous()?.rfft()
        };
        let (queries, keys) = (spectrum(&query)?, spectrum(&key)?);
        let part = |spectrum: &Tensor, index: i64| spectrum.select(-1, index);
        let (query_real, query_imaginary) = (part(&queries, 0)?, part(&queries, 1)?);
        let (key_real, key_imaginary) = (part(&keys, 0)?, part(&keys, 1)?);
        let product = Tensor::stack(
            &[
                query_real
                    .mul(&key_real)?
                    .add(&query_imaginary.mul(&key_imaginary)?)?,
                query_imaginary
                    .mul(&key_real)?
                    .sub(&query_real.mul(&key_imaginary)?)?,
            ],
            -1,
        )?;
        let correlations = product.irfft(length)?.mean(&[0, 1], false)?;
        let count = (self.factor * (length as f64).ln()) as i64;
        let (peaks, delays) = correlations.topk(count, -1, true, true)?;
        let peaks = peaks.to_vec::<f32>()?;
        let delays = delays.to_vec::<i64>()?;
        let most = peaks.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exponents = peaks
            .iter()
            .map(|peak| (peak - most).exp())
            .collect::<Vec<_>>();
        let total = exponents.iter().sum::<f32>();
        let mut aggregated: Option<Tensor> = None;
        for (&delay, exponent) in delays.iter().zip(exponents) {
            let rolled = if delay == 0 {
                value.clone()
            } else {
                Tensor::concat(
                    &[
                        value.narrow(1, delay, length - delay)?,
                        value.narrow(1, 0, delay)?,
                    ],
                    1,
                )?
            };
            let weighted = rolled.mul_scalar(f64::from(exponent / total))?;
            aggregated = Some(match aggregated {
                Some(sum) => sum.add(&weighted)?,
                None => weighted,
            });
        }
        aggregated
            .ok_or_else(|| SingeError::invalid_argument("autocorrelation over no delays"))?
            .transpose(0, 1)?
            .reshape([1, length, width])?
            .apply(&self.output, state)
    }
}

#[derive(Debug, Clone)]
struct EncoderBlock {
    attention: AutoCorrelation,
    attention_norm: LayerNorm,
    feed_forward: Mlp,
    norm: LayerNorm,
}

#[derive(Debug, Clone)]
struct DecoderBlock {
    attention: AutoCorrelation,
    attention_norm: LayerNorm,
    cross_attention: AutoCorrelation,
    cross_attention_norm: LayerNorm,
    feed_forward: Mlp,
    norm: LayerNorm,
    trend: Conv1d,
}

/// Autoformer, from a series' scaled lags and features to each step's Student-T parameters.
#[derive(Debug, Clone)]
pub struct AutoformerLayers {
    categories: Vec<Embedding>,
    encoder_embedding: StepEmbedding,
    encoder: Vec<EncoderBlock>,
    decoder_embedding: StepEmbedding,
    decoder: Vec<DecoderBlock>,
    seasonality: Linear,
    parameters: [Linear; 3],
    config: AutoformerConfig,
}

impl AutoformerLayers {
    pub fn create(schema: &mut Schema<'_>, config: &AutoformerConfig) -> SingeResult<Self> {
        let width = config.width;
        let mlp = |schema: &mut Schema<'_>, inner: i64| {
            schema.mlp(
                "feed_forward",
                [width, inner, width],
                config.activation,
                true,
            )
        };
        let encoder = (0..config.encoder_layers)
            .map(|index| {
                let mut block = schema.scope(format!("encoder.blocks.{index}"));
                Ok(EncoderBlock {
                    attention: AutoCorrelation::create(&mut block.scope("attention"), config)?,
                    attention_norm: block.layer_norm("attention_norm", width, 1e-5)?,
                    feed_forward: mlp(&mut block, config.encoder_inner)?,
                    norm: block.layer_norm("norm", width, 1e-5)?,
                })
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let decoder = (0..config.decoder_layers)
            .map(|index| {
                let mut block = schema.scope(format!("decoder.blocks.{index}"));
                Ok(DecoderBlock {
                    attention: AutoCorrelation::create(&mut block.scope("attention"), config)?,
                    attention_norm: block.layer_norm("attention_norm", width, 1e-5)?,
                    cross_attention: AutoCorrelation::create(
                        &mut block.scope("cross_attention"),
                        config,
                    )?,
                    cross_attention_norm: block.layer_norm("cross_attention_norm", width, 1e-5)?,
                    feed_forward: mlp(&mut block, config.decoder_inner)?,
                    norm: block.layer_norm("norm", width, 1e-5)?,
                    trend: block.conv1d(
                        "trend",
                        width,
                        config.feature_size,
                        3,
                        Conv1dConfig::default(),
                        false,
                    )?,
                })
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let positions = config.context + config.horizon;
        Ok(Self {
            categories: lagged::category_embeddings(
                schema,
                &config.cardinalities,
                &config.embedding_dims,
            )?,
            encoder_embedding: StepEmbedding::create(
                &mut schema.scope("encoder"),
                config.feature_size,
                width,
                positions,
            )?,
            encoder,
            decoder_embedding: StepEmbedding::create(
                &mut schema.scope("decoder"),
                config.feature_size,
                width,
                positions,
            )?,
            decoder,
            seasonality: schema.linear("decoder.seasonality", width, config.feature_size, true)?,
            parameters: [
                schema.linear("parameters.0", config.feature_size, 1, true)?,
                schema.linear("parameters.1", config.feature_size, 1, true)?,
                schema.linear("parameters.2", config.feature_size, 1, true)?,
            ],
            config: config.clone(),
        })
    }

    fn encode(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let kernel = self.config.moving_average;
        let mut hidden = self.encoder_embedding.forward(input, 0, state)?;
        for block in &self.encoder {
            let attended = block.attention.forward(&hidden, &hidden, state)?;
            let (seasonal, _) = decompose(
                &hidden.add(&attended)?.apply(&block.attention_norm, state)?,
                kernel,
            )?;
            let (seasonal, _) = decompose(
                &seasonal.add(&seasonal.apply(&block.feed_forward, state)?)?,
                kernel,
            )?;
            hidden = seasonal_norm(&seasonal, &block.norm, state)?;
        }
        Ok(hidden)
    }

    /// Returns the decoder's projected seasonal output plus its accumulated trend, `[1, L, F]`.
    fn decode(
        &self,
        input: &Tensor,
        trend: &Tensor,
        encoded: &Tensor,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let config = &self.config;
        let kernel = config.moving_average;
        let mut hidden =
            self.decoder_embedding
                .forward(input, config.context - config.label, state)?;
        let mut trend = trend.clone();
        for block in &self.decoder {
            let attended = block.attention.forward(&hidden, &hidden, state)?;
            let (seasonal, first) = decompose(&hidden.add(&attended)?, kernel)?;
            let seasonal = seasonal.apply(&block.attention_norm, state)?;
            let crossed = block.cross_attention.forward(&seasonal, encoded, state)?;
            let (seasonal, second) = decompose(&seasonal.add(&crossed)?, kernel)?;
            let seasonal = seasonal.apply(&block.cross_attention_norm, state)?;
            let (seasonal, third) = decompose(
                &seasonal.add(&seasonal.apply(&block.feed_forward, state)?)?,
                kernel,
            )?;
            hidden = seasonal_norm(&seasonal, &block.norm, state)?;
            let trends = lagged::circular_pad(&first.add(&second)?.add(&third)?.transpose(1, 2)?)?;
            trend = trend.add(&trends.apply(&block.trend, state)?.transpose(1, 2)?)?;
        }
        hidden.apply(&self.seasonality, state)?.add(&trend)
    }

    /// Returns each future step's Student-T degrees of freedom, location, and scale, before the series' scale, from the scaled lags of the context, `[1, C, lags]`, the time features of the context and of the decoder's steps, `[1, C, T]` and `[1, label + H, T]`, the log of the scale, and the static categories.
    fn parameters(
        &self,
        scaled_lags: &Tensor,
        [context_time, decoder_time]: [&Tensor; 2],
        log_scale: f32,
        categories: &[i64],
        state: &ModelState,
    ) -> SingeResult<[Tensor; 3]> {
        let config = &self.config;
        let statics = lagged::statics(
            &self.categories,
            categories,
            &[],
            log_scale,
            scaled_lags,
            state,
        )?;
        let features = |time: &Tensor| -> SingeResult<Tensor> {
            let &[_, length, _] = time.shape().dimensions() else {
                return Err(SingeError::invalid_argument("time features, [1, L, T]"));
            };
            let width = statics.shape().dimensions()[2];
            Tensor::concat(
                &[
                    statics.expand(Shape::create([1, length, width])?)?,
                    time.clone(),
                ],
                2,
            )
        };
        let encoded = self.encode(
            &Tensor::concat(&[scaled_lags.clone(), features(context_time)?], 2)?,
            state,
        )?;
        let (seasonal, trend) = decompose(scaled_lags, config.moving_average)?;
        let lags = scaled_lags.shape().dimensions()[2];
        let start = config.context - config.label;
        let mean =
            scaled_lags
                .mean(&[1], true)?
                .expand(Shape::create([1, config.horizon, lags])?)?;
        let zeros = Tensor::zeroed_on(
            scaled_lags.device(),
            scaled_lags.data_type(),
            Shape::create([1, config.horizon, lags])?,
        )?;
        let decoder_features = features(decoder_time)?;
        let input = Tensor::concat(
            &[
                Tensor::concat(&[seasonal.narrow(1, start, config.label)?, zeros], 1)?,
                decoder_features.clone(),
            ],
            2,
        )?;
        let trend = Tensor::concat(
            &[
                Tensor::concat(&[trend.narrow(1, start, config.label)?, mean], 1)?,
                decoder_features,
            ],
            2,
        )?;
        let output = self.decode(&input, &trend, &encoded, state)?.narrow(
            1,
            config.label,
            config.horizon,
        )?;
        let parameter = |index: usize| {
            output
                .apply(&self.parameters[index], state)?
                .reshape([config.horizon])?
                .cast(DataType::F32)
        };
        lagged::student_t([parameter(0)?, parameter(1)?, parameter(2)?])
    }
}

pub type Autoformer = Model<AutoformerLayers>;

impl Forecaster for Autoformer {
    /// Forecasts one univariate series as `AutoformerForPrediction.generate` does, returning the Student-T it samples each step from, or with `draws`, paths of independent draws from each step's distribution, as `generate` samples them.
    fn forecast(&self, series: &Series, draws: Option<PathSampling>) -> Result<Forecast> {
        series.refuse_extras("Autoformer")?;
        series.refuse_missing("Autoformer")?;
        let config = &self.layer().config;
        let PastSeries::Univariate(values) = &series.past_values else {
            return Err(Error::unsupported(
                "an Autoformer forecast of several channels",
            ));
        };
        let (past, context, horizon, label) = (
            lagged::size(config.past_length(), "past length")?,
            lagged::size(config.context, "context")?,
            lagged::size(config.horizon, "horizon")?,
            lagged::size(config.label, "label length")?,
        );
        let values = lagged::last(values, past, "past values")?;
        let past_time = lagged::last(&series.past_time_features, past, "past time features")?;
        if series.future_time_features.len() != horizon
            || series.static_categorical_features.len() != config.cardinalities.len()
        {
            return Err(Error::invalid(
                "future time features or static categories of other sizes than the config's",
            ));
        }

        let scale = lagged::mean_scale(&values[past - context..], config.minimum_scale);
        let scaled = values.iter().map(|value| value / scale).collect::<Vec<_>>();
        let lags = config
            .lags
            .iter()
            .map(|&lag| lagged::size(lag, "lag"))
            .collect::<Result<Vec<_>>>()?;
        let lag_count =
            i64::try_from(lags.len()).map_err(|_| Error::invalid("a lag count past `i64`"))?;
        let state = self.state();
        let time_width = config.time_features;
        let scaled_lags = upload(
            state,
            [1, config.context, lag_count],
            lagged::at_lags(&scaled, &lags, past - context, context),
        )?;
        let context_time = upload(
            state,
            [1, config.context, time_width],
            lagged::time_rows(&past_time[past - context..], time_width)?,
        )?;
        let decoder_rows = [&past_time[past - label..], &series.future_time_features[..]].concat();
        let decoder_time = upload(
            state,
            [1, config.label + config.horizon, time_width],
            lagged::time_rows(&decoder_rows, time_width)?,
        )?;
        let [df, loc, spread] = self.layer().parameters(
            &scaled_lags,
            [&context_time, &decoder_time],
            scale.ln(),
            &series.static_categorical_features,
            state,
        )?;
        // The distribution mapped back to the series' scale, as `AffineTransformed` maps it.
        let (df, loc, spread) = (
            df.to_vec::<f32>()?,
            loc.mul_scalar(f64::from(scale))?.to_vec::<f32>()?,
            spread.mul_scalar(f64::from(scale))?.to_vec::<f32>()?,
        );
        Ok(match draws {
            None => Forecast::StudentT {
                df,
                loc,
                scale: spread,
            },
            Some(PathSampling { count, seed }) => {
                let mut uniforms = UniformSource::new(seed);
                Forecast::Samples(
                    (0..count)
                        .map(|_| {
                            (0..horizon)
                                .map(|step| loc[step] + spread[step] * uniforms.student_t(df[step]))
                                .collect()
                        })
                        .collect(),
                )
            }
        })
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "autoformer",
    formats: &[(Format::Transformers, &["autoformer"])],
    loaders: &[HeadLoader::TimeSeriesForecasting(|loader| {
        Ok(Box::new(load(loader)?))
    })],
};

pub fn load(loader: &mut Loader<'_>) -> Result<Autoformer> {
    let config: AutoformerConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        AutoformerLayers::create(schema, &config)
    })?;
    loader.load_whole(&model, &transformers::rules())
}
