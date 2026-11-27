//! The time series transformer: an encoder and a causal decoder of post-norm blocks over each step's features, from which three projections give a Student-T distribution per future step.
//! A series is scaled by its mean absolute value over the context, and each step reads the series at a set of lags, the series' static features, and its time features.
//! Informer is this transformer with `ProbSparse` self-attention and, after each encoder block but the last, a distilling convolution that halves the steps.
//! `generate` samples each step and feeds the sample back through the lags; a forecast here feeds back each step's location instead, so only its first step's distribution is `generate`'s, and a series with its future values gives each step's distribution after the real earlier steps, as `forward` does with `future_values`.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Error as SingeError, Result as SingeResult, Shape, Tensor,
    nn::{
        Activation, BatchNorm, Conv1d, Conv1dConfig, Embedding, LayerNorm, Linear, Mlp, Model,
        ModelState,
    },
    shape::spatial::Padding,
    tensor::AttentionOptions,
};

use crate::{
    assembly::{Attention, AttentionSpec, merge_heads, split_heads},
    error::{Error, Result},
    format::Loader,
    head::{Draws, Forecast, Forecaster, HeadLoader, PastValues, Series, Uniforms, upload},
    model::{
        ModelEntry,
        time_series::lagged::{self, StepEmbedding},
    },
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimeSeriesTransformerConfig {
    pub width: i64,
    pub encoder_heads: i64,
    pub decoder_heads: i64,
    pub encoder_layers: i64,
    pub decoder_layers: i64,
    pub encoder_inner: i64,
    pub decoder_inner: i64,
    pub activation: Activation,
    pub context: i64,
    pub horizon: i64,
    pub lags: Vec<i64>,
    pub time_features: i64,
    pub cardinalities: Vec<i64>,
    pub embedding_dims: Vec<i64>,
    pub static_reals: i64,
    /// The width of a step's input: its lags, the static features, and the time features.
    pub feature_size: i64,
    pub minimum_scale: f32,
    /// Informer's `ProbSparse` self-attention and its sampling factor, or `None` for full attention.
    pub sparse_factor: Option<i64>,
    /// Whether Informer's distilling convolutions follow each encoder block but the last.
    pub distil: bool,
}

impl TimeSeriesTransformerConfig {
    /// Returns the past steps a forecast reads: the context and the longest lag before it.
    #[must_use]
    pub fn past_length(&self) -> i64 {
        self.context + self.lags.iter().copied().max().unwrap_or(0)
    }
}

#[derive(Debug, Clone)]
struct EncoderBlock {
    attention: Attention,
    attention_norm: LayerNorm,
    feed_forward: Mlp,
    feed_forward_norm: LayerNorm,
}

/// Informer's distilling layer: a convolution over time that wraps around, batch norm, ELU, and a max pool that halves the steps.
#[derive(Debug, Clone)]
struct Distil {
    convolution: Conv1d,
    norm: BatchNorm,
}

impl Distil {
    /// Halves `hidden`, `[1, L, W]`, to `[1, ⌈L / 2⌉, W]`.
    fn forward(&self, hidden: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        lagged::circular_pad(&hidden.transpose(1, 2)?)?
            .apply(&self.convolution, state)?
            .apply(&self.norm, state)?
            .elu(1.0)?
            .max_pool1d(3, 2, Padding::new(1, 1), 1, false)?
            .transpose(1, 2)
    }
}

/// A block's self-attention: full, or Informer's `ProbSparse` with its sampling factor.
///
/// `ProbSparse` attention attends only from the `min(factor · ⌈ln(1 + L)⌉, L)` queries whose scores' maximum stands farthest above their mean, and gives the others the values' mean, or in the decoder their running sum.
/// `transformers` draws the keys those measures read with `torch.randint`, as many as `min(factor · L · ⌈ln(1 + L)⌉, L)`, which for self-attention is every key; here each key is read once.
fn self_attend(
    attention: &Attention,
    hidden: &Tensor,
    heads: i64,
    sparse_factor: Option<i64>,
    causal: bool,
    state: &ModelState,
) -> SingeResult<Tensor> {
    let options = AttentionOptions {
        causal,
        ..AttentionOptions::default()
    };
    let Some(factor) = sparse_factor else {
        return attention.attend(hidden, hidden, hidden, None, options, state);
    };
    let split = |linear: &Linear| split_heads(&hidden.apply(linear, state)?, heads);
    let (query, key, value) = (
        split(attention.query())?,
        split(attention.key())?,
        split(attention.value())?,
    );
    let &[_, _, length, head] = query.shape().dimensions() else {
        return Err(SingeError::invalid_argument(
            "attention heads, [B, H, L, d]",
        ));
    };
    let attended = query.attend(&key, &value, None, None, options)?;
    let chosen = (factor * ((length as f64).ln_1p().ceil() as i64)).min(length);
    let output = if chosen < length {
        let scores = query
            .mul_scalar(1.0 / (head as f64).sqrt())?
            .matmul(&key.transpose(2, 3)?)?;
        let measure = scores
            .max_reduce(&[-1], true)?
            .sub(&scores.sum(&[-1], true)?.div_scalar(length as f64)?)?;
        let (top, _) = measure.squeeze(-1)?.topk(chosen, -1, true, true)?;
        let selected = measure.greater_equal(&top.narrow(-1, chosen - 1, 1)?.unsqueeze(-1)?)?;
        let rest = if causal {
            value
                .cast(DataType::F32)?
                .cumsum(-2)?
                .cast(value.data_type())?
        } else {
            value.mean(&[-2], true)?.expand(value.shape().clone())?
        };
        Tensor::where_select(&selected, &attended, &rest)?
    } else {
        attended
    };
    merge_heads(&output)?.apply(attention.output(), state)
}

#[derive(Debug, Clone)]
struct DecoderBlock {
    attention: Attention,
    attention_norm: LayerNorm,
    cross_attention: Attention,
    cross_attention_norm: LayerNorm,
    feed_forward: Mlp,
    feed_forward_norm: LayerNorm,
}

/// The time series transformer, from a series' scaled lags and features to each step's Student-T parameters.
#[derive(Debug, Clone)]
pub struct TimeSeriesTransformerLayers {
    categories: Vec<Embedding>,
    encoder_embedding: StepEmbedding,
    encoder: Vec<EncoderBlock>,
    distil: Vec<Distil>,
    decoder_embedding: StepEmbedding,
    decoder: Vec<DecoderBlock>,
    parameters: [Linear; 3],
    config: TimeSeriesTransformerConfig,
}

impl TimeSeriesTransformerLayers {
    pub fn create(
        schema: &mut Schema<'_>,
        config: &TimeSeriesTransformerConfig,
    ) -> SingeResult<Self> {
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
                    attention: Attention::create(
                        &mut block.scope("attention"),
                        AttentionSpec::square(width, config.encoder_heads),
                    )?,
                    attention_norm: block.layer_norm("attention_norm", width, 1e-5)?,
                    feed_forward: mlp(&mut block, config.encoder_inner)?,
                    feed_forward_norm: block.layer_norm("feed_forward_norm", width, 1e-5)?,
                })
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let decoder = (0..config.decoder_layers)
            .map(|index| {
                let mut block = schema.scope(format!("decoder.blocks.{index}"));
                let spec = AttentionSpec::square(width, config.decoder_heads);
                Ok(DecoderBlock {
                    attention: Attention::create(&mut block.scope("attention"), spec)?,
                    attention_norm: block.layer_norm("attention_norm", width, 1e-5)?,
                    cross_attention: Attention::create(&mut block.scope("cross_attention"), spec)?,
                    cross_attention_norm: block.layer_norm("cross_attention_norm", width, 1e-5)?,
                    feed_forward: mlp(&mut block, config.decoder_inner)?,
                    feed_forward_norm: block.layer_norm("feed_forward_norm", width, 1e-5)?,
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
            distil: if config.distil {
                (0..config.encoder_layers - 1)
                    .map(|index| {
                        let mut layer = schema.scope(format!("encoder.distil.{index}"));
                        Ok(Distil {
                            convolution: layer.conv1d(
                                "convolution",
                                width,
                                width,
                                3,
                                Conv1dConfig::default(),
                                true,
                            )?,
                            norm: layer.batch_norm("norm", width, 1e-5)?,
                        })
                    })
                    .collect::<SingeResult<Vec<_>>>()?
            } else {
                Vec::new()
            },
            decoder_embedding: StepEmbedding::create(
                &mut schema.scope("decoder"),
                config.feature_size,
                width,
                positions,
            )?,
            decoder,
            parameters: [
                schema.linear("parameters.0", width, 1, true)?,
                schema.linear("parameters.1", width, 1, true)?,
                schema.linear("parameters.2", width, 1, true)?,
            ],
            config: config.clone(),
        })
    }

    fn encode(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let mut hidden = self.encoder_embedding.forward(input, 0, state)?;
        for (index, block) in self.encoder.iter().enumerate() {
            let attended = self_attend(
                &block.attention,
                &hidden,
                self.config.encoder_heads,
                self.config.sparse_factor,
                false,
                state,
            )?;
            hidden = hidden.add(&attended)?.apply(&block.attention_norm, state)?;
            hidden = hidden
                .add(&hidden.apply(&block.feed_forward, state)?)?
                .apply(&block.feed_forward_norm, state)?;
            if let Some(distil) = self.distil.get(index) {
                hidden = distil.forward(&hidden, state)?;
            }
        }
        Ok(hidden)
    }

    /// Returns each of `input`'s steps' Student-T degrees of freedom, location, and scale, `[P, L]`, before the series' scale, from the decoder's steps, `[P, L, F]`, which read `encoded`, `[P, C, W]`.
    fn decode(
        &self,
        input: &Tensor,
        encoded: &Tensor,
        state: &ModelState,
    ) -> SingeResult<[Tensor; 3]> {
        let &[rows, length, _] = input.shape().dimensions() else {
            return Err(SingeError::invalid_argument("decoder steps, [P, L, F]"));
        };
        let mut hidden = self
            .decoder_embedding
            .forward(input, self.config.context, state)?;
        for block in &self.decoder {
            let attended = self_attend(
                &block.attention,
                &hidden,
                self.config.decoder_heads,
                self.config.sparse_factor,
                true,
                state,
            )?;
            hidden = hidden.add(&attended)?.apply(&block.attention_norm, state)?;
            let crossed = block.cross_attention.attend(
                &hidden,
                encoded,
                encoded,
                None,
                AttentionOptions::default(),
                state,
            )?;
            hidden = hidden
                .add(&crossed)?
                .apply(&block.cross_attention_norm, state)?;
            hidden = hidden
                .add(&hidden.apply(&block.feed_forward, state)?)?
                .apply(&block.feed_forward_norm, state)?;
        }

        let parameter = |index: usize| {
            hidden
                .apply(&self.parameters[index], state)?
                .reshape([rows, length])?
                .cast(DataType::F32)
        };
        lagged::student_t([parameter(0)?, parameter(1)?, parameter(2)?])
    }
}

pub type TimeSeriesTransformer = Model<TimeSeriesTransformerLayers>;

impl Forecaster for TimeSeriesTransformer {
    /// Forecasts one univariate series as the module describes, returning each step's Student-T distribution mapped back to the series' scale, or with `draws`, paths that feed back each step's draw, as `generate` samples them.
    fn forecast(&self, series: &Series, draws: Option<Draws>) -> Result<Forecast> {
        series.refuse_missing("the time series transformer")?;
        let layers = self.layer();
        let config = &layers.config;
        let PastValues::Univariate(values) = &series.past_values else {
            return Err(Error::unsupported(
                "a time series transformer forecast of several channels",
            ));
        };
        let (past, context, horizon) = (
            lagged::size(config.past_length(), "past length")?,
            lagged::size(config.context, "context")?,
            lagged::size(config.horizon, "horizon")?,
        );
        let values = lagged::last(values, past, "past values")?;
        let past_time = lagged::last(&series.past_time_features, past, "past time features")?;
        if series.future_time_features.len() != horizon
            || series.static_categorical_features.len() != config.cardinalities.len()
            || i64::try_from(series.static_real_features.len()).ok() != Some(config.static_reals)
            || series
                .future_values
                .as_ref()
                .is_some_and(|future| future.len() != horizon)
        {
            return Err(Error::invalid(
                "future time features, static features, or future values of other sizes than the config's",
            ));
        }

        let scale = lagged::mean_scale(&values[past - context..], config.minimum_scale);
        let mut scaled = values.iter().map(|value| value / scale).collect::<Vec<_>>();
        let lags = config
            .lags
            .iter()
            .map(|&lag| lagged::size(lag, "lag"))
            .collect::<Result<Vec<_>>>()?;
        let to_i64 =
            |value: usize| i64::try_from(value).map_err(|_| Error::invalid("a count past `i64`"));
        let state = self.state();
        let time_width = config.time_features;
        let context_time = upload(
            state,
            [1, config.context, time_width],
            lagged::time_rows(&past_time[past - context..], time_width)?,
        )?;
        let statics = lagged::statics(
            &layers.categories,
            &series.static_categorical_features,
            &series.static_real_features,
            scale.ln(),
            &context_time,
            state,
        )?;
        // Each path's steps' features: lags `[P, L, lags]`, then the static and time features every path shares.
        let inputs =
            |lag_rows: Vec<f32>, time: &Tensor, paths: usize, length: usize| -> Result<Tensor> {
                let [paths, length, lag_count] =
                    [to_i64(paths)?, to_i64(length)?, to_i64(lags.len())?];
                let width = |tensor: &Tensor| tensor.shape().dimensions()[2];
                Ok(Tensor::concat(
                    &[
                        upload(state, [paths, length, lag_count], lag_rows)?,
                        statics.expand(Shape::create([paths, length, width(&statics)])?)?,
                        time.expand(Shape::create([paths, length, width(time)])?)?,
                    ],
                    2,
                )?)
            };
        let encoded = layers.encode(
            &inputs(
                lagged::at_lags(&scaled, &lags, past - context, context),
                &context_time,
                1,
                context,
            )?,
            state,
        )?;
        let future_time = upload(
            state,
            [1, config.horizon, time_width],
            lagged::time_rows(&series.future_time_features, time_width)?,
        )?;
        let mapped =
            |values: &[f32]| -> Vec<f32> { values.iter().map(|value| value * scale).collect() };
        if let Some(future) = &series.future_values {
            if draws.is_some() {
                return Err(Error::invalid(
                    "sampled paths of a series with its future values",
                ));
            }

            // Teacher-forced: every step reads the real values before it.
            scaled.extend(future.iter().map(|value| value / scale));
            let [df, loc, spread] = layers
                .decode(
                    &inputs(
                        lagged::at_lags(&scaled, &lags, past, horizon),
                        &future_time,
                        1,
                        horizon,
                    )?,
                    &encoded,
                    state,
                )?
                .map(|parameter| parameter.to_vec::<f32>());
            // The distribution mapped back to the series' scale, as `AffineTransformed` maps it.
            return Ok(Forecast::StudentT {
                df: df?,
                loc: mapped(&loc?),
                scale: mapped(&spread?),
            });
        }

        // Each step decodes every path's prefix, as `generate` does, and its last position's distribution gives the path's next value: a draw where paths are sampled, or else the location.
        let (paths, mut uniforms) = draws.map_or((1, None), |Draws { count, seed }| {
            (count, Some(Uniforms::new(seed)))
        });
        let mut sequences = vec![scaled; paths];
        let mut steps = [const { Vec::new() }; 3];
        let &[_, sources, width] = encoded.shape().dimensions() else {
            return Err(Error::invalid("encoder states, [1, C, W]"));
        };
        let encoded = encoded
            .expand(Shape::create([to_i64(paths)?, sources, width])?)?
            .contiguous()?;
        for step in 1..=horizon {
            let length = to_i64(step)?;
            let lag_rows = sequences
                .iter()
                .flat_map(|sequence| lagged::at_lags(sequence, &lags, past, step))
                .collect();
            let decoded = layers.decode(
                &inputs(lag_rows, &future_time.narrow(1, 0, length)?, paths, step)?,
                &encoded,
                state,
            )?;
            let [df, loc, spread] = decoded.map(|parameter| -> Result<Vec<f32>> {
                Ok(parameter
                    .narrow(1, length - 1, 1)?
                    .contiguous()?
                    .to_vec::<f32>()?)
            });
            let [df, loc, spread] = [df?, loc?, spread?];
            for (path, sequence) in sequences.iter_mut().enumerate() {
                sequence.push(uniforms.as_mut().map_or(loc[path], |uniforms| {
                    loc[path] + spread[path] * uniforms.student_t(df[path])
                }));
            }
            for (values, parameter) in steps.iter_mut().zip([&df, &loc, &spread]) {
                values.push(parameter[0]);
            }
        }

        Ok(if uniforms.is_some() {
            Forecast::Samples(
                sequences
                    .into_iter()
                    .map(|sequence| mapped(&sequence[past..]))
                    .collect(),
            )
        } else {
            let [df, loc, spread] = steps;
            Forecast::StudentT {
                df,
                loc: mapped(&loc),
                scale: mapped(&spread),
            }
        })
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "time_series_transformer",
    formats: &[(
        Format::Transformers,
        &["time_series_transformer", "informer"],
    )],
    loaders: &[HeadLoader::TimeSeriesForecasting(|loader| {
        Ok(Box::new(load(loader)?))
    })],
};

pub fn load(loader: &mut Loader<'_>) -> Result<TimeSeriesTransformer> {
    let config: TimeSeriesTransformerConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        TimeSeriesTransformerLayers::create(schema, &config)
    })?;
    loader.load_whole(&model, &transformers::rules())
}
