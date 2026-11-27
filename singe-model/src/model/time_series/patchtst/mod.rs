//! `PatchTST`: each channel of a multivariate series scaled by its own mean and deviation, cut into patches, and read on its own by pre-norm blocks with batch or layer norms; a class token's final state, the patches' mean, or every patch's state projects to the forecast.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Error as SingeError, Result as SingeResult, Shape, Tensor,
    nn::{Activation, BatchNorm, Layer, LayerNorm, Linear, Mlp, Model, ModelState, StateKey},
};

use crate::{
    assembly::{Attention, AttentionSpec},
    error::{Error, Result},
    format::Loader,
    head::{Draws, Forecast, Forecaster, HeadLoader, PastValues, Series, refuse_draws},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PatchTstConfig {
    pub context: i64,
    pub channels: i64,
    pub patch: i64,
    pub stride: i64,
    pub width: i64,
    pub heads: i64,
    pub layers: i64,
    pub inner: i64,
    pub activation: Activation,
    pub eps: f32,
    pub horizon: i64,
    /// What a channel's variance has added under the root, so a constant channel keeps a scale.
    pub minimum_scale: f64,
    pub norm: PatchNorm,
    pub pooling: PatchPooling,
}

/// The norm of a block's sublayers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatchNorm {
    /// A batch norm over the width, with its running statistics.
    Batch,
    Layer,
}

/// What the forecast's projection reads of a channel's final states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatchPooling {
    /// A class token's state, in front of the patches.
    Class,
    /// The patches' mean.
    Mean,
    /// Every patch's state, one after another.
    Flatten,
}

impl PatchTstConfig {
    /// Returns the patches the context cuts into, after the steps that do not fill one are dropped from its start.
    #[must_use]
    pub const fn patches(&self) -> i64 {
        (self.context - self.patch) / self.stride + 1
    }
}

/// A sublayer's norm over the width of `[N, L, W]`.
#[derive(Debug, Clone)]
enum Norm {
    Batch(BatchNorm),
    Layer(LayerNorm),
}

impl Norm {
    fn create(schema: &mut Schema<'_>, name: &str, config: &PatchTstConfig) -> SingeResult<Self> {
        Ok(match config.norm {
            PatchNorm::Batch => Self::Batch(schema.batch_norm(name, config.width, config.eps)?),
            PatchNorm::Layer => Self::Layer(schema.layer_norm(name, config.width, config.eps)?),
        })
    }

    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        match self {
            Self::Batch(norm) => input.transpose(1, 2)?.apply(norm, state)?.transpose(1, 2),
            Self::Layer(norm) => input.apply(norm, state),
        }
    }
}

/// A pre-norm block over each channel's patches, `[C, L, W]`.
#[derive(Debug, Clone)]
struct Block {
    attention_norm: Norm,
    attention: Attention,
    feed_forward_norm: Norm,
    feed_forward: Mlp,
}

impl Layer for Block {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let attended = self.attention_norm.forward(input, state)?;
        let hidden = input.add(
            &self
                .attention
                .forward(&attended, &attended, &attended, state)?,
        )?;
        hidden.add(
            &self
                .feed_forward_norm
                .forward(&hidden, state)?
                .apply(&self.feed_forward, state)?,
        )
    }
}

/// `PatchTST`: from scaled channels' patches, `[C, P, patch]`, to each channel's forecast, `[C, H]`, still scaled.
#[derive(Debug, Clone)]
pub struct PatchTstLayers {
    embedding: Linear,
    positions: StateKey,
    class: Option<StateKey>,
    blocks: Vec<Block>,
    head: Linear,
    config: PatchTstConfig,
}

impl PatchTstLayers {
    pub fn create(schema: &mut Schema<'_>, config: &PatchTstConfig) -> SingeResult<Self> {
        let width = config.width;
        let class_token = config.pooling == PatchPooling::Class;
        Ok(Self {
            embedding: schema.linear("embedding", config.patch, width, true)?,
            positions: schema.parameter(
                "positions",
                [config.patches() + i64::from(class_token), width],
            )?,
            class: class_token
                .then(|| schema.parameter("class", [1, 1, width]))
                .transpose()?,
            blocks: (0..config.layers)
                .map(|index| {
                    let mut block = schema.scope(format!("blocks.{index}"));
                    Ok(Block {
                        attention_norm: Norm::create(&mut block, "attention_norm", config)?,
                        attention: Attention::create(
                            &mut block.scope("attention"),
                            AttentionSpec::square(width, config.heads),
                        )?,
                        feed_forward_norm: Norm::create(&mut block, "feed_forward_norm", config)?,
                        feed_forward: block.mlp(
                            "feed_forward",
                            [width, config.inner, width],
                            config.activation,
                            true,
                        )?,
                    })
                })
                .collect::<SingeResult<Vec<_>>>()?,
            head: schema.linear(
                "head",
                if config.pooling == PatchPooling::Flatten {
                    width * config.patches()
                } else {
                    width
                },
                config.horizon,
                true,
            )?,
            config: config.clone(),
        })
    }
}

impl Layer for PatchTstLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let &[channels, _, _] = input.shape().dimensions() else {
            return Err(SingeError::invalid_argument(
                "PatchTST patches, [C, P, patch]",
            ));
        };
        let width = self.config.width;
        let patches = self.config.patches();
        let positions = state.tensor(&self.positions)?;
        let first = i64::from(self.class.is_some());
        let embedded = input
            .apply(&self.embedding, state)?
            .add(&positions.narrow(0, first, patches)?.unsqueeze(0)?)?;
        let mut hidden = match &self.class {
            Some(class) => {
                let class = state
                    .tensor(class)?
                    .add(&positions.narrow(0, 0, 1)?.unsqueeze(0)?)?
                    .expand(Shape::create([channels, 1, width])?)?;
                Tensor::concat(&[class, embedded], 1)?
            }
            None => embedded,
        };
        for block in &self.blocks {
            hidden = hidden.apply(block, state)?;
        }

        let pooled = match self.config.pooling {
            PatchPooling::Class => hidden.select(1, 0)?,
            PatchPooling::Mean => hidden.mean(&[1], false)?,
            PatchPooling::Flatten => hidden.reshape([channels, patches * width])?,
        };
        pooled.apply(&self.head, state)
    }
}

pub type PatchTst = Model<PatchTstLayers>;

impl Forecaster for PatchTst {
    /// Forecasts the last context of a multivariate series, `[T][C]`, as `PatchTSTForPrediction` forecasts it: each channel scaled by its mean and deviation over the context's observed values, and the forecast scaled back.
    fn forecast(&self, series: &Series, draws: Option<Draws>) -> Result<Forecast> {
        refuse_draws(draws, "PatchTST")?;
        series.refuse_extras("PatchTST")?;
        let config = &self.layer().config;
        let PastValues::Multivariate(steps) = &series.past_values else {
            return Err(Error::unsupported(
                "a PatchTST forecast of one channel without its axis",
            ));
        };
        let context =
            usize::try_from(config.context).map_err(|_| Error::invalid("a negative context"))?;
        let channels = usize::try_from(config.channels)
            .map_err(|_| Error::invalid("a negative channel count"))?;
        if steps.len() < context || steps.iter().any(|step| step.len() != channels) {
            return Err(Error::invalid(format!(
                "a PatchTST series of fewer than {context} steps of {channels} channels"
            )));
        }

        let observed = match &series.past_observed_mask {
            None => vec![1.0; context * channels],
            Some(PastValues::Multivariate(mask))
                if mask.len() == steps.len() && mask.iter().all(|step| step.len() == channels) =>
            {
                mask[mask.len() - context..]
                    .iter()
                    .flatten()
                    .copied()
                    .collect()
            }
            Some(_) => {
                return Err(Error::invalid(
                    "a PatchTST observed mask shaped otherwise than its past values",
                ));
            }
        };
        let values = steps[steps.len() - context..]
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<f32>>();
        let device = self.state().device()?;
        let shape = Shape::create([config.context, config.channels])?;
        let series = Tensor::from_vec(device, shape.clone(), values)?;
        let observed = Tensor::from_vec(device, shape, observed)?;
        // `PatchTSTStdScaler`: each channel's mean over its observed values, and their deviation with the minimum scale under the root.
        let count = observed.sum(&[0], true)?.clamp(1.0, f32::MAX)?;
        let loc = series.mul(&observed)?.sum(&[0], true)?.div(&count)?;
        let centered = series.sub(&loc)?;
        let deviations = centered.mul(&observed)?;
        let scale = deviations
            .mul(&deviations)?
            .sum(&[0], true)?
            .div(&count)?
            .add_scalar(config.minimum_scale)?
            .sqrt()?;
        let start = config.context - (config.patch + config.stride * (config.patches() - 1));
        let patches = centered
            .div(&scale)?
            .narrow(0, start, config.context - start)?
            .transpose(0, 1)?
            .contiguous()?
            .unfold(1, config.patch, config.stride)?
            .cast(self.state().floating_type()?)?;
        let forecast = self
            .forward(&patches)?
            .cast(DataType::F32)?
            .transpose(0, 1)?
            .mul(&scale)?
            .add(&loc)?;
        let values = forecast.contiguous()?.to_vec::<f32>()?;
        Ok(Forecast::Points(
            values.chunks(channels).map(<[f32]>::to_vec).collect(),
        ))
    }
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "patchtst",
    formats: &[(Format::Transformers, &["patchtst"])],
    loaders: &[HeadLoader::TimeSeriesForecasting(|loader| {
        Ok(Box::new(load(loader)?))
    })],
};

pub fn load(loader: &mut Loader<'_>) -> Result<PatchTst> {
    let config: PatchTstConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        PatchTstLayers::create(schema, &config)
    })?;
    loader.load_whole(&model, &transformers::rules())
}
