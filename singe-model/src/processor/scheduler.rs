//! `diffusers`' noise schedulers: the timesteps a pipeline denoises at and each step from one sample to the next, computed on the host in `F32` as `diffusers` computes them with `PyTorch`'s CPU tensors.

use serde::Deserialize;
use singe::Tensor;

use crate::{
    error::{Error, Result},
    source::Source,
};

const fn train_timesteps() -> usize {
    1000
}

const fn one() -> f32 {
    1.0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum SchedulerClass {
    #[serde(rename = "DDPMScheduler")]
    Ddpm,
    #[serde(rename = "PNDMScheduler")]
    Pndm,
    #[serde(rename = "EulerDiscreteScheduler")]
    Euler,
    #[serde(rename = "EulerAncestralDiscreteScheduler")]
    EulerAncestral,
    #[serde(rename = "HeunDiscreteScheduler")]
    Heun,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BetaSchedule {
    #[default]
    Linear,
    ScaledLinear,
    SquaredcosCapV2,
    Sigmoid,
    Exp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PredictionType {
    #[default]
    Epsilon,
    Sample,
    VPrediction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VarianceType {
    #[default]
    FixedSmall,
    FixedSmallLog,
    FixedLarge,
    FixedLargeLog,
    Learned,
    LearnedRange,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimestepSpacing {
    Linspace,
    Leading,
    Trailing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterpolationType {
    #[default]
    Linear,
    LogLinear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinalSigmasType {
    #[default]
    Zero,
    SigmaMin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimestepType {
    #[default]
    Discrete,
    Continuous,
}

/// A scheduler's `scheduler_config.json`, with `diffusers`' defaults; the fields whose defaults differ between classes stay `None` until the class resolves them.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "the fields are `scheduler_config.json`'s"
)]
pub struct SchedulerConfig {
    #[serde(rename = "_class_name")]
    pub class: SchedulerClass,
    #[serde(default = "train_timesteps")]
    pub num_train_timesteps: usize,
    #[serde(default)]
    beta_start: Option<f32>,
    #[serde(default)]
    beta_end: Option<f32>,
    #[serde(default)]
    pub beta_schedule: BetaSchedule,
    /// Betas given outright, which no scheduler here supports; a config that gives them is refused.
    #[serde(default)]
    trained_betas: Option<Vec<f64>>,
    #[serde(default)]
    pub prediction_type: PredictionType,
    #[serde(default)]
    clip_sample: Option<bool>,
    #[serde(default = "one")]
    pub clip_sample_range: f32,
    #[serde(default)]
    pub variance_type: VarianceType,
    #[serde(default)]
    timestep_spacing: Option<TimestepSpacing>,
    #[serde(default)]
    pub steps_offset: i64,
    #[serde(default)]
    pub thresholding: bool,
    #[serde(default)]
    pub skip_prk_steps: bool,
    #[serde(default)]
    pub set_alpha_to_one: bool,
    #[serde(default)]
    pub rescale_betas_zero_snr: bool,
    /// Whether the noise levels follow Karras' spacing, which `diffusers` lets a config leave null, as it does `use_exponential_sigmas` and `use_beta_sigmas`.
    #[serde(default)]
    use_karras_sigmas: Option<bool>,
    #[serde(default)]
    use_exponential_sigmas: Option<bool>,
    #[serde(default)]
    use_beta_sigmas: Option<bool>,
    #[serde(default)]
    pub interpolation_type: InterpolationType,
    #[serde(default)]
    pub final_sigmas_type: FinalSigmasType,
    #[serde(default)]
    pub timestep_type: TimestepType,
}

impl SchedulerConfig {
    /// Returns how timesteps are spaced: as the config says, or as its class does by default, `linspace` for the Euler and Heun schedulers and `leading` for the others.
    #[must_use]
    pub fn timestep_spacing(&self) -> TimestepSpacing {
        self.timestep_spacing.unwrap_or(match self.class {
            SchedulerClass::Euler | SchedulerClass::EulerAncestral | SchedulerClass::Heun => {
                TimestepSpacing::Linspace
            }
            SchedulerClass::Ddpm | SchedulerClass::Pndm => TimestepSpacing::Leading,
        })
    }

    /// Reads a pipeline's `scheduler/scheduler_config.json`, or a model directory's own `scheduler_config.json`.
    ///
    /// # Errors
    ///
    /// Returns an error if neither file exists, or if the one found cannot be read or parsed as a scheduler config.
    pub fn load(source: &Source) -> Result<Self> {
        source
            .optional_json("scheduler/scheduler_config.json")?
            .map_or_else(|| source.json("scheduler_config.json"), Ok)
    }

    /// Returns the first and last training betas: as the config gives them, or as its class defaults them, `0.00085` and `0.012` for Heun's scheduler and `0.0001` and `0.02` for the others.
    fn beta_range(&self) -> (f32, f32) {
        let (start, end) = if self.class == SchedulerClass::Heun {
            (0.00085, 0.012)
        } else {
            (0.0001, 0.02)
        };
        (
            self.beta_start.unwrap_or(start),
            self.beta_end.unwrap_or(end),
        )
    }

    /// Returns the range the predicted clean sample is clipped to, or `None` where the config does not clip it; by default only DDPM's scheduler clips.
    fn clip(&self) -> Option<f32> {
        self.clip_sample
            .unwrap_or(self.class == SchedulerClass::Ddpm)
            .then_some(self.clip_sample_range)
    }

    /// Returns each training timestep's cumulative product of `1 - β`, as `torch.cumprod` computes it, accumulating in `F64` and keeping `F32`.
    fn alphas_cumprod(&self) -> Result<Vec<f32>> {
        if self.trained_betas.is_some() {
            return Err(Error::unsupported("a scheduler with trained betas"));
        }

        if self.rescale_betas_zero_snr {
            return Err(Error::unsupported(
                "a scheduler rescaling its betas to a zero terminal SNR",
            ));
        }

        let (start, end) = self.beta_range();
        let betas = match self.beta_schedule {
            BetaSchedule::Linear => torch_linspace(start, end, self.num_train_timesteps),
            BetaSchedule::ScaledLinear => {
                torch_linspace(start.sqrt(), end.sqrt(), self.num_train_timesteps)
                    .into_iter()
                    .map(|beta| beta * beta)
                    .collect()
            }
            // `betas_for_alpha_bar` with `exp(-12 t)`, in `F64`, kept as `F32`, which only Heun's scheduler offers.
            BetaSchedule::Exp if self.class == SchedulerClass::Heun => {
                let count = self.num_train_timesteps as f64;
                let alpha_bar = |t: f64| (t * -12.0).exp();
                (0..self.num_train_timesteps)
                    .map(|index| {
                        let (t1, t2) = (index as f64 / count, (index + 1) as f64 / count);
                        (1.0 - alpha_bar(t2) / alpha_bar(t1)).min(0.999) as f32
                    })
                    .collect()
            }
            other => {
                return Err(Error::unsupported(format!(
                    "beta schedule {other:?} for {:?}",
                    self.class
                )));
            }
        };
        let mut product = 1.0_f64;
        Ok(betas
            .iter()
            .map(|beta| {
                product *= f64::from(1.0 - beta);
                product as f32
            })
            .collect())
    }

    /// Returns the timesteps of `steps` inference steps, last first, as `set_timesteps` spaces them.
    ///
    /// # Errors
    ///
    /// Returns an error if `steps` is zero or more than the training timesteps.
    #[expect(clippy::cast_possible_wrap, reason = "timestep counts are small")]
    pub fn timesteps(&self, steps: usize) -> Result<Vec<i64>> {
        let train = self.num_train_timesteps;
        if steps == 0 || steps > train {
            return Err(Error::unsupported(
                "inference steps outside the training timesteps",
            ));
        }
        Ok(match self.timestep_spacing() {
            TimestepSpacing::Leading => (0..steps)
                .rev()
                .map(|step| (step * (train / steps)) as i64 + self.steps_offset)
                .collect(),
            // `numpy.arange(T, 0, -T / n)`, which counts from its first value by the difference of its first two, rounded half to even, less one.
            TimestepSpacing::Trailing => {
                let start = train as f64;
                let delta = (start - start / steps as f64) - start;
                (0..steps)
                    .map(|step| (start + step as f64 * delta).round_ties_even() as i64 - 1)
                    .collect()
            }
            TimestepSpacing::Linspace => numpy_linspace(0.0, (train - 1) as f64, steps)
                .into_iter()
                .rev()
                .map(|timestep| timestep.round_ties_even() as i64)
                .collect(),
        })
    }

    /// Returns `numpy.linspace(0, T − 1, n, dtype=float32)` for `T` training timesteps and `n` steps, last first, as the Euler and Heun schedulers space their timesteps.
    fn linspace_timesteps(&self, steps: usize) -> Vec<f32> {
        numpy_linspace(0.0, (self.num_train_timesteps - 1) as f64, steps)
            .into_iter()
            .rev()
            .map(|timestep| timestep as f32)
            .collect()
    }

    /// Returns each training timestep's noise level, `√((1 − ᾱ) / ᾱ)` in `F32`.
    fn train_sigmas(&self) -> Result<Vec<f32>> {
        Ok(self
            .alphas_cumprod()?
            .into_iter()
            .map(|alpha| ((1.0 - alpha) / alpha).sqrt())
            .collect())
    }

    /// Returns the Euler schedulers' timesteps, as `F32`, and their noise levels, each interpolated linearly between the training timesteps', with a zero after the last.
    fn euler_schedule(&self, steps: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let timesteps = if self.timestep_spacing() == TimestepSpacing::Linspace {
            if steps == 0 {
                return Err(Error::unsupported("no inference steps"));
            }
            self.linspace_timesteps(steps)
        } else {
            self.timesteps(steps)?
                .into_iter()
                .map(|timestep| timestep as f32)
                .collect()
        };
        let trained = self.train_sigmas()?;
        let mut sigmas = timesteps
            .iter()
            .map(|&timestep| numpy_interp(f64::from(timestep), &trained) as f32)
            .collect::<Vec<f32>>();
        sigmas.push(0.0);
        Ok((timesteps, sigmas))
    }
}

/// Returns `torch.linspace(start, end, count)` in `F32`: the first half counted up from `start`, and the second down from `end`.
fn torch_linspace(start: f32, end: f32, count: usize) -> Vec<f32> {
    if count == 1 {
        return vec![start];
    }

    let step = (end - start) / (count - 1) as f32;
    (0..count)
        .map(|index| {
            if index < count / 2 {
                start + step * index as f32
            } else {
                end - step * (count - index - 1) as f32
            }
        })
        .collect()
}

/// Returns `numpy.linspace(start, stop, count)` in `F64`: each index times the step, plus `start`, and `stop` itself last.
fn numpy_linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    if count <= 1 {
        return vec![start; count];
    }

    let spacing = (stop - start) / (count - 1) as f64;
    (0..count)
        .map(|index| {
            if index == count - 1 {
                stop
            } else {
                index as f64 * spacing + start
            }
        })
        .collect()
}

/// Returns `numpy.interp(at, [0, 1, …], values)` in `F64`: `values` interpolated linearly at `at`, held at their ends outside them.
fn numpy_interp(at: f64, values: &[f32]) -> f64 {
    let last = values.len() - 1;
    let at = at.clamp(0.0, last as f64);
    let below = at.floor() as usize;
    let above = (below + 1).min(last);
    let (low, high) = (f64::from(values[below]), f64::from(values[above]));
    if below == above {
        low
    } else {
        (high - low) * (at - below as f64) + low
    }
}

#[derive(Debug, Clone)]
struct AlphaProducts {
    products: Vec<f32>,
    before_first: f32,
}

impl AlphaProducts {
    fn at(&self, timestep: i64) -> f32 {
        usize::try_from(timestep).map_or(self.before_first, |timestep| self.products[timestep])
    }
}

/// Returns what the first noise is scaled by: the largest noise level, or `√(σ² + 1)` of it for leading timesteps.
fn initial_scale(sigmas: &[f32], spacing: TimestepSpacing) -> f32 {
    let largest = sigmas.iter().copied().fold(0.0_f32, f32::max);
    match spacing {
        TimestepSpacing::Linspace | TimestepSpacing::Trailing => largest,
        TimestepSpacing::Leading => (largest * largest + 1.0).sqrt(),
    }
}

/// Returns the model's input at a step of noise level `sigma`: the sample over `√(σ² + 1)`.
fn scale_input(sigma: f32, sample: &Tensor) -> Result<Tensor> {
    Ok(sample.mul_scalar(f64::from(1.0 / (sigma * sigma + 1.0).sqrt()))?)
}

/// `DDPMScheduler`: ancestral sampling, a step's prediction of the clean sample mixed with the sample, and noise added at every timestep above zero.
#[derive(Debug, Clone)]
struct DdpmScheduler {
    alphas: AlphaProducts,
    timesteps: Vec<i64>,
    clip: Option<f32>,
    /// What the step scales its noise by the square root of: the posterior's variance for [`VarianceType::FixedSmall`], or the step's beta for [`VarianceType::FixedLarge`].
    variance: VarianceType,
}

impl DdpmScheduler {
    fn create(config: &SchedulerConfig, steps: usize) -> Result<Self> {
        if config.thresholding || config.prediction_type != PredictionType::Epsilon {
            return Err(Error::unsupported(
                "a DDPM scheduler other than one predicting noise",
            ));
        }

        let variance = match config.variance_type {
            variance @ (VarianceType::FixedSmall | VarianceType::FixedLarge) => variance,
            other => return Err(Error::unsupported(format!("DDPM variance {other:?}"))),
        };
        Ok(Self {
            alphas: AlphaProducts {
                products: config.alphas_cumprod()?,
                before_first: 1.0,
            },
            timesteps: config.timesteps(steps)?,
            clip: config.clip(),
            variance,
        })
    }

    /// Returns the sample at the step after `index`, from the model's predicted noise, drawing `noise` for the variance at every timestep above zero.
    fn step(
        &self,
        index: usize,
        predicted: &Tensor,
        sample: &Tensor,
        noise: impl FnOnce() -> Result<Tensor>,
    ) -> Result<Tensor> {
        let timestep = self.timesteps[index];
        let previous = self.timesteps.get(index + 1).copied().unwrap_or(-1);
        let (alpha, alpha_previous) = (self.alphas.at(timestep), self.alphas.at(previous));
        let (beta, beta_previous) = (1.0 - alpha, 1.0 - alpha_previous);
        let current_alpha = alpha / alpha_previous;
        let current_beta = 1.0 - current_alpha;
        // PyTorch's CUDA kernels divide by a host scalar by multiplying with its reciprocal.
        let mut original = sample
            .sub(&predicted.mul_scalar(f64::from(beta.sqrt()))?)?
            .mul_scalar(f64::from(1.0 / alpha.sqrt()))?;
        if let Some(range) = self.clip {
            original = original.clamp(-range, range)?;
        }

        let original_coefficient = alpha_previous.sqrt() * current_beta / beta;
        let sample_coefficient = current_alpha.sqrt() * beta_previous / beta;
        let next = original
            .mul_scalar(f64::from(original_coefficient))?
            .add(&sample.mul_scalar(f64::from(sample_coefficient))?)?;
        if timestep <= 0 {
            return Ok(next);
        }

        let variance = match self.variance {
            VarianceType::FixedSmall => {
                ((1.0 - alpha_previous) / (1.0 - alpha) * current_beta).max(1e-20)
            }
            _ => current_beta,
        };
        Ok(next.add(&noise()?.mul_scalar(f64::from(variance.sqrt()))?)?)
    }
}

/// `EulerDiscreteScheduler` without churn: Euler steps of the probability flow between noise levels that draw no noise.
/// When `ancestral`, `EulerAncestralDiscreteScheduler`: each step an Euler step down to a lower noise level and noise drawn to add back up to the next.
#[derive(Debug, Clone)]
struct EulerScheduler {
    timesteps: Vec<f32>,
    sigmas: Vec<f32>,
    spacing: TimestepSpacing,
    ancestral: bool,
}

impl EulerScheduler {
    fn create(config: &SchedulerConfig, steps: usize, ancestral: bool) -> Result<Self> {
        if config.prediction_type != PredictionType::Epsilon {
            return Err(Error::unsupported(
                "an Euler scheduler other than one predicting noise",
            ));
        }

        // The options only `EulerDiscreteScheduler` reads.
        if !ancestral
            && [
                config.use_karras_sigmas,
                config.use_exponential_sigmas,
                config.use_beta_sigmas,
            ]
            .contains(&Some(true))
        {
            return Err(Error::unsupported(
                "an Euler scheduler with Karras, exponential, or beta noise levels",
            ));
        }

        if !ancestral
            && (config.interpolation_type != InterpolationType::Linear
                || config.final_sigmas_type != FinalSigmasType::Zero
                || config.timestep_type != TimestepType::Discrete)
        {
            return Err(Error::unsupported(
                "an Euler scheduler other than one of discrete timesteps, linearly interpolated noise levels, and a final zero",
            ));
        }

        let (timesteps, sigmas) = config.euler_schedule(steps)?;
        Ok(Self {
            timesteps,
            sigmas,
            spacing: config.timestep_spacing(),
            ancestral,
        })
    }

    /// Returns the sample after step `index`, from the model's predicted noise, drawing `noise` where the step is ancestral.
    fn step(
        &self,
        index: usize,
        predicted: &Tensor,
        sample: &Tensor,
        noise: impl FnOnce() -> Result<Tensor>,
    ) -> Result<Tensor> {
        let (sigma, next) = (self.sigmas[index], self.sigmas[index + 1]);
        let original = sample.sub(&predicted.mul_scalar(f64::from(sigma))?)?;
        let derivative = sample.sub(&original)?.mul_scalar(f64::from(1.0 / sigma))?;
        if !self.ancestral {
            return Ok(sample.add(&derivative.mul_scalar(f64::from(next - sigma))?)?);
        }

        let up = (next * next * (sigma * sigma - next * next) / (sigma * sigma)).sqrt();
        let down = (next * next - up * up).sqrt();
        Ok(sample
            .add(&derivative.mul_scalar(f64::from(down - sigma))?)?
            .add(&noise()?.mul_scalar(f64::from(up))?)?)
    }
}

/// `PNDMScheduler` with its Runge–Kutta steps skipped, as Stable Diffusion 1.5's is: linear multistep updates of the predicted noise over the steps before, starting with one step taken twice.
#[derive(Debug, Clone)]
struct PndmScheduler {
    alphas: AlphaProducts,
    /// The timesteps as `set_timesteps` lists them for PLMS steps, last first, with the second-largest repeated.
    timesteps: Vec<i64>,
    /// The training timesteps between two inference timesteps.
    stride: i64,
}

/// The multistep state of one generation: the last noise predictions, up to four, the sample the first step started from, and the steps taken.
#[derive(Debug, Default)]
struct PndmState {
    predictions: Vec<Tensor>,
    first_sample: Option<Tensor>,
    counter: usize,
}

impl PndmScheduler {
    #[expect(clippy::cast_possible_wrap, reason = "timestep counts are small")]
    fn create(config: &SchedulerConfig, steps: usize) -> Result<Self> {
        if !config.skip_prk_steps || config.prediction_type != PredictionType::Epsilon {
            return Err(Error::unsupported(
                "a PNDM scheduler other than one skipping its Runge-Kutta steps and predicting noise",
            ));
        }

        let products = config.alphas_cumprod()?;
        // `set_timesteps`' timesteps, first to last, before PNDM reverses them; listing them refuses zero steps.
        let mut ascending = config.timesteps(steps)?;
        ascending.reverse();
        let count = ascending.len();
        let mut timesteps = ascending[..count - 1].to_vec();
        timesteps.extend(&ascending[count.saturating_sub(2)..count - 1]);
        timesteps.push(ascending[count - 1]);
        timesteps.reverse();
        Ok(Self {
            alphas: AlphaProducts {
                before_first: if config.set_alpha_to_one {
                    1.0
                } else {
                    products[0]
                },
                products,
            },
            timesteps,
            stride: (config.num_train_timesteps / steps) as i64,
        })
    }

    /// Returns the sample after the step at `timestep`, from the model's predicted noise, as `step_plms` takes it.
    fn step(
        &self,
        state: &mut PndmState,
        timestep: i64,
        predicted: &Tensor,
        sample: &Tensor,
    ) -> Result<Tensor> {
        let (mut timestep, mut previous) = (timestep, timestep - self.stride);
        if state.counter == 1 {
            (previous, timestep) = (timestep, timestep + self.stride);
        } else {
            let keep = state.predictions.len().saturating_sub(3);
            state.predictions.drain(..keep);
            state.predictions.push(predicted.clone());
        }

        let history = &state.predictions;
        let last = |back: usize| &history[history.len() - back];
        let mut sample = sample.clone();
        let combined = match (history.len(), state.counter) {
            (1, 0) => {
                state.first_sample = Some(sample.clone());
                predicted.clone()
            }
            (1, 1) => {
                sample = state
                    .first_sample
                    .take()
                    .ok_or_else(|| Error::unsupported("a PNDM step without its first sample"))?;
                predicted.add(last(1))?.mul_scalar(0.5)?
            }
            (2, _) => last(1).mul_scalar(3.0)?.sub(last(2))?.mul_scalar(0.5)?,
            (3, _) => last(1)
                .mul_scalar(23.0)?
                .sub(&last(2).mul_scalar(16.0)?)?
                .add(&last(3).mul_scalar(5.0)?)?
                .mul_scalar(f64::from(1.0_f32 / 12.0))?,
            _ => last(1)
                .mul_scalar(55.0)?
                .sub(&last(2).mul_scalar(59.0)?)?
                .add(&last(3).mul_scalar(37.0)?)?
                .sub(&last(4).mul_scalar(9.0)?)?
                .mul_scalar(f64::from(1.0_f32 / 24.0))?,
        };
        state.counter += 1;
        let (alpha, alpha_previous) = (self.alphas.at(timestep), self.alphas.at(previous));
        let (beta, beta_previous) = (1.0 - alpha, 1.0 - alpha_previous);
        let sample_coefficient = (alpha_previous / alpha).sqrt();
        let denominator = alpha * beta_previous.sqrt() + (alpha * beta * alpha_previous).sqrt();
        Ok(sample.mul_scalar(f64::from(sample_coefficient))?.sub(
            &combined
                .mul_scalar(f64::from(alpha_previous - alpha))?
                .mul_scalar(f64::from(1.0 / denominator))?,
        )?)
    }
}

/// `HeunDiscreteScheduler` with linearly spaced timesteps, with or without Karras noise levels: each step an Euler step to the next noise level, then a second evaluation there averaging the two derivatives, the last step Euler's alone.
#[derive(Debug, Clone)]
struct HeunScheduler {
    /// The noise levels, the ones after the first twice, and a zero.
    sigmas: Vec<f32>,
    /// The timesteps the model reads, the ones after the first twice.
    timesteps: Vec<f32>,
    clip: Option<f32>,
    /// The first evaluation's derivative, step, and sample, while a step waits for its second.
    pending: Option<(Tensor, f32, Tensor)>,
}

impl HeunScheduler {
    fn create(config: &SchedulerConfig, steps: usize) -> Result<Self> {
        if config.prediction_type != PredictionType::Sample {
            return Err(Error::unsupported(
                "a Heun scheduler predicting other than the clean sample",
            ));
        }

        if config.timestep_spacing() != TimestepSpacing::Linspace || steps < 2 {
            return Err(Error::unsupported(
                "a Heun scheduler other than one of linearly spaced timesteps",
            ));
        }

        if [config.use_exponential_sigmas, config.use_beta_sigmas].contains(&Some(true)) {
            return Err(Error::unsupported(
                "a Heun scheduler with exponential or beta noise levels",
            ));
        }

        let train = config.num_train_timesteps;
        let trained = config.train_sigmas()?;
        let log_trained = trained.iter().map(|sigma| sigma.ln()).collect::<Vec<f32>>();
        // The timesteps, last first, and the noise levels interpolated there.
        let timesteps = config.linspace_timesteps(steps);
        let interpolated = timesteps
            .iter()
            .map(|&timestep| numpy_interp(f64::from(timestep), &trained))
            .collect::<Vec<f64>>();
        let (sigmas, timesteps) = if config.use_karras_sigmas == Some(true) {
            let (low, high) = (interpolated[steps - 1], interpolated[0]);
            let (low_root, high_root) = (low.powf(1.0 / 7.0), high.powf(1.0 / 7.0));
            let sigmas = numpy_linspace(0.0, 1.0, steps)
                .into_iter()
                .map(|ramp| (high_root + ramp * (low_root - high_root)).powi(7))
                .collect::<Vec<f64>>();
            // `_sigma_to_t`: each noise level's place between the training noise levels whose logs bracket its log.
            let timesteps = sigmas
                .iter()
                .map(|&sigma| {
                    let log = sigma.max(1e-10).ln();
                    let low_index = log_trained
                        .iter()
                        .rposition(|&trained| log - f64::from(trained) >= 0.0)
                        .unwrap_or(0)
                        .min(train - 2);
                    let (low, high) = (
                        f64::from(log_trained[low_index]),
                        f64::from(log_trained[low_index + 1]),
                    );
                    let weight = ((low - log) / (low - high)).clamp(0.0, 1.0);
                    ((1.0 - weight) * low_index as f64 + weight * (low_index + 1) as f64) as f32
                })
                .collect::<Vec<f32>>();
            (sigmas, timesteps)
        } else {
            (interpolated, timesteps)
        };
        let mut doubled_sigmas = vec![sigmas[0] as f32];
        for &sigma in &sigmas[1..] {
            doubled_sigmas.extend([sigma as f32, sigma as f32]);
        }

        doubled_sigmas.push(0.0);
        let mut doubled_timesteps = vec![timesteps[0]];
        for &timestep in &timesteps[1..] {
            doubled_timesteps.extend([timestep, timestep]);
        }
        Ok(Self {
            sigmas: doubled_sigmas,
            timesteps: doubled_timesteps,
            clip: config.clip(),
            pending: None,
        })
    }

    /// Returns the sample after evaluation `index`, from the model's predicted clean sample.
    fn step(&mut self, index: usize, predicted: &Tensor, sample: &Tensor) -> Result<Tensor> {
        let original = match self.clip {
            Some(range) => predicted.clamp(-range, range)?,
            None => predicted.clone(),
        };
        match self.pending.take() {
            None => {
                let (sigma, next) = (self.sigmas[index], self.sigmas[index + 1]);
                let derivative = sample.sub(&original)?.mul_scalar(f64::from(1.0 / sigma))?;
                let step = next - sigma;
                let output = sample.add(&derivative.mul_scalar(f64::from(step))?)?;
                self.pending = Some((derivative, step, sample.clone()));
                Ok(output)
            }
            Some((first, step, start)) => {
                let next = self.sigmas[index];
                let derivative = sample
                    .sub(&original)?
                    .mul_scalar(f64::from(1.0 / next))?
                    .add(&first)?
                    .mul_scalar(0.5)?;
                Ok(start.add(&derivative.mul_scalar(f64::from(step))?)?)
            }
        }
    }
}

#[derive(Debug)]
enum Kind {
    Ddpm(DdpmScheduler),
    Pndm(PndmScheduler, PndmState),
    Euler(EulerScheduler),
    Heun(HeunScheduler),
}

/// The scheduler a latent diffusion pipeline's config names, behind one interface: its timesteps, its first noise's scale, its input scaling, its steps, which draw noise where the scheduler does, and noise added to a clean sample for pipelines that start from one.
#[derive(Debug)]
pub struct Sampler {
    kind: Kind,
    /// Every step's timestep, last first, as the model reads it.
    timesteps: Vec<f32>,
    /// The step the schedule starts at, as a pipeline's `strength` skips the first.
    begin: usize,
}

impl Sampler {
    pub fn create(config: &SchedulerConfig, steps: usize) -> Result<Self> {
        let kind = match config.class {
            SchedulerClass::Ddpm => Kind::Ddpm(DdpmScheduler::create(config, steps)?),
            SchedulerClass::Pndm => {
                Kind::Pndm(PndmScheduler::create(config, steps)?, PndmState::default())
            }
            SchedulerClass::Euler => Kind::Euler(EulerScheduler::create(config, steps, false)?),
            SchedulerClass::EulerAncestral => {
                Kind::Euler(EulerScheduler::create(config, steps, true)?)
            }
            SchedulerClass::Heun => Kind::Heun(HeunScheduler::create(config, steps)?),
        };
        let discrete = |timesteps: &[i64]| -> Vec<f32> {
            timesteps.iter().map(|&timestep| timestep as f32).collect()
        };
        let timesteps = match &kind {
            Kind::Ddpm(scheduler) => discrete(&scheduler.timesteps),
            Kind::Pndm(scheduler, _) => discrete(&scheduler.timesteps),
            Kind::Euler(scheduler) => scheduler.timesteps.clone(),
            Kind::Heun(scheduler) => scheduler.timesteps.clone(),
        };
        Ok(Self {
            kind,
            timesteps,
            begin: 0,
        })
    }

    /// Starts the schedule `count` steps in, at most at its end, as `get_timesteps` does for a `strength` below one.
    pub fn skip(&mut self, count: usize) {
        self.begin = count.min(self.timesteps.len());
    }

    /// Returns the timesteps the model reads from the first kept step, last first.
    #[must_use]
    pub fn timesteps(&self) -> &[f32] {
        &self.timesteps[self.begin..]
    }

    #[must_use]
    pub fn initial_scale(&self) -> f32 {
        match &self.kind {
            Kind::Ddpm(_) | Kind::Pndm(..) => 1.0,
            Kind::Euler(scheduler) => initial_scale(&scheduler.sigmas, scheduler.spacing),
            Kind::Heun(scheduler) => initial_scale(&scheduler.sigmas, TimestepSpacing::Linspace),
        }
    }

    pub fn scale_input(&self, index: usize, sample: &Tensor) -> Result<Tensor> {
        let index = self.begin + index;
        match &self.kind {
            Kind::Ddpm(_) | Kind::Pndm(..) => Ok(sample.clone()),
            Kind::Euler(scheduler) => scale_input(scheduler.sigmas[index], sample),
            Kind::Heun(scheduler) => scale_input(scheduler.sigmas[index], sample),
        }
    }

    /// Returns a clean sample noised to the first kept step: `√ᾱ x + √(1 − ᾱ) ε` for DDPM's and PNDM's discrete timesteps, and `x + σ ε` for Euler's noise levels.
    ///
    /// # Errors
    ///
    /// Returns an error under Heun's scheduler, which this does not noise for.
    pub fn add_noise(&self, sample: &Tensor, noise: &Tensor) -> Result<Tensor> {
        let alpha = match &self.kind {
            Kind::Ddpm(scheduler) => scheduler.alphas.at(scheduler.timesteps[self.begin]),
            Kind::Pndm(scheduler, _) => scheduler.alphas.at(scheduler.timesteps[self.begin]),
            Kind::Euler(scheduler) => {
                return Ok(sample.add(&noise.mul_scalar(f64::from(scheduler.sigmas[self.begin]))?)?);
            }
            Kind::Heun(_) => {
                return Err(Error::unsupported(
                    "noising a clean sample under Heun's scheduler",
                ));
            }
        };
        Ok(sample
            .mul_scalar(f64::from(alpha.sqrt()))?
            .add(&noise.mul_scalar(f64::from((1.0 - alpha).sqrt()))?)?)
    }

    /// Returns the sample after step `index` of the kept steps, drawing `noise` where the scheduler adds noise.
    pub fn step(
        &mut self,
        index: usize,
        predicted: &Tensor,
        sample: &Tensor,
        noise: impl FnOnce() -> Result<Tensor>,
    ) -> Result<Tensor> {
        let index = self.begin + index;
        match &mut self.kind {
            Kind::Ddpm(scheduler) => scheduler.step(index, predicted, sample, noise),
            Kind::Pndm(scheduler, state) => {
                let timestep = scheduler.timesteps[index];
                scheduler.step(state, timestep, predicted, sample)
            }
            Kind::Euler(scheduler) => scheduler.step(index, predicted, sample, noise),
            Kind::Heun(scheduler) => scheduler.step(index, predicted, sample),
        }
    }
}
