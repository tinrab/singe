use crate::{
    data_type::DataType,
    error::{Error, Result},
    graph::capture,
    random::Generator,
    tensor::Tensor,
};

/// Temperatures at or below this value select the most likely tokens instead of sampling.
const GREEDY_TEMPERATURE: f32 = 1.0e-5;

/// Settings for drawing token indices from logits with [`Tensor::sample`].
///
/// The defaults draw one sample with replacement at temperature one, without top-k or top-p filtering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingOptions {
    temperature: f32,
    top_k: Option<i64>,
    top_p: f32,
    samples: i64,
    replacement: bool,
}

impl Default for SamplingOptions {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: 1.0,
            samples: 1,
            replacement: true,
        }
    }
}

impl SamplingOptions {
    /// Returns the default options.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Divides logits by `temperature` before filtering; temperatures near zero select the most likely tokens.
    #[must_use]
    pub const fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Keeps only the `top_k` largest logits of each row, or every logit for `None`.
    #[must_use]
    pub const fn with_top_k(mut self, top_k: Option<i64>) -> Self {
        self.top_k = top_k;
        self
    }

    /// Keeps the smallest set of most likely tokens whose probabilities reach `top_p`, always including the most likely one.
    #[must_use]
    pub const fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    /// Draws `samples` indices per row.
    #[must_use]
    pub const fn with_samples(mut self, samples: i64) -> Self {
        self.samples = samples;
        self
    }

    /// Draws with replacement, so an index may repeat, or without, so the indices of a row differ.
    #[must_use]
    pub const fn with_replacement(mut self, replacement: bool) -> Self {
        self.replacement = replacement;
        self
    }

    fn validate(&self, vocabulary: i64) -> Result<()> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(Error::invalid_argument(
                "sampling temperature must be finite and nonnegative",
            ));
        }

        if !(self.top_p > 0.0 && self.top_p <= 1.0) {
            return Err(Error::out_of_range("sampling top-p"));
        }

        if self.top_k.is_some_and(|top_k| top_k < 1) {
            return Err(Error::out_of_range("sampling top-k"));
        }

        if self.samples < 1 {
            return Err(Error::invalid_argument(
                "sampling needs at least one sample",
            ));
        }

        if !self.replacement && self.samples > vocabulary {
            return Err(Error::invalid_argument(
                "sampling without replacement needs at most one sample per category",
            ));
        }
        Ok(())
    }
}

impl Tensor {
    /// Draws token indices from logits of shape `[vocabulary]` or `[batch, vocabulary]`, returning `I64` indices of shape `[samples]` or `[batch, samples]`.
    ///
    /// Logits are divided by the temperature, filtered by top-k and then top-p, and sampled from their softmax, with replacement through `multinomial` or without replacement through the Gumbel top-k trick.
    /// Filtered tokens are never drawn; tokens tied with the last one top-p keeps are kept too.
    /// The work runs on the logits' device from `generator`, or the device's default generator, so graph capture records it like other random operations.
    ///
    /// # Errors
    ///
    /// Returns an error if the logits are not a floating vector or matrix, an option is out of range, or the backend fails.
    pub fn sample(&self, options: &SamplingOptions, generator: Option<&Generator>) -> Result<Self> {
        let rank = self.shape().rank();
        if !matches!(rank, 1 | 2) || !self.data_type().is_float() {
            return Err(Error::invalid_argument(
                "sampling needs a floating vector or matrix of logits",
            ));
        }

        let vocabulary = self.shape().dimensions()[usize::try_from(rank - 1).unwrap_or(0)];
        options.validate(vocabulary)?;
        let logits = if self.data_type() == DataType::F32 {
            self.clone()
        } else {
            self.cast(DataType::F32)?
        };
        if options.temperature <= GREEDY_TEMPERATURE {
            return logits
                .topk(options.samples.min(vocabulary), -1, true, true)
                .map(|(_, indices)| indices);
        }

        let mut logits = logits.mul_scalar(1.0 / options.temperature)?;
        if let Some(top_k) = options.top_k.filter(|&top_k| top_k < vocabulary) {
            let (values, _) = logits.topk(top_k, -1, true, true)?;
            let threshold = values.narrow(-1, top_k - 1, 1)?;
            logits = masked_below(&logits, &threshold)?;
        }

        if options.top_p < 1.0 {
            logits = nucleus(&logits, options.top_p)?;
        }
        if options.replacement {
            logits.softmax(-1)?.multinomial(options.samples, generator)
        } else {
            let uniform = Self::rand(self.device(), logits.shape(), DataType::F32, generator)?;
            let gumbel = uniform.log()?.neg()?.log()?.neg()?;
            logits
                .add(&gumbel)?
                .topk(options.samples, -1, true, true)
                .map(|(_, indices)| indices)
        }
    }
}

/// Keeps the most likely logits whose probabilities before them sum to less than `top_p`, so the most likely one always stays.
fn nucleus(logits: &Tensor, top_p: f32) -> Result<Tensor> {
    let (sorted, _) = logits.sort(-1, true)?;
    let probabilities = sorted.softmax(-1)?;
    let preceding = probabilities.cumsum(-1)?.sub(&probabilities)?;
    let removed = preceding.greater_equal(&literal(logits, top_p)?)?;
    let kept = Tensor::where_select(&removed, &literal(logits, f32::INFINITY)?, &sorted)?;
    let threshold = kept.min_reduce(&[-1], true)?;
    masked_below(logits, &threshold)
}

/// Replaces logits below `threshold`, which broadcasts along the last dimension, with negative infinity.
fn masked_below(logits: &Tensor, threshold: &Tensor) -> Result<Tensor> {
    let below = logits.less(threshold)?;
    Tensor::where_select(&below, &literal(logits, f32::NEG_INFINITY)?, logits)
}

/// Creates an `F32` scalar on the device of `like`, recorded as a literal during graph capture.
fn literal(like: &Tensor, value: f32) -> Result<Tensor> {
    capture::literal(like.device(), |device| Tensor::scalar(device, value))
}
