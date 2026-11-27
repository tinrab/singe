use crate::{
    data_type::DataType,
    error::{Error, Result},
    graph::{DimensionExpr, capture},
    random::Generator,
    shape::Shape,
    tensor::Tensor,
};

/// Temperatures at or below this value select the most likely tokens instead of sampling.
const GREEDY_TEMPERATURE: f32 = 1.0e-5;

/// Settings for drawing a token index per row of logits with [`Tensor::sample`].
///
/// The defaults draw at temperature one, without top-k or top-p filtering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingOptions {
    temperature: f32,
    top_k: Option<i64>,
    top_p: f32,
    min_p: f32,
}

impl Default for SamplingOptions {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: 1.0,
            min_p: 0.0,
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

    #[must_use]
    pub const fn with_min_p(mut self, min_p: f32) -> Self {
        self.min_p = min_p;
        self
    }

    fn validate(&self) -> Result<()> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(Error::invalid_argument(
                "sampling temperature must be finite and nonnegative",
            ));
        }

        if !(self.top_p > 0.0 && self.top_p <= 1.0) {
            return Err(Error::out_of_range("sampling top-p"));
        }

        if !(0.0..=1.0).contains(&self.min_p) {
            return Err(Error::out_of_range("sampling min-p"));
        }

        if self.top_k.is_some_and(|top_k| top_k < 1) {
            return Err(Error::out_of_range("sampling top-k"));
        }
        Ok(())
    }

    fn settings(options: Option<&Self>, vocabulary: i64) -> [f32; 4] {
        let Some(options) = options else {
            return [0.0, vocabulary as f32, f32::INFINITY, 0.0];
        };
        let kept = options
            .top_k
            .map_or(vocabulary, |top_k| top_k.min(vocabulary));
        let top_p = if options.top_p < 1.0 {
            options.top_p
        } else {
            f32::INFINITY
        };
        [options.temperature, kept as f32, top_p, options.min_p]
    }
}

#[derive(Debug, Clone, Copy)]
pub enum DrawSource<'a> {
    Generator(Option<&'a Generator>),
    Streams(&'a Tensor),
}

impl Tensor {
    /// Draws a token index per row of `[rows, vocabulary]` logits, returning `[rows]` of `I64`.
    ///
    /// `options` holds a row's settings each, or one for every row; a row without settings takes its most likely token.
    /// Logits are divided by the temperature, filtered by min-p, top-k, and then top-p, and sampled from their softmax.
    /// The filters are thresholds on one descending sort of each row, so filtered tokens are never drawn, and tokens tied with the last one a filter keeps are kept too.
    /// `draws` names where the random values come from: one generator, or the device's default, or a stream per row, `[rows, 2]` of `I64` `[seed, offset]`, which makes a row's draw independent of the others.
    ///
    /// # Errors
    ///
    /// Returns an error if the logits are not a floating matrix, `options` has neither one entry nor one per row, an option is out of range, or the backend fails.
    pub fn sample(
        &self,
        options: &[Option<SamplingOptions>],
        draws: DrawSource<'_>,
    ) -> Result<Self> {
        let (rows, vocabulary) = match *self.shape().dimensions() {
            [rows, vocabulary] if self.data_type().is_float() => (rows, vocabulary),
            _ => {
                return Err(Error::invalid_argument(
                    "sampling needs a floating matrix of logits",
                ));
            }
        };
        if options.len() != 1 && i64::try_from(options.len()).ok() != Some(rows) {
            return Err(Error::length_mismatch(
                "sampling options per row",
                usize::try_from(rows).unwrap_or(usize::MAX),
                options.len(),
            ));
        }

        for options in options.iter().flatten() {
            options.validate()?;
        }

        let logits = if self.data_type() == DataType::F32 {
            self.clone()
        } else {
            self.cast(DataType::F32)?
        };
        let greedy = logits.argmax(-1)?;
        let row_dimension = self.dimension(0)?;
        if options.iter().all(Option::is_none) {
            return Ok(greedy);
        }

        let settings = options
            .iter()
            .flat_map(|options| SamplingOptions::settings(options.as_ref(), vocabulary))
            .collect::<Vec<_>>();
        let setting_rows = if options.len() == 1 { 1 } else { rows };
        let settings = capture::literal(logits.device(), |device| {
            Self::from_vec(device, Shape::create([setting_rows, 4])?, settings)
        })?;
        let setting = |index| settings.narrow(1, index, 1);
        let (temperature, kept, top_p, min_p) =
            (setting(0)?, setting(1)?, setting(2)?, setting(3)?);

        let tempered = logits.div(&temperature.clamp(GREEDY_TEMPERATURE, f32::MAX)?)?;
        let (sorted, _) = tempered.sort(-1, true)?;
        // A token's probability over the likeliest's is the exponential of their logits' difference.
        let by_min_p = sorted.narrow(-1, 0, 1)?.add(&min_p.log()?)?;
        // One setting for every row gathers each row's own position, which `gather` would not broadcast to.
        let positions = kept
            .sub_scalar(1.0)?
            .cast(DataType::I64)?
            .expand_with(&[row_dimension.clone(), DimensionExpr::constant(1)])?;
        let by_top_k = sorted.gather(-1, &positions)?;
        let threshold = by_min_p.maximum(&by_top_k)?;
        let filtered = masked_below(&sorted, &threshold)?;
        let probabilities = filtered.softmax(-1)?;
        let preceding = probabilities.cumsum(-1)?.sub(&probabilities)?;
        let removed = preceding.greater_equal(&top_p)?;
        let by_top_p = Self::where_select(&removed, &literal(&logits, f32::INFINITY)?, &filtered)?
            .min_reduce(&[-1], true)?;
        let weights = masked_below(&tempered, &threshold.maximum(&by_top_p)?)?.softmax(-1)?;

        let drawn = match draws {
            DrawSource::Generator(generator) => weights.multinomial(1, generator)?,
            DrawSource::Streams(states) => weights.multinomial_rows(1, states)?,
        }
        .reshape_with(&[row_dimension])?;
        let is_greedy = temperature
            .less_equal(&literal(&logits, GREEDY_TEMPERATURE)?)?
            .squeeze(1)?;
        Self::where_select(&is_greedy, &greedy, &drawn)
    }
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
