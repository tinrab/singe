use crate::{
    data_type::DataType,
    error::{Error, Result},
    generation::GenerationOptions,
    graph::capture,
    shape::Shape,
    tensor::{DrawSource, SamplingOptions, Tensor},
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PenaltySet {
    pub repetition: f32,
    pub frequency: f32,
    pub presence: f32,
}

impl Default for PenaltySet {
    fn default() -> Self {
        Self {
            repetition: 1.0,
            frequency: 0.0,
            presence: 0.0,
        }
    }
}

impl PenaltySet {
    #[must_use]
    pub fn is_neutral(&self) -> bool {
        *self == Self::default()
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if !(self.repetition.is_finite() && self.repetition > 0.0) {
            return Err(Error::out_of_range("repetition penalty"));
        }

        if !(self.frequency.is_finite() && self.presence.is_finite()) {
            return Err(Error::invalid_argument(
                "frequency and presence penalties must be finite",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Bias<'a> {
    pub tokens: &'a Tensor,
    pub values: &'a Tensor,
}

#[derive(Debug, Clone, Copy)]
pub struct PenaltyCounts<'a> {
    pub settings: &'a Tensor,
    pub generated: &'a Tensor,
    pub prompt: &'a Tensor,
}

#[derive(Debug, Clone, Copy)]
pub struct Choice<'a> {
    pub first_tokens: &'a Tensor,
    pub suppressed_tokens: &'a [i64],
    pub first_suppressed_tokens: &'a [i64],
    pub sampling: &'a [Option<SamplingOptions>],
    pub draws: DrawSource<'a>,
    pub bias: Option<Bias<'a>>,
    pub penalties: Option<PenaltyCounts<'a>>,
    pub logprobs: Option<i64>,
}

impl<'a> Choice<'a> {
    #[must_use]
    pub fn new(
        first_tokens: &'a Tensor,
        options: &'a GenerationOptions,
        sampling: &'a [Option<SamplingOptions>],
        draws: DrawSource<'a>,
    ) -> Self {
        Self {
            first_tokens,
            suppressed_tokens: &options.suppressed_tokens,
            first_suppressed_tokens: &options.first_suppressed_tokens,
            sampling,
            draws,
            bias: None,
            penalties: None,
            logprobs: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Chosen {
    pub tokens: Tensor,
    pub logprobs: Option<ChosenLogprobs>,
}

#[derive(Debug, Clone)]
pub struct ChosenLogprobs {
    pub tokens: Tensor,
    pub values: Tensor,
}

// Chooses a token for each row of `logits`, `[S, V]`, in vLLM's order: the bias, the penalties, the suppressed tokens, then each row's most likely token or a draw as `choice.sampling` asks.
// A captured step chooses through this too, so a program chooses as a running batch does.
pub fn choose(logits: &Tensor, choice: &Choice<'_>) -> Result<Chosen> {
    let rows = logits.dimension(0)?;
    let mut adjusted = logits.clone();
    if choice.bias.is_some() || choice.penalties.is_some() {
        adjusted = float32(&adjusted)?;
    }

    if let Some(bias) = choice.bias {
        adjusted = adjusted.scatter_add(-1, bias.tokens, bias.values)?;
    }

    if let Some(penalties) = choice.penalties {
        adjusted = penalized(&adjusted, &penalties)?;
    }

    let adjusted = suppressed(&adjusted, choice)?;
    let tokens = adjusted
        .sample(choice.sampling, choice.draws)?
        .reshape_with(&[rows])?;
    let logprobs = choice
        .logprobs
        .map(|alternatives| log_probabilities(logits, &tokens, alternatives))
        .transpose()?;
    Ok(Chosen { tokens, logprobs })
}

fn float32(tensor: &Tensor) -> Result<Tensor> {
    if tensor.data_type() == DataType::F32 {
        Ok(tensor.clone())
    } else {
        tensor.cast(DataType::F32)
    }
}

fn penalized(logits: &Tensor, penalties: &PenaltyCounts<'_>) -> Result<Tensor> {
    let setting = |index| penalties.settings.narrow(1, index, 1);
    let (repetition, frequency, presence) = (setting(0)?, setting(1)?, setting(2)?);
    let zero = literal(logits, 0.0)?;
    let appeared = penalties.generated.greater(&zero)?.cast(DataType::F32)?;
    let seen = penalties
        .generated
        .add(penalties.prompt)?
        .greater(&zero)?
        .cast(DataType::F32)?;
    let scale = seen.mul(&repetition.sub_scalar(1.0)?)?.add_scalar(1.0)?;
    let scaled = Tensor::where_select(
        &logits.greater(&zero)?,
        &logits.div(&scale)?,
        &logits.mul(&scale)?,
    )?;
    scaled
        .sub(&penalties.generated.mul(&frequency)?)?
        .sub(&appeared.mul(&presence)?)
}

fn suppressed(logits: &Tensor, choice: &Choice<'_>) -> Result<Tensor> {
    if choice.suppressed_tokens.is_empty() && choice.first_suppressed_tokens.is_empty() {
        return Ok(logits.clone());
    }

    let &[_, vocabulary, ..] = logits.shape().dimensions() else {
        return Err(Error::invalid_argument("generation logits"));
    };
    let width =
        usize::try_from(vocabulary).map_err(|_| Error::out_of_range("generation vocabulary"))?;
    // What each suppression adds to the logits, `[2, V]`: the first row for later tokens, the second for a sequence's first.
    let mut table = vec![0.0_f32; 2 * width];
    let mut mask = |token: i64, rows: &[usize]| {
        let index = usize::try_from(token)
            .ok()
            .filter(|&index| index < width)
            .ok_or_else(|| {
                Error::invalid_argument(format!(
                    "suppressed token {token} is outside a vocabulary of {width}"
                ))
            })?;
        for row in rows {
            table[row * width + index] = f32::NEG_INFINITY;
        }
        Ok::<_, Error>(())
    };
    for &token in choice.suppressed_tokens {
        mask(token, &[0, 1])?;
    }

    for &token in choice.first_suppressed_tokens {
        mask(token, &[1])?;
    }

    let table = capture::literal(logits.device(), |device| {
        Tensor::from_vec(device, Shape::create([2, vocabulary])?, table)
    })?
    .cast(logits.data_type())?;
    logits.add(&table.index_select(0, choice.first_tokens)?)
}

fn log_probabilities(
    logits: &Tensor,
    tokens: &Tensor,
    alternatives: i64,
) -> Result<ChosenLogprobs> {
    let log_probabilities = float32(logits)?.log_softmax(-1)?;
    let chosen = tokens.unsqueeze(1)?;
    let chosen_values = log_probabilities.gather(-1, &chosen)?;
    let vocabulary = logits.shape().dimensions().last().copied().unwrap_or(0);
    let alternatives = alternatives.min(vocabulary);
    if alternatives == 0 {
        return Ok(ChosenLogprobs {
            tokens: chosen,
            values: chosen_values,
        });
    }

    let (top_values, top_tokens) = log_probabilities.topk(alternatives, -1, true, true)?;
    Ok(ChosenLogprobs {
        tokens: Tensor::concat(&[chosen, top_tokens], 1)?,
        values: Tensor::concat(&[chosen_values, top_values], 1)?,
    })
}

fn literal(like: &Tensor, value: f32) -> Result<Tensor> {
    capture::literal(like.device(), |device| Tensor::scalar(device, value))
}
