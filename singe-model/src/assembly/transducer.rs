//! A transducer's prediction and joint networks, and greedy decoding over an encoder's rows, as `NeMo`'s `RNNTDecoder`, `RNNTJoint`, and greedy RNN-T and TDT decoders write them.
//!
//! Its state is `prediction.embedding`, `prediction.recurrent.N` (as [`Schema::recurrent`] names it), `prediction.projection` where the model has one, `joint.encoder`, `joint.prediction` where the model has one, and `joint.head`.

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Error as SingeError, Result as SingeResult, Shape, Tensor,
    nn::{
        Activation, Embedding, Layer, Linear, ModelState, Recurrent, RecurrentConfig,
        RecurrentState,
    },
};

use crate::{
    assembly::recurrent_rules,
    format::{Rules, join},
    schema::Schema,
};

/// How greedy decoding moves along the encoder's rows.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransducerDecoding {
    /// RNN-T: a blank advances one row, and a row emits at most `max_symbols` tokens before decoding moves on.
    Rnnt { max_symbols: i64 },
    /// TDT: the joint also scores how many rows each emission spans, one of `durations`, and a blank spans at least one; with `max_symbols`, a row emits at most that many tokens before decoding moves on.
    Tdt {
        durations: Vec<i64>,
        max_symbols: Option<i64>,
    },
}

/// A transducer's sizes and tokens.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransducerConfig {
    /// The width of the encoder's rows.
    pub encoder_width: i64,
    /// The tokens the joint scores, the blank among them.
    pub vocabulary: i64,
    pub blank: i64,
    /// A token that ends a transcription, if the model has one.
    pub end: Option<i64>,
    pub prediction: RecurrentConfig,
    /// Whether the prediction network projects its recurrent output, as `transformers`' `decoder_projector` does.
    pub prediction_projection: bool,
    pub joint_width: i64,
    pub joint_activation: Activation,
    /// Whether the joint projects the prediction network's output before adding it, as `NeMo`'s `pred` linear does.
    pub joint_prediction: bool,
    pub decoding: TransducerDecoding,
    /// The steps decoding takes per encoder row, at most, beside any per-row cap: `transformers`' `max_symbols_per_step`, which its caller turns into the steps of a clip.
    pub steps_per_row: i64,
}

impl TransducerConfig {
    /// Returns the number of logits the joint gives: one per token, then one per duration for TDT.
    #[must_use]
    pub fn logits(&self) -> i64 {
        self.vocabulary
            + match &self.decoding {
                TransducerDecoding::Rnnt { .. } => 0,
                TransducerDecoding::Tdt { durations, .. } => {
                    i64::try_from(durations.len()).unwrap_or(i64::MAX)
                }
            }
    }
}

/// The prediction network: each emitted token's embedding through a recurrent stack, then a projection where the model has one.
#[derive(Debug, Clone)]
pub struct PredictionNetwork {
    embedding: Embedding,
    recurrent: Recurrent,
    projection: Option<Linear>,
}

impl PredictionNetwork {
    pub fn create(schema: &mut Schema<'_>, config: &TransducerConfig) -> SingeResult<Self> {
        let width = config.prediction.input;
        let output = config.prediction.output();
        Ok(Self {
            embedding: Embedding::new(
                schema.parameter("embedding.weight", [config.vocabulary, width])?,
            ),
            recurrent: schema.recurrent("recurrent", &config.prediction)?,
            projection: config
                .prediction_projection
                .then(|| schema.linear("projection", output, output, true))
                .transpose()?,
        })
    }

    #[must_use]
    pub const fn recurrent(&self) -> &Recurrent {
        &self.recurrent
    }

    /// Runs one step from `tokens`, `[B]` of `I64`, and returns the output, `[B, O]`, and the new state.
    pub fn predict(
        &self,
        tokens: &Tensor,
        recurrent: &RecurrentState,
        state: &ModelState,
    ) -> SingeResult<(Tensor, RecurrentState)> {
        let embedded = tokens.apply(&self.embedding, state)?;
        let (output, recurrent) = self.recurrent.step(&embedded, recurrent, state)?;
        let output = match &self.projection {
            Some(projection) => output.apply(projection, state)?,
            None => output,
        };
        Ok((output, recurrent))
    }
}

/// The joint network: the encoder's rows and the prediction network's output, each projected to its width, added, activated, and scored.
#[derive(Debug, Clone)]
pub struct JointNetwork {
    encoder: Linear,
    prediction: Option<Linear>,
    activation: Activation,
    head: Linear,
}

impl JointNetwork {
    pub fn create(schema: &mut Schema<'_>, config: &TransducerConfig) -> SingeResult<Self> {
        let width = config.joint_width;
        Ok(Self {
            encoder: schema.linear("encoder", config.encoder_width, width, true)?,
            prediction: config
                .joint_prediction
                .then(|| schema.linear("prediction", config.prediction.output(), width, true))
                .transpose()?,
            activation: config.joint_activation,
            head: schema.linear("head", width, config.logits(), true)?,
        })
    }

    /// Returns the encoder's rows, `[…, W]`, projected to the joint's width, which decoding needs once per clip.
    pub fn project(&self, rows: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        rows.apply(&self.encoder, state)
    }

    /// Returns the logits of projected rows `[B, J]` against the prediction network's outputs `[B, O]`.
    pub fn joint(
        &self,
        rows: &Tensor,
        prediction: &Tensor,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let prediction = match &self.prediction {
            Some(linear) => prediction.apply(linear, state)?,
            None => prediction.clone(),
        };
        rows.add(&prediction)?
            .apply(&self.activation, state)?
            .apply(&self.head, state)
    }
}

/// A transducer's layers past its encoder: the prediction and joint networks, `prediction` and `joint`.
#[derive(Debug, Clone)]
pub struct TransducerLayers {
    pub prediction: PredictionNetwork,
    pub joint: JointNetwork,
    pub config: TransducerConfig,
}

impl TransducerLayers {
    pub fn create(schema: &mut Schema<'_>, config: &TransducerConfig) -> SingeResult<Self> {
        Ok(Self {
            prediction: PredictionNetwork::create(&mut schema.scope("prediction"), config)?,
            joint: JointNetwork::create(&mut schema.scope("joint"), config)?,
            config: config.clone(),
        })
    }
}

impl Layer for TransducerLayers {
    /// Returns the joint's projection of the encoder's rows, which is all the transducer computes of them at once.
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        self.joint.project(input, state)
    }
}

/// One emission of greedy decoding: the token, the encoder row it was emitted at, and the rows it spans, one for RNN-T.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct TransducerStep {
    pub token: i64,
    pub row: i64,
    pub duration: i64,
}

/// Decodes one clip greedily: `rows`, the joint's projection of the encoder's rows, `[T, J]`, of which the first `count` are the clip's own, emitting at most `max_tokens` tokens in at most `max_steps` steps.
/// The prediction network starts from the blank and a zero state, and runs again only after a token other than the blank; each step takes the best token and, for TDT, the best duration.
/// It stops when decoding passes the clip's rows, emits the end token, has taken `max_steps` steps, or has emitted `max_tokens`.
pub fn greedy_decode(
    layers: &TransducerLayers,
    rows: &Tensor,
    count: i64,
    max_tokens: usize,
    max_steps: i64,
    state: &ModelState,
) -> SingeResult<Vec<TransducerStep>> {
    let config = &layers.config;
    let &[row_count, _] = rows.shape().dimensions() else {
        return Err(SingeError::invalid_argument(
            "greedy decoding reads one clip's rows, [rows, width]",
        ));
    };
    let count = count.min(row_count);
    let device = rows.device();
    let token_tensor = |token: i64| Tensor::from_vec(device, Shape::create([1])?, vec![token]);
    let mut recurrent = layers
        .prediction
        .recurrent()
        .zeros(1, device, rows.data_type())?;
    let mut last = config.blank;
    let mut prediction = None;
    let mut emitted = Vec::new();
    let mut row = 0;
    let mut symbols = 0;
    for _ in 0..max_steps {
        if row >= count || emitted.len() >= max_tokens {
            break;
        }

        if prediction.is_none() || last != config.blank {
            let (output, next) =
                layers
                    .prediction
                    .predict(&token_tensor(last)?, &recurrent, state)?;
            prediction = Some(output);
            recurrent = next;
        }

        let Some(output) = &prediction else {
            return Err(SingeError::missing_output(
                "the prediction network's output",
            ));
        };
        let logits = layers
            .joint
            .joint(&rows.narrow(0, row, 1)?, output, state)?
            .cast(DataType::F32)?;
        let (token, duration) = best(&logits, config.vocabulary)?;
        if Some(token) == config.end {
            break;
        }

        let blank = token == config.blank;
        let advance = match &config.decoding {
            TransducerDecoding::Rnnt { max_symbols } => {
                symbols = if blank { 0 } else { symbols + 1 };
                if blank || symbols >= *max_symbols {
                    symbols = 0;
                    1
                } else {
                    0
                }
            }
            TransducerDecoding::Tdt {
                durations,
                max_symbols,
            } => {
                let index = usize::try_from(duration.unwrap_or(0))
                    .map_err(|_| SingeError::out_of_range("a TDT duration"))?;
                let mut advance = *durations
                    .get(index)
                    .ok_or_else(|| SingeError::out_of_range("a TDT duration"))?;
                if blank && advance == 0 {
                    advance = 1;
                }

                symbols = if advance > 0 { 0 } else { symbols + 1 };
                if max_symbols.is_some_and(|max| symbols >= max) {
                    symbols = 0;
                    advance = 1;
                }
                advance
            }
        };
        if !blank {
            let duration = match &config.decoding {
                TransducerDecoding::Rnnt { .. } => 1,
                TransducerDecoding::Tdt { .. } => advance,
            };
            emitted.push(TransducerStep {
                token,
                row,
                duration,
            });
        }

        last = token;
        row += advance;
    }
    Ok(emitted)
}

/// Returns the best token among the first `vocabulary` logits of `[1, L]`, and the best duration among the rest, if any, read back together.
fn best(logits: &Tensor, vocabulary: i64) -> SingeResult<(i64, Option<i64>)> {
    let total = logits.shape().dimensions()[1];
    let (_, token) = logits.narrow(1, 0, vocabulary)?.topk(1, -1, true, false)?;
    if total == vocabulary {
        let token = token.contiguous()?.to_vec::<i64>()?;
        return Ok((token[0], None));
    }

    let (_, duration) = logits
        .narrow(1, vocabulary, total - vocabulary)?
        .topk(1, -1, true, false)?;
    let both = Tensor::concat(&[token, duration], 1)?
        .contiguous()?
        .to_vec::<i64>()?;
    Ok((both[0], Some(both[1])))
}

/// Imports the prediction and joint networks `transformers` stores for Parakeet's transducers: `decoder.embedding`, `decoder.lstm`, `decoder.decoder_projector`, `encoder_projector`, and `joint.head`, under `to`.
#[must_use]
pub fn transducer_rules(to: &str) -> Rules {
    Rules::new()
        .rename("decoder.embedding", &join(to, "prediction.embedding"))
        .rename(
            "decoder.decoder_projector",
            &join(to, "prediction.projection"),
        )
        .then(recurrent_rules(
            "decoder.lstm",
            &join(to, "prediction.recurrent"),
        ))
        .rename("encoder_projector", &join(to, "joint.encoder"))
        .rename("joint.head", &join(to, "joint.head"))
}
