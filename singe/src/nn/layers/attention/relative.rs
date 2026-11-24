// Self-attention over relative positions, as Transformer-XL and the Conformers of NeMo, ESPnet, and wav2vec2-conformer compute it.
// Scores are `((q + u) kᵀ + (q + v) rᵀ) / √D`, where `r` projects the sinusoids of each query's distance to each key, and `u` and `v` are learned biases per head.

use serde::{Deserialize, Serialize};

use crate::{
    data_type::DataType,
    error::{Error, Result},
    graph::capture,
    nn::{Layer, Linear, ModelState, StateKey},
    scalar::Scalar,
    shape::Shape,
    tensor::{AttentionOptions, Tensor},
};

// The projections and biases of a [`RelativeAttention`]: queries, keys, values, and output, `positions` without a bias, and the content and position biases `[H, D]`.
#[derive(Debug, Clone)]
pub struct RelativeAttentionLayers {
    pub query: Linear,
    pub key: Linear,
    pub value: Linear,
    pub output: Linear,
    pub positions: Linear,
    pub content_bias: StateKey,
    pub position_bias: StateKey,
}

// Local attention, as NeMo's `rel_pos_local_attn` computes it: above `above` rows, each row attends only to the `window` rows before and after it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalAttention {
    pub window: i64,
    pub above: i64,
}

// What relative attention reads: `hidden`, `[B, T, W]`; `positions`, the embeddings of the distances `R` down to `-R`, `[2R + 1, W]`, where `R` is [`RelativeAttention::reach`] of `T`; and `key_lengths`, `[B]`, each sequence's count of real steps, which come first, if a batch's sequences differ in length.
#[derive(Debug, Clone, Copy)]
pub struct RelativeAttentionInput<'a> {
    pub hidden: &'a Tensor,
    pub positions: &'a Tensor,
    pub key_lengths: Option<&'a Tensor>,
}

// The bytes of scores against distances that one chunk of queries holds; a layer whose queries all fit runs them at once.
const SCORE_BUDGET: f64 = (256 << 20) as f64;

// Chunks of queries are a multiple of this many rows.
const CHUNK_ROWS: i64 = 64;

// Multi-head self-attention over relative positions, attending to every key, or with `local`, to the keys near each query.
// Queries run in chunks when their scores against every distance would pass `SCORE_BUDGET`, so a layer's memory grows with the length times the chunk rather than the length's square; each chunk gives exactly the rows full attention gives.
// Local attention runs its queries in groups of `2 · window`, each against its `4 · window` nearest keys, so its memory and work grow with the length.
#[derive(Debug, Clone)]
pub struct RelativeAttention {
    layers: RelativeAttentionLayers,
    heads: i64,
    head_dim: i64,
    local: Option<LocalAttention>,
}

impl RelativeAttention {
    pub fn create(layers: RelativeAttentionLayers, heads: i64, width: i64) -> Result<Self> {
        if heads <= 0 || width <= 0 || width % heads != 0 {
            return Err(Error::invalid_argument("attention heads and width"));
        }

        Ok(Self {
            layers,
            heads,
            head_dim: width / heads,
            local: None,
        })
    }

    pub fn local(mut self, local: LocalAttention) -> Result<Self> {
        if local.window <= 0 || local.above < 0 {
            return Err(Error::invalid_argument(
                "a local attention's window and rows",
            ));
        }

        self.local = Some(local);
        Ok(self)
    }

    // Whether `steps` rows attend locally: when local attention is on and they are more than its `above`, even if its window reaches every row.
    #[must_use]
    pub const fn attends_locally(&self, steps: i64) -> bool {
        matches!(self.local, Some(local) if steps > local.above)
    }

    // The largest distance whose position attention reads over `steps` rows: the local window when it attends locally, or `steps - 1`.
    #[must_use]
    pub const fn reach(&self, steps: i64) -> i64 {
        match self.local {
            Some(local) if self.attends_locally(steps) => local.window,
            _ => steps - 1,
        }
    }

    #[must_use]
    pub const fn layers(&self) -> &RelativeAttentionLayers {
        &self.layers
    }

    // `[B, T, W]` to `[B, H, T, D]`.
    fn heads(&self, tensor: &Tensor) -> Result<Tensor> {
        let &[batch, steps, _] = tensor.shape().dimensions() else {
            return Err(Error::invalid_argument(
                "relative attention reads [batch, steps, width]",
            ));
        };
        tensor
            .reshape([batch, steps, self.heads, self.head_dim])?
            .transpose(1, 2)
    }

    fn bias(&self, key: &StateKey, state: &ModelState) -> Result<Tensor> {
        state
            .tensor(key)?
            .reshape([1, self.heads, 1, self.head_dim])
    }

    // The rows of queries one chunk takes: all `steps` when their scores against every distance fit `SCORE_BUDGET`, or the most rows, a multiple of `CHUNK_ROWS`, whose `[B, H, rows, rows + steps - 1]` scores fit it.
    fn chunk_rows(&self, batch: i64, steps: i64, data_type: DataType) -> i64 {
        let bytes = |rows: i64| {
            (batch * self.heads) as f64
                * rows as f64
                * (rows + steps - 1) as f64
                * data_type.size_bytes() as f64
        };
        if bytes(steps) <= SCORE_BUDGET {
            return steps;
        }

        let mut rows = CHUNK_ROWS;
        while rows + CHUNK_ROWS < steps && bytes(rows + CHUNK_ROWS) <= SCORE_BUDGET {
            rows += CHUNK_ROWS;
        }
        rows.min(steps)
    }
}

// Local attention over `[B, H, T, D]` queries, keys, and values, each query attending to the keys up to `window` rows from it, with `positions` the projections of the distances `window` down to `-window`.
// Queries run in groups of `c = 2 · window` rows; group `g` reads the `2c` keys from `g · c - window`, so with the keys padded by `window` at each end, a group's keys are the padded keys' groups `g` and `g + 1`.
// Query `a` of a group reads its key `k` at distance column `k - a`, which a skew of the `[c, 2 · window + 1]` distance scores to `[c, 2c]` gives; a band mask hides the keys outside the window, and another the keys past each sequence's length.
// The groups run in batches whose scores fit `SCORE_BUDGET`.
fn attend_local(input: &LocalInput<'_>, window: i64, scale: f64) -> Result<Tensor> {
    let &[batch, heads, steps, depth] = input.content.shape().dimensions() else {
        return Err(Error::invalid_argument(
            "local attention reads [batch, heads, steps, depth]",
        ));
    };
    let data_type = input.content.data_type();
    let device = input.content.device();
    let group = 2 * window;
    let groups = (steps + group - 1) / group;
    let padded = groups * group;
    let rows = |tensor: &Tensor, before: i64, after: i64| {
        tensor.pad(&[(0, 0), (0, 0), (before, after), (0, 0)], 0.0)
    };
    let content = rows(input.content, 0, padded - steps)?;
    let positional = rows(input.positional, 0, padded - steps)?;
    let key = rows(input.key, window, padded - steps + window)?;
    let value = rows(input.value, window, padded - steps + window)?;
    let lengths = match input.lengths {
        Some(lengths) => lengths.cast(DataType::I64)?,
        None => capture::literal(device, |device| {
            Tensor::from_vec(device, Shape::create([1])?, vec![steps])
        })?,
    }
    .reshape([-1, 1, 1, 1])?;
    // Finite, unlike `padding_mask`'s, so that a query with no key in reach, which only a row past its sequence's length can be, gets finite weights rather than NaN that its keys would carry into the next layer; half of `F16`'s lowest for `F16`.
    let masked = match data_type {
        DataType::F16 => -32_752.0,
        _ => -1.0e30,
    };
    let additive = |value: f64| {
        capture::literal(device, |device| {
            Scalar::Float(value).tensor(device, data_type)
        })
    };
    let band = capture::literal(device, |device| {
        let values = (0..group)
            .flat_map(|query| {
                (0..2 * group).map(move |key| {
                    if (0..=2 * window).contains(&(key - query)) {
                        0.0
                    } else {
                        masked as f32
                    }
                })
            })
            .collect();
        Tensor::from_vec(device, Shape::create([group, 2 * group])?, values)?.cast(data_type)
    })?;
    let per_group = (batch * heads * group * 2 * group) as f64 * data_type.size_bytes() as f64;
    let batch_groups = ((SCORE_BUDGET / per_group) as i64).clamp(1, groups);
    let grouped = |tensor: &Tensor, first: i64, count: i64| {
        tensor
            .narrow(2, first * group, count * group)?
            .contiguous()?
            .reshape([batch, heads, count, group, depth])
    };
    let mut pieces = Vec::new();
    let mut first = 0;
    while first < groups {
        let count = batch_groups.min(groups - first);
        let windows = |tensor: &Tensor| -> Result<Tensor> {
            let blocks = grouped(tensor, first, count + 1)?;
            Tensor::concat(
                &[blocks.narrow(2, 0, count)?, blocks.narrow(2, 1, count)?],
                3,
            )?
            .reshape([batch, heads * count, 2 * group, depth])
        };
        // Each key's row in the clip, with the padding before it moved past the end so that one comparison with the lengths hides both.
        let positions = capture::literal(device, |device| {
            let values = (first..first + count)
                .flat_map(|index| {
                    (0..2 * group).map(move |key| {
                        let row = index * group - window + key;
                        if row < 0 { i64::MAX } else { row }
                    })
                })
                .collect();
            Tensor::from_vec(device, Shape::create([1, count, 1, 2 * group])?, values)
        })?;
        let mask = Tensor::where_select(&positions.less(&lengths)?, &band, &additive(masked)?)?
            .unsqueeze(1)?;
        let scores = positional
            .narrow(2, first * group, count * group)?
            .matmul(input.positions)?
            .reshape([batch, heads, count, group, 2 * window + 1])?
            .pad(&[(0, 0), (0, 0), (0, 0), (0, 0), (0, group)], 0.0)?
            .contiguous()?
            .reshape([batch, heads, count, group * (2 * group + 1)])?
            .narrow(-1, 0, group * 2 * group)?
            .contiguous()?
            .reshape([batch, heads, count, group, 2 * group])?
            .mul_scalar(scale)?
            .add(&mask)?
            .reshape([batch, heads * count, group, 2 * group])?;
        let attended = grouped(&content, first, count)?
            .reshape([batch, heads * count, group, depth])?
            .attend(
                &windows(&key)?,
                &windows(&value)?,
                Some(&scores),
                None,
                AttentionOptions {
                    scale: Some(scale as f32),
                    ..AttentionOptions::default()
                },
            )?;
        pieces.push(attended.reshape([batch, heads, count * group, depth])?);
        first += count;
    }

    let attended = if pieces.len() == 1 {
        pieces.remove(0)
    } else {
        Tensor::concat(&pieces, 2)?
    };
    attended.narrow(2, 0, steps)
}

// The tensors local attention reads, `[B, H, T, D]` but for `positions`, `[1, H, D, 2 · window + 1]`.
struct LocalInput<'a> {
    content: &'a Tensor,
    positional: &'a Tensor,
    key: &'a Tensor,
    value: &'a Tensor,
    positions: &'a Tensor,
    lengths: Option<&'a Tensor>,
}

impl Layer<RelativeAttentionInput<'_>> for RelativeAttention {
    fn forward(&self, input: &RelativeAttentionInput<'_>, state: &ModelState) -> Result<Tensor> {
        let layers = &self.layers;
        let hidden = input.hidden;
        let &[batch, steps, width] = hidden.shape().dimensions() else {
            return Err(Error::invalid_argument(
                "relative attention reads [batch, steps, width]",
            ));
        };
        let reach = self.reach(steps);
        let distances = 2 * reach + 1;
        if input.positions.shape().dimensions() != [distances, width] {
            return Err(Error::invalid_argument(
                "relative attention's positions, [2 · reach + 1, width]",
            ));
        }

        let query = self.heads(&hidden.apply(&layers.query, state)?)?;
        let key = self.heads(&hidden.apply(&layers.key, state)?)?;
        let value = self.heads(&hidden.apply(&layers.value, state)?)?;
        // The positions' projection per head, `[1, H, D, 2R + 1]`.
        let positions = input
            .positions
            .apply(&layers.positions, state)?
            .reshape([distances, self.heads, self.head_dim])?
            .permute(&[1, 2, 0])?
            .contiguous()?
            .unsqueeze(0)?;
        let content = query.add(&self.bias(&layers.content_bias, state)?)?;
        let positional = query.add(&self.bias(&layers.position_bias, state)?)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        if self.attends_locally(steps) {
            let attended = attend_local(
                &LocalInput {
                    content: &content,
                    positional: &positional,
                    key: &key,
                    value: &value,
                    positions: &positions,
                    lengths: input.key_lengths,
                },
                reach,
                scale,
            )?;
            return attended
                .transpose(1, 2)?
                .reshape([batch, steps, width])?
                .apply(&layers.output, state);
        }

        let padding = input
            .key_lengths
            .map(|lengths| padding_mask(lengths, steps, hidden.data_type()))
            .transpose()?;
        let rows = self.chunk_rows(batch, steps, hidden.data_type());
        let mut chunks = Vec::new();
        let mut start = 0;
        while start < steps {
            let length = rows.min(steps - start);
            // The distances the chunk's queries have to the keys, `T - 1 - i + j` for query `i` and key `j`, are the columns from `T - length - start` on.
            let distances = positions.narrow(3, steps - length - start, length + steps - 1)?;
            let mut scores = rel_shift(&positional.narrow(2, start, length)?.matmul(&distances)?)?
                .mul_scalar(scale)?;
            if let Some(padding) = &padding {
                scores = scores.add(padding)?;
            }

            chunks.push(content.narrow(2, start, length)?.attend(
                &key,
                &value,
                Some(&scores),
                None,
                AttentionOptions {
                    scale: Some(scale as f32),
                    ..AttentionOptions::default()
                },
            )?);
            start += length;
        }

        let attended = if chunks.len() == 1 {
            chunks.remove(0)
        } else {
            Tensor::concat(&chunks, 2)?
        };
        attended
            .transpose(1, 2)?
            .reshape([batch, steps, width])?
            .apply(&layers.output, state)
    }
}

// Shifts the scores of `c` queries against distances, `[…, c, c + T - 1]`, whose row `a` holds query `a`'s score for key `j` at column `c - 1 - a + j`, to scores against each of the `T` keys, `[…, c, T]`.
// It pads a column before the first, flattens the last two axes, drops the first `c` values, and reads the rest back as `[c, c + T - 1]`, as Transformer-XL's `rel_shift` does for `c = T`.
fn rel_shift(scores: &Tensor) -> Result<Tensor> {
    let dimensions = scores.shape().dimensions().to_vec();
    let rank = dimensions.len();
    let (&[rows, columns], outer) = (
        &dimensions[rank.saturating_sub(2)..],
        &dimensions[..rank.saturating_sub(2)],
    ) else {
        return Err(Error::invalid_argument(
            "relative scores, [..., queries, queries + keys - 1]",
        ));
    };
    if columns < rows {
        return Err(Error::invalid_argument(
            "relative scores, [..., queries, queries + keys - 1]",
        ));
    }

    let keys = columns - rows + 1;
    let mut padding = vec![(0, 0); rank];
    padding[rank - 1] = (1, 0);
    let shape = |last: &[i64]| -> Vec<i64> { outer.iter().chain(last).copied().collect() };
    scores
        .pad(&padding, 0.0)?
        .contiguous()?
        .reshape(shape(&[rows * (columns + 1)]))?
        .narrow(-1, rows, rows * columns)?
        .contiguous()?
        .reshape(shape(&[rows, columns]))?
        .narrow(-1, 0, keys)
}

// An additive mask that hides each sequence's keys from its length on, `[B, 1, 1, T]`.
fn padding_mask(lengths: &Tensor, steps: i64, data_type: DataType) -> Result<Tensor> {
    let device = lengths.device();
    let positions =
        capture::literal(device, |device| Tensor::arange(device, 0_i64, steps, 1))?.unsqueeze(0)?;
    let kept = positions.less(&lengths.cast(DataType::I64)?.unsqueeze(1)?)?;
    let additive = |value: f64| {
        capture::literal(device, |device| {
            Scalar::Float(value).tensor(device, data_type)
        })
    };
    Tensor::where_select(&kept, &additive(0.0)?, &additive(f64::NEG_INFINITY)?)?
        .unsqueeze(1)?
        .unsqueeze(1)
}
