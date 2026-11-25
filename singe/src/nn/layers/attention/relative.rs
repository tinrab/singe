// Self-attention over relative positions, as Transformer-XL and the Conformers of NeMo, ESPnet, and wav2vec2-conformer compute it.
// Scores are `((q + u) kᵀ + (q + v) rᵀ) / √D`, where `r` projects the sinusoids of each query's distance to each key, and `u` and `v` are learned biases per head.

use serde::{Deserialize, Serialize};

use crate::{
    data_type::DataType,
    error::{Error, Result},
    graph::{DimensionExpr, capture},
    nn::{Layer, Linear, ModelState, StateKey},
    scalar::Scalar,
    shape::{Shape, spatial::Padding},
    tensor::{AttentionOptions, Tensor},
};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalAttention {
    pub window: i64,
    pub above: i64,
}

#[derive(Debug, Clone, Copy)]
pub struct RelativeAttentionInput<'a> {
    pub hidden: &'a Tensor,
    pub positions: &'a Tensor,
    pub key_lengths: Option<&'a Tensor>,
}

const SCORE_BUDGET: f64 = (256 << 20) as f64;

const CHUNK_ROWS: i64 = 64;

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

    // Whether the attention reads `steps` rows through its local window.
    // A program with run-time axes does not know its rows while it records, so it attends to every row, as `transformers` does at any length; a window at every length has no such program.
    pub fn attends_locally(&self, steps: i64) -> Result<bool> {
        let Some(local) = self.local else {
            return Ok(false);
        };
        if !capture::has_run_time_axes()? {
            return Ok(steps > local.above);
        }

        if local.above == 0 {
            return Err(Error::unsupported_operation_with_details(
                "relative attention",
                "a local window at every length under run-time axes",
            ));
        }
        Ok(false)
    }

    pub fn reach(&self, steps: i64) -> Result<i64> {
        Ok(match self.local {
            Some(local) if self.attends_locally(steps)? => local.window,
            _ => steps - 1,
        })
    }

    #[must_use]
    pub const fn layers(&self) -> &RelativeAttentionLayers {
        &self.layers
    }

    fn heads(&self, tensor: &Tensor) -> Result<Tensor> {
        if tensor.shape().rank() != 3 {
            return Err(Error::invalid_argument(
                "relative attention reads [batch, steps, width]",
            ));
        }

        tensor
            .reshape_with(&[
                tensor.dimension(0)?,
                tensor.dimension(1)?,
                self.heads.into(),
                self.head_dim.into(),
            ])?
            .transpose(1, 2)
    }

    fn bias(&self, key: &StateKey, state: &ModelState) -> Result<Tensor> {
        state
            .tensor(key)?
            .reshape([1, self.heads, 1, self.head_dim])
    }

    // The query rows each chunk of scores holds, every row when they fit `SCORE_BUDGET` or the program has run-time axes.
    fn chunk_rows(&self, batch: i64, steps: i64, data_type: DataType) -> Result<i64> {
        if capture::has_run_time_axes()? {
            return Ok(steps);
        }

        let bytes = |rows: i64| {
            (batch * self.heads) as f64
                * rows as f64
                * (rows + steps - 1) as f64
                * data_type.size_bytes() as f64
        };
        if bytes(steps) <= SCORE_BUDGET {
            return Ok(steps);
        }

        let mut rows = CHUNK_ROWS;
        while rows + CHUNK_ROWS < steps && bytes(rows + CHUNK_ROWS) <= SCORE_BUDGET {
            rows += CHUNK_ROWS;
        }
        Ok(rows.min(steps))
    }
}

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
        tensor.pad(
            &[
                Padding::ZERO,
                Padding::ZERO,
                Padding::new(before, after),
                Padding::ZERO,
            ],
            0.0,
        )
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
            .pad(
                &[
                    Padding::ZERO,
                    Padding::ZERO,
                    Padding::ZERO,
                    Padding::ZERO,
                    Padding::new(0, group),
                ],
                0.0,
            )?
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
        let reach = self.reach(steps)?;
        if input.positions.shape().dimensions() != [2 * reach + 1, width] {
            return Err(Error::invalid_argument(
                "relative attention's positions, [2 · reach + 1, width]",
            ));
        }

        let query = self.heads(&hidden.apply(&layers.query, state)?)?;
        let key = self.heads(&hidden.apply(&layers.key, state)?)?;
        let value = self.heads(&hidden.apply(&layers.value, state)?)?;
        // The positions' projection per head, `[1, H, D, 2R + 1]`.
        let projected = input.positions.apply(&layers.positions, state)?;
        let positions = projected
            .reshape_with(&[
                projected.dimension(0)?,
                self.heads.into(),
                self.head_dim.into(),
            ])?
            .permute(&[1, 2, 0])?
            .contiguous()?
            .unsqueeze(0)?;
        let content = query.add(&self.bias(&layers.content_bias, state)?)?;
        let positional = query.add(&self.bias(&layers.position_bias, state)?)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let output_shape = [hidden.dimension(0)?, hidden.dimension(1)?, width.into()];
        if self.attends_locally(steps)? {
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
                .reshape_with(&output_shape)?
                .apply(&layers.output, state);
        }

        let padding = input
            .key_lengths
            .map(|lengths| padding_mask(lengths, hidden.dimension(1)?, hidden.data_type()))
            .transpose()?;
        let rows = self.chunk_rows(batch, steps, hidden.data_type())?;
        let mut chunks = Vec::new();
        let mut start = 0;
        while start < steps {
            let length = rows.min(steps - start);
            // The distances the chunk's queries have to the keys, `T - 1 - i + j` for query `i` and key `j`, are the columns from `T - length - start` on, every column for a chunk of every row.
            let (distances, queries) = if length == steps {
                (positions.clone(), positional.clone())
            } else {
                (
                    positions.narrow(3, steps - length - start, length + steps - 1)?,
                    positional.narrow(2, start, length)?,
                )
            };
            let mut scores = rel_shift(&queries.matmul(&distances)?)?.mul_scalar(scale)?;
            if let Some(padding) = &padding {
                scores = scores.add(padding)?;
            }

            let content = if length == steps {
                content.clone()
            } else {
                content.narrow(2, start, length)?
            };
            chunks.push(content.attend(
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
            .reshape_with(&output_shape)?
            .apply(&layers.output, state)
    }
}

fn rel_shift(scores: &Tensor) -> Result<Tensor> {
    let dimensions = scores.shape().dimensions().to_vec();
    let rank = dimensions.len();
    if rank < 2 || dimensions[rank - 1] < dimensions[rank - 2] {
        return Err(Error::invalid_argument(
            "relative scores, [..., queries, queries + keys - 1]",
        ));
    }

    let extents = (0..rank)
        .map(|axis| scores.dimension(axis))
        .collect::<Result<Vec<_>>>()?;
    let (outer, last) = extents.split_at(rank - 2);
    let (rows, columns) = (last[0].clone(), last[1].clone());
    let keys = columns.clone() - rows.clone() + 1;
    let mut padding = vec![Padding::ZERO; rank];
    padding[rank - 1] = Padding::new(1, 0);
    let shape = |last: &[DimensionExpr]| outer.iter().chain(last).cloned().collect::<Vec<_>>();
    scores
        .pad(&padding, 0.0)?
        .contiguous()?
        .reshape_with(&shape(&[rows.clone() * (columns.clone() + 1)]))?
        .narrow_with(-1, rows.clone(), rows.clone() * columns.clone())?
        .contiguous()?
        .reshape_with(&shape(&[rows, columns]))?
        .narrow_with(-1, 0_i64, keys)
}

fn padding_mask(lengths: &Tensor, steps: DimensionExpr, data_type: DataType) -> Result<Tensor> {
    let device = lengths.device();
    let positions = Tensor::range(device, 0_i64, steps)?.unsqueeze(0)?;
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
