// CPU kernels of the operations that carry a state per sequence, computed in `F64` as the definitions read.

use crate::{
    backend::registry::Operation,
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    tensor::Tensor,
};

// Where each sequence's rows sit in a step, read from its query offsets, real token counts, and positions.
struct Spans {
    offsets: Vec<i64>,
    real: Vec<i64>,
    positions: Vec<i64>,
}

impl Spans {
    fn read(offsets: &Tensor, real: &Tensor, positions: &Tensor) -> Result<Self> {
        let indices = |tensor: &Tensor| tensor.cast(DataType::I64)?.to_vec::<i64>();
        Ok(Self {
            offsets: indices(offsets)?,
            real: indices(real)?,
            positions: indices(positions)?,
        })
    }

    fn sequences(&self) -> usize {
        self.real.len()
    }

    // Returns sequence `sequence`'s first row, its count of real rows, and whether it starts from a zero state.
    fn of(&self, sequence: usize) -> Result<(usize, usize, bool)> {
        let index = |value: i64| {
            usize::try_from(value).map_err(|_| Error::out_of_range("state recurrence spans"))
        };
        let (start, count) = (index(self.offsets[sequence])?, index(self.real[sequence])?);
        let fresh = count > 0
            && self
                .positions
                .get(start)
                .is_some_and(|&position| position == 0);
        Ok((start, count, fresh))
    }
}

fn values(tensor: &Tensor) -> Result<Vec<f64>> {
    tensor.cast(DataType::F64)?.contiguous()?.to_vec::<f64>()
}

fn tensor_like(values: Vec<f64>, like: &Tensor, data_type: DataType) -> Result<Tensor> {
    Tensor::from_vec(Device::Cpu, like.shape().clone(), values)?.cast(data_type)
}

// Runs `operation`, one of the causal convolution and the gated delta rule and their states, on `inputs`.
pub fn state_recurrence(operation: Operation, inputs: &[Tensor]) -> Result<Tensor> {
    match (operation, inputs) {
        (Operation::CausalConv1d, [input, weight, state, offsets, real, positions]) => {
            let spans = Spans::read(offsets, real, positions)?;
            causal_conv1d(input, Some(weight), state, &spans)
        }
        (Operation::CausalConv1dState, [input, state, offsets, real, positions]) => {
            let spans = Spans::read(offsets, real, positions)?;
            causal_conv1d(input, None, state, &spans)
        }
        (
            Operation::GatedDeltaRule | Operation::GatedDeltaRuleState,
            [
                query,
                key,
                value,
                decay,
                strength,
                state,
                offsets,
                real,
                positions,
            ],
        ) => {
            let spans = Spans::read(offsets, real, positions)?;
            gated_delta_rule(
                [query, key, value, decay, strength, state],
                &spans,
                operation == Operation::GatedDeltaRuleState,
            )
        }
        _ => Err(Error::missing_input("cpu state recurrence operands")),
    }
}

// Returns the convolution's output when given its taps, and else the state it leaves.
fn causal_conv1d(
    input: &Tensor,
    weight: Option<&Tensor>,
    state: &Tensor,
    spans: &Spans,
) -> Result<Tensor> {
    let &[_, channels, history] = state.shape().dimensions() else {
        return Err(Error::invalid_argument("causal convolution state rank"));
    };
    let to_usize =
        |value: i64| usize::try_from(value).map_err(|_| Error::out_of_range("causal convolution"));
    let (channels, history) = (to_usize(channels)?, to_usize(history)?);
    let (rows, mut kept) = (values(input)?, values(state)?);
    let taps = weight.map(values).transpose()?;
    let mut output = vec![0.0; rows.len()];
    for sequence in 0..spans.sequences() {
        let (start, count, fresh) = spans.of(sequence)?;
        if count == 0 {
            continue;
        }

        for channel in 0..channels {
            let held = &mut kept[(sequence * channels + channel) * history..][..history];
            // The inputs the window slides over: the state's, or zeros at a sequence's start, then its rows.
            let window = (0..history)
                .map(|index| if fresh { 0.0 } else { held[index] })
                .chain((0..count).map(|row| rows[(start + row) * channels + channel]))
                .collect::<Vec<_>>();
            if let Some(taps) = &taps {
                let taps = &taps[channel * (history + 1)..][..=history];
                for row in 0..count {
                    output[(start + row) * channels + channel] = taps
                        .iter()
                        .zip(&window[row..=row + history])
                        .map(|(tap, value)| tap * value)
                        .sum();
                }
            }

            held.copy_from_slice(&window[window.len() - history..]);
        }
    }

    match weight {
        Some(_) => tensor_like(output, input, input.data_type()),
        None => tensor_like(kept, state, state.data_type()),
    }
}

// Returns the rule's output, or with `want_state` the state it leaves.
fn gated_delta_rule(operands: [&Tensor; 6], spans: &Spans, want_state: bool) -> Result<Tensor> {
    let [query, key, value, decay, strength, state] = operands;
    let &[_, heads, key_dim, value_dim] = state.shape().dimensions() else {
        return Err(Error::invalid_argument("gated delta rule state rank"));
    };
    let to_usize =
        |value: i64| usize::try_from(value).map_err(|_| Error::out_of_range("gated delta rule"));
    let (heads, key_dim, value_dim) = (to_usize(heads)?, to_usize(key_dim)?, to_usize(value_dim)?);
    let (queries, keys, values_in) = (values(query)?, values(key)?, values(value)?);
    let (decays, strengths, mut states) = (values(decay)?, values(strength)?, values(state)?);
    let mut output = vec![0.0; values_in.len()];
    for sequence in 0..spans.sequences() {
        let (start, count, fresh) = spans.of(sequence)?;
        if count == 0 {
            continue;
        }

        for head in 0..heads {
            let held = &mut states[(sequence * heads + head) * key_dim * value_dim..]
                [..key_dim * value_dim];
            if fresh {
                held.fill(0.0);
            }

            for row in (start..start + count).map(|row| row * heads + head) {
                let (q, k) = (
                    &queries[row * key_dim..][..key_dim],
                    &keys[row * key_dim..][..key_dim],
                );
                let v = &values_in[row * value_dim..][..value_dim];
                let factor = decays[row].exp();
                for entry in held.iter_mut() {
                    *entry *= factor;
                }
                for column in 0..value_dim {
                    let read = (0..key_dim)
                        .map(|dim| held[dim * value_dim + column] * k[dim])
                        .sum::<f64>();
                    let delta = (v[column] - read) * strengths[row];
                    for dim in 0..key_dim {
                        let entry = &mut held[dim * value_dim + column];
                        *entry = k[dim].mul_add(delta, *entry);
                    }

                    output[row * value_dim + column] = (0..key_dim)
                        .map(|dim| held[dim * value_dim + column] * q[dim])
                        .sum();
                }
            }
        }
    }

    if want_state {
        tensor_like(states, state, DataType::F32)
    } else {
        tensor_like(output, value, value.data_type())
    }
}
