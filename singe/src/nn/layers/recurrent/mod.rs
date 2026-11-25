// Recurrent layers, as `torch.nn.RNN`, `LSTM`, and `GRU` compute them in inference: stacked, optionally bidirectional, and for LSTMs optionally projected.
// Each step is matmuls, activations, and elementwise operations, so a layer runs eagerly and under capture on every backend.

use std::iter;

use serde::{Deserialize, Serialize};

use crate::{
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    graph::capture,
    nn::{Layer, Linear, ModelState},
    scalar::Scalar,
    shape::Shape,
    tensor::Tensor,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecurrentCell {
    Tanh,
    Relu,
    Lstm,
    Gru,
}

impl RecurrentCell {
    #[must_use]
    pub const fn gates(self) -> i64 {
        match self {
            Self::Tanh | Self::Relu => 1,
            Self::Lstm => 4,
            Self::Gru => 3,
        }
    }

    #[must_use]
    pub const fn has_cell_state(self) -> bool {
        matches!(self, Self::Lstm)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecurrentConfig {
    pub cell: RecurrentCell,
    pub input: i64,
    pub hidden: i64,
    pub layers: i64,
    pub bidirectional: bool,
    pub biases: bool,
    pub projection: Option<i64>,
}

impl RecurrentConfig {
    #[must_use]
    pub const fn directions(&self) -> i64 {
        if self.bidirectional { 2 } else { 1 }
    }

    #[must_use]
    pub const fn output(&self) -> i64 {
        match self.projection {
            Some(projection) => projection,
            None => self.hidden,
        }
    }

    #[must_use]
    pub const fn layer_input(&self, layer: i64) -> i64 {
        if layer == 0 {
            self.input
        } else {
            self.output() * self.directions()
        }
    }
}

#[derive(Debug, Clone)]
pub struct RecurrentDirection {
    pub input: Linear,
    pub hidden: Linear,
    pub projection: Option<Linear>,
}

#[derive(Debug, Clone)]
pub struct RecurrentLayer {
    pub forward: RecurrentDirection,
    pub reverse: Option<RecurrentDirection>,
}

#[derive(Debug, Clone)]
pub struct RecurrentState {
    pub hidden: Tensor,
    pub cell: Option<Tensor>,
}

#[derive(Debug, Clone)]
pub struct Recurrent {
    config: RecurrentConfig,
    layers: Vec<RecurrentLayer>,
}

type CellState = (Tensor, Option<Tensor>);

impl Recurrent {
    pub fn create(config: RecurrentConfig, layers: Vec<RecurrentLayer>) -> Result<Self> {
        let count = usize::try_from(config.layers)
            .map_err(|_| Error::out_of_range("recurrent layer count"))?;
        if count == 0 || layers.len() != count {
            return Err(Error::invalid_argument("a recurrent layer's layers"));
        }

        if config.projection.is_some() && config.cell != RecurrentCell::Lstm {
            return Err(Error::invalid_argument(
                "a projection on a recurrent layer other than an LSTM",
            ));
        }

        for layer in &layers {
            if layer.reverse.is_some() != config.bidirectional {
                return Err(Error::invalid_argument(
                    "a recurrent layer's directions and its config",
                ));
            }

            for direction in iter_directions(layer) {
                if direction.projection.is_some() != config.projection.is_some() {
                    return Err(Error::invalid_argument(
                        "a recurrent layer's projections and its config",
                    ));
                }
            }
        }
        Ok(Self { config, layers })
    }

    #[must_use]
    pub const fn config(&self) -> &RecurrentConfig {
        &self.config
    }

    #[must_use]
    pub fn layers(&self) -> &[RecurrentLayer] {
        &self.layers
    }

    pub fn zeros(&self, batch: i64, device: Device, data_type: DataType) -> Result<RecurrentState> {
        let config = &self.config;
        let count = config.layers * config.directions();
        let zeros = |width: i64| {
            capture::literal(device, |device| {
                Tensor::zeroed_on(device, data_type, Shape::create([count, batch, width])?)
            })
        };
        Ok(RecurrentState {
            hidden: zeros(config.output())?,
            cell: config
                .cell
                .has_cell_state()
                .then(|| zeros(config.hidden))
                .transpose()?,
        })
    }

    pub fn step(
        &self,
        input: &Tensor,
        state: &RecurrentState,
        model: &ModelState,
    ) -> Result<(Tensor, RecurrentState)> {
        if self.config.bidirectional {
            return Err(Error::invalid_argument(
                "a single step of a bidirectional recurrent layer",
            ));
        }

        let mut input = input.clone();
        let mut hidden = Vec::with_capacity(self.layers.len());
        let mut cells = Vec::with_capacity(self.layers.len());
        for (index, layer) in self.layers.iter().enumerate() {
            let index = i64::try_from(index).map_err(|_| Error::out_of_range("layer"))?;
            let previous = layer_state(state, index)?;
            let gates = input.apply(&layer.forward.input, model)?;
            let (next, cell) = self.cell(&gates, &layer.forward, previous, model)?;
            input.clone_from(&next);
            hidden.push(next);
            cells.extend(cell);
        }
        Ok((input, stacked_state(&hidden, &cells)?))
    }

    pub fn forward_sequence(
        &self,
        input: &Tensor,
        state: Option<&RecurrentState>,
        lengths: Option<&Tensor>,
        model: &ModelState,
    ) -> Result<(Tensor, RecurrentState)> {
        let &[batch, steps, _] = input.shape().dimensions() else {
            return Err(Error::invalid_argument(
                "a recurrent layer reads sequences, [batch, steps, features]",
            ));
        };
        let zeros;
        let state = if let Some(state) = state {
            state
        } else {
            zeros = self.zeros(batch, input.device(), input.data_type())?;
            &zeros
        };
        let masks = lengths
            .map(|lengths| step_masks(lengths, steps))
            .transpose()?;
        let directions = self.config.directions();
        let mut sequence = input.clone();
        let mut hidden = Vec::new();
        let mut cells = Vec::new();
        for (index, layer) in self.layers.iter().enumerate() {
            let index = i64::try_from(index).map_err(|_| Error::out_of_range("layer"))?;
            let mut outputs = Vec::with_capacity(2);
            for (offset, (direction, reverse)) in
                iter_directions(layer).zip([false, true]).enumerate()
            {
                let offset = i64::try_from(offset).map_err(|_| Error::out_of_range("direction"))?;
                let previous = layer_state(state, index * directions + offset)?;
                let (output, (last, cell)) = self.run_direction(
                    &sequence,
                    direction,
                    reverse,
                    previous,
                    masks.as_deref(),
                    model,
                )?;
                outputs.push(output);
                hidden.push(last);
                cells.extend(cell);
            }

            sequence = if outputs.len() == 1 {
                outputs.remove(0)
            } else {
                Tensor::concat(&outputs, -1)?
            };
        }
        Ok((sequence, stacked_state(&hidden, &cells)?))
    }

    fn run_direction(
        &self,
        sequence: &Tensor,
        direction: &RecurrentDirection,
        reverse: bool,
        mut state: CellState,
        masks: Option<&[Tensor]>,
        model: &ModelState,
    ) -> Result<(Tensor, CellState)> {
        let steps = sequence.shape().dimensions()[1];
        let projected = sequence.apply(&direction.input, model)?;
        let order: Vec<i64> = if reverse {
            (0..steps).rev().collect()
        } else {
            (0..steps).collect()
        };
        let mut outputs = vec![None; usize::try_from(steps).unwrap_or(0)];
        let zero = capture::literal(sequence.device(), |device| {
            Scalar::Float(0.0).tensor(device, projected.data_type())
        })?;
        for step in order {
            let index = usize::try_from(step).map_err(|_| Error::out_of_range("step"))?;
            let gates = projected.select(1, step)?;
            let (hidden, cell) = self.cell(&gates, direction, state.clone(), model)?;
            let (hidden, cell, output) = match masks {
                Some(masks) => {
                    let mask = &masks[index];
                    let kept = Tensor::where_select(mask, &hidden, &state.0)?;
                    let cell = match (cell, &state.1) {
                        (Some(cell), Some(previous)) => {
                            Some(Tensor::where_select(mask, &cell, previous)?)
                        }
                        (cell, _) => cell,
                    };
                    let output = Tensor::where_select(mask, &hidden, &zero)?;
                    (kept, cell, output)
                }
                None => (hidden.clone(), cell, hidden),
            };
            outputs[index] = Some(output);
            state = (hidden, cell);
        }

        let outputs = outputs
            .into_iter()
            .map(|output| output.ok_or_else(|| Error::missing_output("a recurrent step")))
            .collect::<Result<Vec<_>>>()?;
        Ok((Tensor::stack(&outputs, 1)?, state))
    }

    fn cell(
        &self,
        gates: &Tensor,
        direction: &RecurrentDirection,
        (hidden, cell): CellState,
        model: &ModelState,
    ) -> Result<CellState> {
        let recurrent = hidden.apply(&direction.hidden, model)?;
        let (hidden, cell) =
            cell_step(self.config.cell, gates, &recurrent, &hidden, cell.as_ref())?;
        let hidden = match &direction.projection {
            Some(projection) => hidden.apply(projection, model)?,
            None => hidden,
        };
        Ok((hidden, cell))
    }
}

impl Layer for Recurrent {
    fn forward(&self, input: &Tensor, state: &ModelState) -> Result<Tensor> {
        Ok(self.forward_sequence(input, None, None, state)?.0)
    }
}

fn iter_directions(layer: &RecurrentLayer) -> impl Iterator<Item = &RecurrentDirection> {
    iter::once(&layer.forward).chain(layer.reverse.as_ref())
}

fn layer_state(state: &RecurrentState, index: i64) -> Result<CellState> {
    Ok((
        state.hidden.select(0, index)?,
        state
            .cell
            .as_ref()
            .map(|cell| cell.select(0, index))
            .transpose()?,
    ))
}

fn stacked_state(hidden: &[Tensor], cells: &[Tensor]) -> Result<RecurrentState> {
    Ok(RecurrentState {
        hidden: Tensor::stack(hidden, 0)?,
        cell: if cells.is_empty() {
            None
        } else {
            Some(Tensor::stack(cells, 0)?)
        },
    })
}

pub fn step_masks(lengths: &Tensor, steps: i64) -> Result<Vec<Tensor>> {
    let lengths = lengths.cast(DataType::I64)?.unsqueeze(1)?;
    (0..steps)
        .map(|step| {
            let step = capture::literal(lengths.device(), |device| {
                Tensor::from_vec(device, Shape::create([1, 1])?, vec![step])
            })?;
            lengths.greater(&step)
        })
        .collect()
}

pub fn cell_step(
    cell: RecurrentCell,
    gates: &Tensor,
    recurrent: &Tensor,
    hidden: &Tensor,
    previous: Option<&Tensor>,
) -> Result<(Tensor, Option<Tensor>)> {
    let width = gates.shape().dimensions().last().copied().unwrap_or(0) / cell.gates();
    let gate = |tensor: &Tensor, index: i64| tensor.narrow(-1, index * width, width);
    match cell {
        RecurrentCell::Tanh => Ok((gates.add(recurrent)?.tanh()?, None)),
        RecurrentCell::Relu => Ok((gates.add(recurrent)?.relu()?, None)),
        RecurrentCell::Lstm => {
            let previous = previous.ok_or_else(|| Error::missing_input("an LSTM's cell state"))?;
            let summed = gates.add(recurrent)?;
            let input = gate(&summed, 0)?.sigmoid()?;
            let forget = gate(&summed, 1)?.sigmoid()?;
            let candidate = gate(&summed, 2)?.tanh()?;
            let output = gate(&summed, 3)?.sigmoid()?;
            let cell = forget.mul(previous)?.add(&input.mul(&candidate)?)?;
            let hidden = output.mul(&cell.tanh()?)?;
            Ok((hidden, Some(cell)))
        }
        RecurrentCell::Gru => {
            let reset = gate(gates, 0)?.add(&gate(recurrent, 0)?)?.sigmoid()?;
            let update = gate(gates, 1)?.add(&gate(recurrent, 1)?)?.sigmoid()?;
            let new = gate(gates, 2)?
                .add(&reset.mul(&gate(recurrent, 2)?)?)?
                .tanh()?;
            // `(1 - z) n + z h`, as `n + z (h - n)`.
            let hidden = new.add(&update.mul(&hidden.sub(&new)?)?)?;
            Ok((hidden, None))
        }
    }
}
