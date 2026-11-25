// ONNX's recurrent operations, `RNN`, `LSTM`, and `GRU`, unrolled over the steps the input's shape fixes, each step the cell `nn::cell_step` computes.
// Their gates are reordered from ONNX's orders, `iofc` for LSTM and `zrh` for GRU, into PyTorch's, which the cell reads.
// They refuse what no exporter writes today: `clip`, activation parameters, activations other than the defaults (or ReLU for `RNN`), LSTM's coupled input and forget gates, and peepholes.

use singe_onnx::Node;

use crate::{
    error::{Error as SingeError, Result as SingeResult},
    graph::{capture, ir::ValueId},
    nn::{RecurrentCell, cell_step, step_masks},
    onnx::{
        attributes::Attributes,
        error::{Error, Result},
        importer::{ImportedValue, Importer},
        ops::{expect_input_range, optional_input, register_node_outputs},
    },
    shape::Shape,
    tensor::Tensor,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxRecurrence {
    Rnn,
    Lstm,
    Gru,
}

impl OnnxRecurrence {
    const fn operator(self) -> &'static str {
        match self {
            Self::Rnn => "RNN",
            Self::Lstm => "LSTM",
            Self::Gru => "GRU",
        }
    }

    const fn gates(self) -> i64 {
        match self {
            Self::Rnn => 1,
            Self::Lstm => 4,
            Self::Gru => 3,
        }
    }

    const fn pytorch_order(self) -> &'static [i64] {
        match self {
            Self::Rnn => &[0],
            // ONNX's input, output, forget, cell; PyTorch's input, forget, cell, output.
            Self::Lstm => &[0, 2, 3, 1],
            // ONNX's update, reset, hidden; PyTorch's reset, update, new.
            Self::Gru => &[1, 0, 2],
        }
    }

    const fn default_activations(self) -> &'static [&'static str] {
        match self {
            Self::Rnn => &["Tanh"],
            Self::Lstm => &["Sigmoid", "Tanh", "Tanh"],
            Self::Gru => &["Sigmoid", "Tanh"],
        }
    }

    const fn max_inputs(self) -> usize {
        match self {
            Self::Lstm => 8,
            Self::Rnn | Self::Gru => 6,
        }
    }

    const fn max_outputs(self) -> usize {
        match self {
            Self::Lstm => 3,
            Self::Rnn | Self::Gru => 2,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Recurrence {
    kind: OnnxRecurrence,
    cell: RecurrentCell,
    directions: &'static [bool],
    batch_first: bool,
    linear_before_reset: bool,
    hidden: i64,
}

#[derive(Debug, Clone, Copy, Default)]
struct OptionalInputs {
    bias: Option<usize>,
    lengths: Option<usize>,
    hidden: Option<usize>,
    cell: Option<usize>,
}

pub fn lower_recurrent(importer: &mut Importer, node: &Node, kind: OnnxRecurrence) -> Result<()> {
    expect_input_range(importer, node, 3, kind.max_inputs())?;
    if node.outputs.is_empty() || node.outputs.len() > kind.max_outputs() {
        return Err(refuse(importer, node, kind, "an unexpected output count"));
    }

    let attrs = Attributes::create(node)?;
    let recurrence = recurrence(importer, node, kind, &attrs)?;
    let input = importer.required_input(node, 0)?;
    let weights = importer.required_input(node, 1)?;
    let recurrent = importer.required_input(node, 2)?;
    let directions = i64::try_from(recurrence.directions.len()).unwrap_or(1);
    let gates = kind.gates() * recurrence.hidden;
    let &[weight_directions, weight_rows, _] = weights.shape.dimensions() else {
        return Err(refuse(
            importer,
            node,
            kind,
            "weights other than [directions, gates, input]",
        ));
    };
    if weight_directions != directions
        || weight_rows != gates
        || recurrent.shape.dimensions() != [directions, gates, recurrence.hidden]
        || input.shape.rank() != 3
    {
        return Err(refuse(
            importer,
            node,
            kind,
            "inputs whose shapes disagree with the direction and hidden size",
        ));
    }

    if kind == OnnxRecurrence::Lstm && optional_input(importer, node, 7)?.is_some() {
        return Err(refuse(importer, node, kind, "peepholes"));
    }

    let mut operands = vec![input.node, weights.node, recurrent.node];
    let mut optional = OptionalInputs::default();
    let mut take = |index: usize, slot: &mut Option<usize>| -> Result<()> {
        if let Some(value) = optional_input(importer, node, index)? {
            *slot = Some(operands.len());
            operands.push(value.node);
        }
        Ok(())
    };
    take(3, &mut optional.bias)?;
    take(4, &mut optional.lengths)?;
    take(5, &mut optional.hidden)?;
    if kind == OnnxRecurrence::Lstm {
        take(6, &mut optional.cell)?;
    }

    if !importer.have_fixed_shapes(&operands)? {
        return Err(refuse(importer, node, kind, "inputs with run-time axes"));
    }

    let outputs = node.outputs.len();
    let device = importer.options().device;
    let ids = importer
        .graph()
        .append_traced(&operands, device, |inputs| {
            unrolled(&recurrence, inputs, optional, outputs)
        })?;
    let values = ids
        .into_iter()
        .map(|id| imported(importer, id))
        .collect::<Result<Vec<_>>>()?;
    register_node_outputs(importer, node, values)
}

fn refuse(importer: &Importer, node: &Node, kind: OnnxRecurrence, reason: &str) -> Error {
    Error::unsupported_operator(
        node.domain.as_deref(),
        kind.operator(),
        importer.opset_version(node.domain.as_deref()),
        reason,
    )
}

fn imported(importer: &mut Importer, id: ValueId) -> Result<ImportedValue> {
    let node = importer.graph().required_node(id)?;
    let (data_type, shape) = (node.output_data_type(), node.output_shape().clone());
    Ok(ImportedValue::new(id, data_type, shape))
}

fn recurrence(
    importer: &Importer,
    node: &Node,
    kind: OnnxRecurrence,
    attrs: &Attributes<'_>,
) -> Result<Recurrence> {
    let refused = |reason: &str| Err(refuse(importer, node, kind, reason));
    if attrs.value("clip").is_some() {
        return refused("clipping");
    }

    if attrs.value("activation_alpha").is_some() || attrs.value("activation_beta").is_some() {
        return refused("activation parameters");
    }

    if attrs.int("input_forget", 0)? != 0 {
        return refused("coupled input and forget gates");
    }

    let directions: &'static [bool] = match attrs.string("direction", "forward")?.as_str() {
        "forward" => &[false],
        "reverse" => &[true],
        "bidirectional" => &[false, true],
        _ => return refused("an unknown direction"),
    };
    let mut cell = match kind {
        OnnxRecurrence::Rnn => RecurrentCell::Tanh,
        OnnxRecurrence::Lstm => RecurrentCell::Lstm,
        OnnxRecurrence::Gru => RecurrentCell::Gru,
    };
    if let Some(activations) = attrs.strings("activations")? {
        let defaults = kind.default_activations();
        let named = |index: usize, name: &str| activations[index].eq_ignore_ascii_case(name);
        if activations.len() != defaults.len() * directions.len() {
            return refused("activations other than one set per direction");
        }

        let relu =
            kind == OnnxRecurrence::Rnn && (0..activations.len()).all(|index| named(index, "Relu"));
        if relu {
            cell = RecurrentCell::Relu;
        } else if !(0..activations.len())
            .all(|index| named(index, defaults[index % defaults.len()]))
        {
            return refused("activations other than the defaults");
        }
    }

    let Some(&hidden) = importer.required_input(node, 2)?.shape.dimensions().get(2) else {
        return refused("recurrent weights other than [directions, gates, hidden]");
    };
    if attrs.int("hidden_size", hidden)? != hidden {
        return refused("a hidden size that disagrees with the recurrent weights");
    }

    Ok(Recurrence {
        kind,
        cell,
        directions,
        batch_first: match attrs.int("layout", 0)? {
            0 => false,
            1 => true,
            _ => return Err(refuse(importer, node, kind, "a layout other than 0 or 1")),
        },
        linear_before_reset: attrs.int("linear_before_reset", 0)? != 0,
        hidden,
    })
}

fn unrolled(
    recurrence: &Recurrence,
    inputs: &[&Tensor],
    optional: OptionalInputs,
    outputs: usize,
) -> SingeResult<Vec<Tensor>> {
    let hidden_width = recurrence.hidden;
    let batch_first = |tensor: &Tensor| {
        if recurrence.batch_first {
            tensor.transpose(0, 1)
        } else {
            Ok(tensor.clone())
        }
    };
    // `[T, B, I]`, and the initial states `[D, B, H]`.
    let input = batch_first(inputs[0])?;
    let initial_hidden = optional
        .hidden
        .map(|index| batch_first(inputs[index]))
        .transpose()?;
    let initial_cell = optional
        .cell
        .map(|index| batch_first(inputs[index]))
        .transpose()?;
    let &[steps, batch, width] = input.shape().dimensions() else {
        return Err(SingeError::invalid_argument(
            "a recurrent input other than [steps, batch, features]",
        ));
    };
    let masks = optional
        .lengths
        .map(|index| step_masks(inputs[index], steps))
        .transpose()?;
    let flat = input.reshape([steps * batch, width])?;
    let gates = recurrence.kind.gates() * hidden_width;
    let reorder = |tensor: &Tensor| -> SingeResult<Tensor> {
        let parts = recurrence
            .kind
            .pytorch_order()
            .iter()
            .map(|&gate| tensor.narrow(0, gate * hidden_width, hidden_width))
            .collect::<SingeResult<Vec<_>>>()?;
        Tensor::concat(&parts, 0)
    };

    let mut sequences = Vec::new();
    let mut finals = Vec::new();
    let mut cells = Vec::new();
    for (direction, &reverse) in recurrence.directions.iter().enumerate() {
        let direction = i64::try_from(direction).unwrap_or(0);
        let input_weights = reorder(&inputs[1].select(0, direction)?)?;
        let recurrent_weights = reorder(&inputs[2].select(0, direction)?)?.transpose(0, 1)?;
        let biases = optional
            .bias
            .map(|index| -> SingeResult<_> {
                let bias = inputs[index].select(0, direction)?;
                Ok((
                    reorder(&bias.narrow(0, 0, gates)?)?,
                    reorder(&bias.narrow(0, gates, gates)?)?,
                ))
            })
            .transpose()?;
        let mut projected = flat.matmul(&input_weights.transpose(0, 1)?)?;
        if let Some((input_bias, _)) = &biases {
            projected = projected.add(input_bias)?;
        }

        let projected = projected.reshape([steps, batch, gates])?;
        let zeros = capture::literal(projected.device(), |device| {
            Tensor::zeroed_on(
                device,
                projected.data_type(),
                Shape::create([batch, hidden_width])?,
            )
        })?;
        let mut hidden = match &initial_hidden {
            Some(initial) => initial.select(0, direction)?,
            None => zeros.clone(),
        };
        let mut cell = match (&initial_cell, recurrence.cell) {
            (Some(initial), _) => Some(initial.select(0, direction)?),
            (None, RecurrentCell::Lstm) => Some(zeros.clone()),
            (None, _) => None,
        };
        let mut sequence = vec![None; usize::try_from(steps).unwrap_or(0)];
        let order: Box<dyn Iterator<Item = i64>> = if reverse {
            Box::new((0..steps).rev())
        } else {
            Box::new(0..steps)
        };
        for step in order {
            let step_gates = projected.select(0, step)?;
            let mut recurrent = hidden.matmul(&recurrent_weights)?;
            if let Some((_, recurrent_bias)) = &biases {
                recurrent = recurrent.add(recurrent_bias)?;
            }

            let (next, next_cell) =
                if recurrence.kind == OnnxRecurrence::Gru && !recurrence.linear_before_reset {
                    gru_reset_before(
                        &step_gates,
                        &recurrent,
                        &hidden,
                        &recurrent_weights,
                        biases.as_ref().map(|(_, bias)| bias),
                        hidden_width,
                    )?
                } else {
                    cell_step(
                        recurrence.cell,
                        &step_gates,
                        &recurrent,
                        &hidden,
                        cell.as_ref(),
                    )?
                };
            let output = if let Some(masks) = &masks {
                let mask = &masks[usize::try_from(step).unwrap_or(0)];
                hidden = Tensor::where_select(mask, &next, &hidden)?;
                cell = if let (Some(next), Some(previous)) = (&next_cell, &cell) {
                    Some(Tensor::where_select(mask, next, previous)?)
                } else {
                    next_cell
                };
                Tensor::where_select(mask, &next, &zeros)?
            } else {
                hidden = next.clone();
                cell = next_cell;
                next
            };
            sequence[usize::try_from(step).unwrap_or(0)] = Some(output);
        }

        let sequence = sequence.into_iter().flatten().collect::<Vec<_>>();
        sequences.push(Tensor::stack(&sequence, 0)?);
        finals.push(hidden);
        if let Some(cell) = cell {
            cells.push(cell);
        }
    }

    // `Y`, `[T, D, B, H]`, and the final states, `[D, B, H]`.
    let mut results = vec![Tensor::stack(&sequences, 1)?, Tensor::stack(&finals, 0)?];
    if recurrence.kind == OnnxRecurrence::Lstm {
        results.push(Tensor::stack(&cells, 0)?);
    }

    if recurrence.batch_first {
        results[0] = results[0].permute(&[2, 0, 1, 3])?;
        for result in &mut results[1..] {
            *result = result.transpose(0, 1)?;
        }
    }

    results.truncate(outputs);
    Ok(results)
}

fn gru_reset_before(
    gates: &Tensor,
    recurrent: &Tensor,
    hidden: &Tensor,
    recurrent_weights: &Tensor,
    recurrent_bias: Option<&Tensor>,
    width: i64,
) -> SingeResult<(Tensor, Option<Tensor>)> {
    let gate = |tensor: &Tensor, index: i64| tensor.narrow(-1, index * width, width);
    let reset = gate(gates, 0)?.add(&gate(recurrent, 0)?)?.sigmoid()?;
    let update = gate(gates, 1)?.add(&gate(recurrent, 1)?)?.sigmoid()?;
    let mut reset_projection =
        reset
            .mul(hidden)?
            .matmul(&recurrent_weights.narrow(1, 2 * width, width)?)?;
    if let Some(bias) = recurrent_bias {
        reset_projection = reset_projection.add(&bias.narrow(0, 2 * width, width)?)?;
    }

    let new = gate(gates, 2)?.add(&reset_projection)?.tanh()?;
    Ok((new.add(&update.mul(&hidden.sub(&new)?)?)?, None))
}
