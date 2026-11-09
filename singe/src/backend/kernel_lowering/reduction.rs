use std::collections::HashMap;

use singe_codegen::{
    ir::{AxisId, AxisKind, BinaryOp, Combiner, ElementType, Literal, NodeId, UnaryOp},
    schedule::{Schedule, element_schedule, reduces_contiguously, reduction_vector},
};

use crate::{
    backend::{
        kernel_lowering::{
            domain::{DomainKernel, LoweredKernel},
            element_type,
            epilogue::{EpilogueValue, EpilogueWriter, RowEpilogue, RowPoint},
            group::ElementGroup,
        },
        registry::{BatchNormalizationAttributes, LayerNormalizationAttributes, Operation},
        spec,
    },
    data_type::DataType,
    error::{Error, Result},
    layout::Layout,
    shape::normalize_dim_usize,
    value::ValueSpec,
};

// The column each epilogue node's operand reads, by node and operand position.
type OperandColumns = HashMap<(usize, usize), usize>;

// How a reduction folds its points: combining their values, or finding the position of the largest or smallest.
#[derive(Debug, Clone, Copy)]
enum Reducer {
    Combine(Combiner),
    Position { maximum: bool },
}

/// Lowers `sum`, `mean`, `max`, `min`, `product`, `argmax`, or `argmin` of `group`'s value over `dimensions`.
///
/// # Errors
///
/// Returns an error if the dimensions are invalid, the value is not floating-point, an input does not broadcast to the group's shape, or the reduction is empty for an operation without an identity.
pub fn reduction(
    operation: Operation,
    group: &ElementGroup<'_>,
    dimensions: &[i64],
    keepdim: bool,
    epilogue: Option<&RowEpilogue<'_>>,
) -> Result<LoweredKernel> {
    let data_type = group.data_type;
    if matches!(operation, Operation::Mean | Operation::LogSumExp) && !data_type.is_float() {
        return Err(Error::unsupported_data_type(
            "generated mean or log-sum-exp of integers",
        ));
    }

    let problem = spec::reduction_problem(&group.shape, dimensions, keepdim)?;
    let folding = checked_reducer(operation, group, &problem, epilogue)?;
    let output_type = match folding {
        Reducer::Combine(_) => data_type,
        Reducer::Position { .. } => DataType::I64,
    };
    let input_dimensions = group.shape.dimensions();
    let kinds = (0..input_dimensions.len())
        .map(|axis| {
            if problem.dimensions.contains(&axis) {
                AxisKind::Reduction
            } else {
                AxisKind::Parallel
            }
        })
        .collect::<Vec<_>>();
    // An epilogue whose last value has the input's shape stores it along the reduced axes too, as a row kernel stores its row.
    let whole = epilogue
        .and_then(|epilogue| epilogue.nodes.last())
        .is_some_and(|last| last.shape == group.shape);
    let output_strides = if whole {
        Layout::contiguous(input_dimensions)?.strides().to_vec()
    } else {
        contiguous_over(input_dimensions, &kinds, AxisKind::Parallel)
    };
    let mut strides = group_strides(group)?;
    let epilogue_columns = match epilogue {
        Some(epilogue) => {
            let (columns, positions) =
                epilogue_input_columns(epilogue, group, &problem, &kinds, keepdim, strides.len())?;
            strides.extend(columns);
            positions
        }
        None => HashMap::new(),
    };
    strides.push(output_strides);
    if group.stored {
        strides.push(Layout::contiguous(input_dimensions)?.strides().to_vec());
    }

    let mut builder = DomainKernel::create(
        &kernel_name(operation, group),
        input_dimensions,
        &kinds,
        &strides,
    )?;
    let value = builder.group_value(group)?;
    let reduced = builder.reduction_axes();
    let first_axis = reduced.first().copied();
    // Booleans reduce as bytes, whose maximum is the disjunction and minimum the conjunction.
    let reduced_value = if data_type == DataType::Bool {
        builder.convert(value, ElementType::U8)
    } else {
        value
    };
    // A scalar input reduces over no axes: its value is the result, and its position zero.
    let mut result = match (folding, reduced.is_empty()) {
        (Reducer::Combine(_), true) => reduced_value,
        (Reducer::Position { .. }, true) => {
            builder.kernel.literal(ElementType::I64, Literal::Int(0))
        }
        (Reducer::Combine(combiner), false) => {
            builder.kernel.reduce(combiner, reduced, reduced_value)
        }
        (Reducer::Position { maximum }, false) => {
            builder.kernel.arg_reduce(maximum, reduced, reduced_value)
        }
    };
    if data_type == DataType::Bool {
        result = builder.convert(result, ElementType::Bool);
    }

    if operation == Operation::Mean {
        let count = builder.count(AxisKind::Reduction);
        let count = builder.scalar(
            element_type(data_type)?.compute(),
            count,
            "reduced element count",
        );
        result = builder.kernel.binary(BinaryOp::Div, result, count);
    }

    let mut column = group.inputs.len();
    if let Some(epilogue) = epilogue {
        for (slot, &(input_type, _)) in epilogue.inputs.iter().enumerate() {
            builder.shared_input(element_type(input_type)?, column + slot);
        }

        let axis = first_axis.ok_or_else(|| {
            Error::unsupported_operation("cuda generated epilogue of a reduction over no axes")
        })?;
        let mut writer =
            EpilogueWriter::new(epilogue, data_type, axis, epilogue_columns, HashMap::new());
        let row_value = result;
        result = writer.value(
            &mut builder,
            epilogue.nodes.len() - 1,
            RowPoint::Own,
            &mut |_, _| Ok(row_value),
        )?;
        column += epilogue.inputs.len();
    }

    let output = builder.output(element_type(output_type)?, column);
    builder.kernel.store(output, result);
    builder.store_group(group, value, strides.len() - 1)?;
    let schedule = reduction_schedule(&builder, whole);
    let shape = if whole {
        group.shape.clone()
    } else {
        problem.output_shape
    };
    Ok(builder.finish(ValueSpec::new(output_type, shape), schedule))
}

// Checks that `operation` can reduce `group`, and returns how it folds.
fn checked_reducer(
    operation: Operation,
    group: &ElementGroup<'_>,
    problem: &spec::ReductionProblem,
    epilogue: Option<&RowEpilogue<'_>>,
) -> Result<Reducer> {
    // An argument reduction yields the `I64` position of the extreme value, and has none over no points.
    let folding = match operation {
        Operation::Sum | Operation::Mean => Reducer::Combine(Combiner::Sum),
        Operation::Product => Reducer::Combine(Combiner::Product),
        Operation::Max => Reducer::Combine(Combiner::Max),
        Operation::Min => Reducer::Combine(Combiner::Min),
        Operation::LogSumExp => Reducer::Combine(Combiner::LogSumExp),
        Operation::ArgMax => Reducer::Position { maximum: true },
        Operation::ArgMin => Reducer::Position { maximum: false },
        _ => return Err(Error::unsupported_operation("cuda generated reduction")),
    };
    if let Reducer::Position { .. } = folding {
        if epilogue.is_some() || group.data_type == DataType::Bool {
            return Err(Error::unsupported_operation(
                "cuda generated argument reduction of booleans or with an epilogue",
            ));
        }
        if problem
            .dimensions
            .iter()
            .any(|&axis| group.shape.dimensions()[axis] == 0)
        {
            return Err(Error::invalid_argument(
                "arg reduction over an empty dimension",
            ));
        }
    } else {
        spec::empty_reduction_fill(&group.shape, problem, operation)?;
    }
    Ok(folding)
}

// An epilogue's inputs take columns after the group's: broadcast to the reduced shape, along the parallel axes alone, and broadcast to the input's shape, along every axis.
//
// Returns each input's column, and for each epilogue node's operand the column it reads, counted from `first`.
fn epilogue_input_columns(
    epilogue: &RowEpilogue<'_>,
    group: &ElementGroup<'_>,
    problem: &spec::ReductionProblem,
    kinds: &[AxisKind],
    keepdim: bool,
    first: usize,
) -> Result<(Vec<Vec<i64>>, OperandColumns)> {
    let mut positions = HashMap::new();
    let mut slots: Vec<Option<Vec<i64>>> = vec![None; epilogue.inputs.len()];
    for (index, node) in epilogue.nodes.iter().enumerate() {
        let EpilogueValue::Step { operands, .. } = &node.value else {
            continue;
        };
        for (position, &operand) in operands.iter().enumerate() {
            let EpilogueValue::Input(slot) = epilogue.nodes[operand].value else {
                continue;
            };
            let column = if node.shape == group.shape {
                epilogue.inputs[slot]
                    .1
                    .expand(&group.shape)?
                    .strides()
                    .to_vec()
            } else {
                let layout = epilogue.inputs[slot].1.expand(&problem.output_shape)?;
                let mut output_axis = 0;
                kinds
                    .iter()
                    .map(|&kind| {
                        let reduced = kind == AxisKind::Reduction;
                        let stride = if reduced {
                            0
                        } else {
                            layout.strides()[output_axis]
                        };
                        if !reduced || keepdim {
                            output_axis += 1;
                        }
                        stride
                    })
                    .collect::<Vec<_>>()
            };
            match &slots[slot] {
                None => slots[slot] = Some(column),
                Some(known) if *known == column => {}
                Some(_) => {
                    return Err(Error::unsupported_operation(
                        "cuda generated reduction epilogue input read in two shapes",
                    ));
                }
            }

            positions.insert((index, position), first + slot);
        }
    }

    let columns = slots
        .into_iter()
        .map(|column| column.ok_or_else(|| Error::missing_input("column of an epilogue input")))
        .collect::<Result<Vec<_>>>()?;
    Ok((columns, positions))
}

// A reduction over outer axes of at least 128 points runs column blocks, whose consecutive threads load consecutive points; normalizations, softmax, and reductions that store along their reduced axes keep row schedules, which can keep the row in registers between its loops.
// Each block takes enough points for about 64 blocks, from 8 to 128, and splits the reduction among the rest of its 256 threads, as the tuner measured best on `sum` over outer axes of `[4096, 4096]`, `[16384, 128]`, and `[8, 1024, 64]`.
fn reduction_schedule(builder: &DomainKernel, whole: bool) -> Schedule {
    if !whole
        && !reduces_contiguously(&builder.kernel)
        && builder.count(AxisKind::Reduction) >= 128.0
    {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "the point count is positive and clamped below"
        )]
        let points = ((builder.count(AxisKind::Parallel) / 64.0).ceil().max(1.0) as u32)
            .next_power_of_two()
            .clamp(8, 128);
        Schedule::Columns {
            threads: 256,
            splits: (256 / points).clamp(2, 32),
        }
    } else {
        row_schedule(builder)
    }
}

/// Lowers `softmax` or `log_softmax` of `group`'s value along `dim`, as one row kernel with a maximum and a sum, storing `epilogue`'s last value instead of the result when there is one.
///
/// The epilogue's inputs come after the group's among the kernel's inputs.
///
/// # Errors
///
/// Returns an error if `dim` is invalid, the value is not floating-point, an input does not broadcast to the group's shape, or an epilogue runs along an axis other than the last or its values do not fit the row.
pub fn softmax(
    operation: Operation,
    group: &ElementGroup<'_>,
    dim: i64,
    epilogue: Option<&RowEpilogue<'_>>,
) -> Result<LoweredKernel> {
    let data_type = group.data_type;
    if !data_type.is_float() {
        return Err(Error::unsupported_data_type("cuda generated softmax"));
    }

    let shape = &group.shape;
    let mut strides = group_strides(group)?;
    // A scalar is one row of one element.
    let (dimensions, axis) = if shape.rank() == 0 {
        for strides in &mut strides {
            *strides = vec![0];
        }
        (vec![1], 0)
    } else {
        (
            shape.dimensions().to_vec(),
            spec::softmax_problem(shape, dim)?.axis,
        )
    };
    if epilogue.is_some() && axis + 1 != dimensions.len() {
        return Err(Error::unsupported_operation(
            "cuda generated softmax epilogue along an inner axis",
        ));
    }

    let kinds = (0..dimensions.len())
        .map(|index| {
            if index == axis {
                AxisKind::Reduction
            } else {
                AxisKind::Parallel
            }
        })
        .collect::<Vec<_>>();
    // Every column's stride along the row, for reading it at other points.
    let row_strides = strides
        .iter()
        .map(|column| column.get(axis).copied().unwrap_or(0))
        .collect::<Vec<_>>();
    let contiguous = Layout::contiguous(dimensions.as_slice())?
        .strides()
        .to_vec();
    let columns = epilogue
        .map(|epilogue| EpilogueColumns::create(epilogue, &dimensions, &mut strides))
        .transpose()?;
    let output = columns
        .as_ref()
        .map_or_else(|| contiguous.clone(), |columns| columns.output.clone());
    strides.push(output);
    if group.stored {
        strides.push(contiguous);
    }

    let mut builder = DomainKernel::create(
        &kernel_name(operation, group),
        &dimensions,
        &kinds,
        &strides,
    )?;
    let element = element_type(data_type)?;
    let value = builder.group_value(group)?;
    let reduced = builder.reduction_axes();
    let row_axis = *reduced
        .first()
        .ok_or_else(|| Error::missing_node("softmax"))?;
    let maximum = builder.kernel.reduce(Combiner::Max, reduced.clone(), value);
    let shifted = builder.kernel.binary(BinaryOp::Sub, value, maximum);
    let exponential = builder.kernel.unary(UnaryOp::Exp, shifted);
    let sum = builder.kernel.reduce(Combiner::Sum, reduced, exponential);
    let log = (operation == Operation::LogSoftmax).then(|| builder.kernel.unary(UnaryOp::Log, sum));
    let finish = |builder: &mut DomainKernel, shifted: NodeId, exponential: NodeId| match log {
        Some(log) => builder.kernel.binary(BinaryOp::Sub, shifted, log),
        None => builder.kernel.binary(BinaryOp::Div, exponential, sum),
    };
    let mut result = finish(&mut builder, shifted, exponential);
    let mut column = group.inputs.len();
    if let Some(epilogue) = epilogue {
        // The result at another point of the row, from the group's input there and the row's reductions.
        let own = result;
        let mut row = |builder: &mut DomainKernel, point: RowPoint| -> Result<NodeId> {
            let RowPoint::At(index) = point else {
                return Ok(own);
            };
            let value = builder.group_value_at(group, row_axis, index, &row_strides)?;
            let shifted = builder.kernel.binary(BinaryOp::Sub, value, maximum);
            let exponential = builder.kernel.unary(UnaryOp::Exp, shifted);
            Ok(finish(builder, shifted, exponential))
        };
        let columns = columns.ok_or_else(|| Error::missing_node("softmax"))?;
        result = columns.write(
            &mut builder,
            epilogue,
            data_type,
            row_axis,
            column,
            &mut row,
        )?;
        column += epilogue.inputs.len();
    }

    let output = builder.output(element, column);
    builder.kernel.store(output, result);
    builder.store_group(group, value, column + 1)?;
    let schedule = row_schedule(&builder);
    let shape = epilogue
        .and_then(|epilogue| epilogue.nodes.last())
        .map_or_else(|| shape.clone(), |last| last.shape.clone());
    Ok(builder.finish(ValueSpec::new(data_type, shape), schedule))
}

/// Lowers `layer_norm` or `rms_norm` of `inputs`, the input followed by its optional weight and bias, over the trailing normalized dimensions.
///
/// Layer normalization computes the mean and then the variance in two passes over each row; RMS normalization computes the mean square.
///
/// # Errors
///
/// Returns an error if the attributes or parameters do not match the input, or the types are not floating-point.
pub fn normalization(
    operation: Operation,
    attributes: &LayerNormalizationAttributes,
    group: &ElementGroup<'_>,
    parameters: &[(DataType, &Layout)],
) -> Result<LoweredKernel> {
    normalization_kernel(operation, attributes, group, parameters, None)
}

/// Lowers a normalization as [`normalization`] does, storing `epilogue`'s last value instead of the normalized one.
///
/// The epilogue's inputs come after the parameters among the kernel's inputs.
///
/// # Errors
///
/// Returns an error if [`normalization`] would, the normalization covers more than one axis, or the epilogue's values do not fit the row.
pub fn normalization_with_epilogue(
    operation: Operation,
    attributes: &LayerNormalizationAttributes,
    group: &ElementGroup<'_>,
    parameters: &[(DataType, &Layout)],
    epilogue: &RowEpilogue<'_>,
) -> Result<LoweredKernel> {
    normalization_kernel(operation, attributes, group, parameters, Some(epilogue))
}

#[expect(
    clippy::too_many_lines,
    reason = "the normalization's reductions, its epilogue's columns, and the recomputation of its output at other points share the kernel's columns"
)]
fn normalization_kernel(
    operation: Operation,
    attributes: &LayerNormalizationAttributes,
    group: &ElementGroup<'_>,
    parameters: &[(DataType, &Layout)],
    epilogue: Option<&RowEpilogue<'_>>,
) -> Result<LoweredKernel> {
    let data_type = group.data_type;
    if !data_type.is_float() {
        return Err(Error::unsupported_data_type("cuda generated normalization"));
    }

    let problem = spec::normalization_problem(
        &group.shape,
        &attributes.normalized_shape,
        "cuda generated normalization",
    )?;
    let weight = attributes
        .has_weight
        .then(|| {
            parameters
                .first()
                .ok_or_else(|| Error::missing_input("normalization kernel"))
        })
        .transpose()?;
    let bias = attributes
        .has_bias
        .then(|| {
            parameters
                .get(usize::from(attributes.has_weight))
                .ok_or_else(|| Error::missing_input("normalization kernel"))
        })
        .transpose()?;
    spec::validate_normalization_parameters(
        &problem,
        weight.map(|(_, layout)| layout.shape()),
        bias.map(|(_, layout)| layout.shape()),
        "cuda generated normalization",
    )?;
    let dimensions = group.shape.dimensions();
    let normalized = attributes.normalized_shape.len();
    let leading = dimensions.len() - normalized;
    let kinds = (0..dimensions.len())
        .map(|index| {
            if index >= leading {
                AxisKind::Reduction
            } else {
                AxisKind::Parallel
            }
        })
        .collect::<Vec<_>>();
    // Parameters broadcast over the leading dimensions.
    let parameter_strides = |(_, layout): &(DataType, &Layout)| {
        let mut strides = vec![0; leading];
        strides.extend_from_slice(layout.strides());
        strides
    };
    let mut strides = group_strides(group)?;
    strides.extend(weight.iter().map(|parameter| parameter_strides(parameter)));
    strides.extend(bias.iter().map(|parameter| parameter_strides(parameter)));
    // Every column's stride along the row, for reading it at other points.
    let row_strides = strides
        .iter()
        .map(|column| column.last().copied().unwrap_or(0))
        .collect::<Vec<_>>();
    let contiguous = Layout::contiguous(dimensions)?.strides().to_vec();
    if epilogue.is_some() && normalized != 1 {
        return Err(Error::unsupported_operation(
            "cuda generated normalization epilogue over several axes",
        ));
    }

    let columns = epilogue
        .map(|epilogue| EpilogueColumns::create(epilogue, dimensions, &mut strides))
        .transpose()?;
    let output = columns
        .as_ref()
        .map_or_else(|| contiguous.clone(), |columns| columns.output.clone());
    strides.push(output);
    if group.stored {
        strides.push(contiguous);
    }

    let mut builder =
        DomainKernel::create(&kernel_name(operation, group), dimensions, &kinds, &strides)?;
    let element = element_type(data_type)?;
    let value = builder.group_value(group)?;
    let reduced = builder.reduction_axes();
    let width = builder.count(AxisKind::Reduction);
    let width = builder.scalar(element.compute(), width, "normalized width");
    let eps = builder.scalar(element.compute(), f64::from(attributes.eps), "epsilon");
    let axis = *reduced
        .first()
        .ok_or_else(|| Error::missing_node("normalization kernel"))?;
    let (centered, moment, mean) = if operation == Operation::LayerNorm {
        let sum = builder.kernel.reduce(Combiner::Sum, reduced.clone(), value);
        let mean = builder.kernel.binary(BinaryOp::Div, sum, width);
        let delta = builder.kernel.binary(BinaryOp::Sub, value, mean);
        let square = builder.kernel.binary(BinaryOp::Mul, delta, delta);
        (
            delta,
            builder.kernel.reduce(Combiner::Sum, reduced, square),
            Some(mean),
        )
    } else {
        let square = builder.kernel.binary(BinaryOp::Mul, value, value);
        (
            value,
            builder.kernel.reduce(Combiner::Sum, reduced, square),
            None,
        )
    };
    let variance = builder.kernel.binary(BinaryOp::Div, moment, width);
    let shifted = builder.kernel.binary(BinaryOp::Add, variance, eps);
    let scale = builder.kernel.unary(UnaryOp::Rsqrt, shifted);
    let mut result = builder.kernel.binary(BinaryOp::Mul, centered, scale);
    let mut column = group.inputs.len();
    for (parameter, op) in [(weight, BinaryOp::Mul), (bias, BinaryOp::Add)] {
        if let Some(&(parameter_type, _)) = parameter {
            let access = builder.input(element_type(parameter_type)?, column);
            let parameter = builder.kernel.load(access);
            let parameter = builder.convert(parameter, element.compute());
            result = builder.kernel.binary(op, result, parameter);
            column += 1;
        }
    }

    if let Some(epilogue) = epilogue {
        // The normalized value at another point of the row, from the group's input there and the row's reductions.
        let parameter_columns = (group.inputs.len()..column).collect::<Vec<_>>();
        let operations = [weight.map(|_| BinaryOp::Mul), bias.map(|_| BinaryOp::Add)]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let own = result;
        let mut row = |builder: &mut DomainKernel, point: RowPoint| -> Result<NodeId> {
            let RowPoint::At(index) = point else {
                return Ok(own);
            };
            let value = builder.group_value_at(group, axis, index, &row_strides)?;
            let centered = match mean {
                Some(mean) => builder.kernel.binary(BinaryOp::Sub, value, mean),
                None => value,
            };
            let mut normalized = builder.kernel.binary(BinaryOp::Mul, centered, scale);
            for (&parameter, &op) in parameter_columns.iter().zip(&operations) {
                let parameter_type = parameters[parameter - group.inputs.len()].0;
                let access = builder.input_at(
                    element_type(parameter_type)?,
                    parameter,
                    axis,
                    index,
                    row_strides[parameter],
                )?;
                let loaded = builder.kernel.load(access);
                let loaded = builder.convert(loaded, element.compute());
                normalized = builder.kernel.binary(op, normalized, loaded);
            }
            Ok(normalized)
        };
        let columns = columns.ok_or_else(|| Error::missing_node("normalization kernel"))?;
        result = columns.write(&mut builder, epilogue, data_type, axis, column, &mut row)?;
        column += epilogue.inputs.len();
    }

    let output = builder.output(element, column);
    builder.kernel.store(output, result);
    builder.store_group(group, value, column + 1)?;
    let schedule = row_schedule(&builder);
    let shape = epilogue
        .and_then(|epilogue| epilogue.nodes.last())
        .map_or(problem.output_shape, |last| last.shape.clone());
    Ok(builder.finish(ValueSpec::new(data_type, shape), schedule))
}

// The kernel columns an epilogue reads, and the strides its last value is stored with.
struct EpilogueColumns {
    // The column of each step's input operand, by step and operand position.
    columns: HashMap<(usize, usize), usize>,
    // The stride along the row of each of those columns.
    row_strides: HashMap<usize, i64>,
    output: Vec<i64>,
}

impl EpilogueColumns {
    // Appends a column to `strides` for each of the epilogue's inputs, in input order, so the kernel reads each input through one operand.
    fn create(
        epilogue: &RowEpilogue<'_>,
        dimensions: &[i64],
        strides: &mut Vec<Vec<i64>>,
    ) -> Result<Self> {
        let (mut columns, mut row_strides) = (HashMap::new(), HashMap::new());
        let orders = epilogue.orders(dimensions.len())?;
        // One column per input.
        let mut slots = vec![None; epilogue.inputs.len()];
        for (index, node) in epilogue.nodes.iter().enumerate() {
            let EpilogueValue::Step { operands, .. } = &node.value else {
                continue;
            };
            let order = orders[index]
                .as_ref()
                .ok_or_else(|| Error::missing_node("epilogue columns create"))?;
            for (position, &operand) in operands.iter().enumerate() {
                if let EpilogueValue::Input(slot) = epilogue.nodes[operand].value {
                    let layout = epilogue.inputs[slot].1.expand(&node.shape)?;
                    let column = RowEpilogue::domain_strides(&layout, order);
                    match &slots[slot] {
                        None => slots[slot] = Some(column),
                        Some(known) if *known == column => {}
                        Some(_) => {
                            return Err(Error::unsupported_operation(
                                "cuda generated row epilogue input read in two layouts",
                            ));
                        }
                    }

                    columns.insert((index, position), strides.len() + slot);
                }
            }
        }

        for column in slots {
            let column = column.ok_or_else(|| Error::missing_input("epilogue columns create"))?;
            row_strides.insert(strides.len(), column.last().copied().unwrap_or(0));
            strides.push(column);
        }

        let last = epilogue
            .nodes
            .last()
            .ok_or_else(|| Error::missing_node("epilogue columns create"))?;
        let order = orders[epilogue.nodes.len() - 1]
            .as_ref()
            .ok_or_else(|| Error::missing_node("epilogue columns create"))?;
        if last.shape.dimensions().len() != dimensions.len()
            || last.shape.dimensions().last() != dimensions.last()
        {
            return Err(Error::unsupported_operation(
                "cuda generated row epilogue of another row width",
            ));
        }

        let output =
            RowEpilogue::domain_strides(&Layout::contiguous(last.shape.dimensions())?, order);
        Ok(Self {
            columns,
            row_strides,
            output,
        })
    }

    // Builds the epilogue's last value at the kernel's own point, with its inputs as operands from `column` on, where `row` computes the row operation's output at a point.
    fn write(
        self,
        builder: &mut DomainKernel,
        epilogue: &RowEpilogue<'_>,
        data_type: DataType,
        axis: AxisId,
        column: usize,
        row: &mut dyn FnMut(&mut DomainKernel, RowPoint) -> Result<NodeId>,
    ) -> Result<NodeId> {
        // The epilogue's inputs become operands in input order before anything reads them.
        for (slot, &(input_type, _)) in epilogue.inputs.iter().enumerate() {
            builder.shared_input(element_type(input_type)?, column + slot);
        }

        let mut writer =
            EpilogueWriter::new(epilogue, data_type, axis, self.columns, self.row_strides);
        writer.value(builder, epilogue.nodes.len() - 1, RowPoint::Own, row)
    }
}

/// Returns `singe_` followed by the operation's name, after `fused_` and the names of the group's steps when it has any.
fn kernel_name(operation: Operation, group: &ElementGroup<'_>) -> String {
    let mut name = String::from("singe_");
    if !group.steps.is_empty() {
        name.push_str("fused_");
        for step in group.steps {
            name.push_str(step.operation.name());
            name.push('_');
        }
    }
    name + operation.name()
}

/// Returns the strides of each of `group`'s inputs broadcast to its shape.
fn group_strides(group: &ElementGroup<'_>) -> Result<Vec<Vec<i64>>> {
    group
        .inputs
        .iter()
        .map(|(_, layout)| Ok(layout.expand(&group.shape)?.strides().to_vec()))
        .collect()
}

/// Returns the strides of a contiguous buffer over the dimensions of `kind`, zero along the others.
fn contiguous_over(dimensions: &[i64], kinds: &[AxisKind], kind: AxisKind) -> Vec<i64> {
    let mut strides = vec![0; dimensions.len()];
    let mut stride = 1;
    for index in (0..dimensions.len()).rev() {
        if kinds[index] == kind {
            strides[index] = stride;
            stride *= dimensions[index];
        }
    }
    strides
}

/// Threads per row: a power of two from 32 to 1024 that covers `points`.
fn row_threads(points: f64) -> u32 {
    let mut threads = 32;
    while f64::from(threads) < points && threads < 1024 {
        threads *= 2;
    }
    threads
}

/// Walks rows of fewer than 32 elements in one thread, gives each row of up to 256 elements, or up to 512 among at least 1024 rows, some lanes of a warp, and splits wider rows across a block of 64 to 512 threads, each with whole vectors of up to 16 bytes, which the emitter keeps in registers for a static row; reductions whose loads all stride along the reduction take few-lane warps or blocks.
///
/// These are the shapes the autotuner chose on the decoder benchmark and on FlashInfer-Bench's RMS norm problems: many short rows share a warp with the widest vectors, few short rows take a whole warp each, wide rows take a thread per vector, and among at least 128 rows a thread per four vectors.
pub fn row_schedule(builder: &DomainKernel) -> Schedule {
    let width = builder.count(AxisKind::Reduction);
    let rows = builder.count(AxisKind::Parallel);
    let widest = reduction_vector(&builder.kernel, 16);
    // The widest vector that still gives each of `lanes` threads one.
    let vector = |lanes: u32| {
        let mut vector = 1;
        while vector < widest && f64::from(lanes * vector * 2) <= width {
            vector *= 2;
        }
        vector
    };
    if width < 32.0 {
        Schedule::Elements {
            threads: 128,
            vector: 1,
        }
    } else if !reduces_contiguously(&builder.kernel) {
        // Consecutive lanes of a reduction over outer axes read points the reduction strides apart, so few lanes share a point and consecutive groups read consecutive points of the parallel axes; measured best by the tuner on `sum` over the outer axes of `[64, 4096]` and `[16, 256, 256]`.
        if width <= 512.0 {
            Schedule::Warps {
                threads: 128,
                vector: 1,
                lanes: if width >= 128.0 { 4 } else { 1 },
                tiles: [1, 1],
            }
        } else {
            Schedule::Rows {
                threads: 64,
                vector: 1,
                tiles: [1, 1],
            }
        }
    } else if width <= 256.0 || (width <= 512.0 && rows >= 1024.0) {
        let (lanes, vector) = if rows >= 1024.0 {
            let mut lanes = 32;
            while lanes > 1 && f64::from(lanes * widest) > width {
                lanes /= 2;
            }
            (lanes, widest)
        } else {
            (32, vector(32))
        };
        Schedule::Warps {
            threads: 128,
            vector,
            lanes,
            tiles: [1, 1],
        }
    } else {
        // With enough rows to fill the GPU, each thread covers about four vectors of its row, which the tuner measured faster on 256 rows of 1536 to 7168 points; fewer rows spread each row over more threads.
        let per_thread = if rows >= 128.0 { 4.0 } else { 1.0 };
        // Every thread takes a whole vector even when some of the block's threads then have none, which the tuner measured faster on rows of 512 to 1536 points.
        let threads = row_threads(width / (f64::from(widest) * per_thread)).clamp(64, 512);
        Schedule::Rows {
            threads,
            vector: widest,
            tiles: [1, 1],
        }
    }
}

/// Lowers inference batch normalization of `inputs`, the input followed by its present weight, bias, running mean, and running variance, over dimension one.
///
/// An absent running mean is zero and an absent running variance is one.
///
/// # Errors
///
/// Returns an error for training, mismatched parameters, or types that are not floating-point.
pub fn batch_norm(
    attributes: &BatchNormalizationAttributes,
    inputs: &[(DataType, &Layout)],
) -> Result<LoweredKernel> {
    if attributes.training {
        return Err(Error::unsupported_operation(
            "cuda generated batch normalization training",
        ));
    }

    let &(data_type, layout) = inputs
        .first()
        .ok_or_else(|| Error::missing_input("batch norm"))?;
    if !data_type.is_float() {
        return Err(Error::unsupported_data_type(
            "cuda generated batch normalization",
        ));
    }

    let mut next = 0;
    let mut parameter = |present: bool| -> Result<Option<(DataType, &Layout)>> {
        if !present {
            return Ok(None);
        }

        next += 1;
        inputs
            .get(next)
            .copied()
            .map(Some)
            .ok_or_else(|| Error::missing_input("batch norm"))
    };
    let weight = parameter(attributes.has_weight)?;
    let bias = parameter(attributes.has_bias)?;
    let mean = parameter(attributes.has_running_mean)?;
    let variance = parameter(attributes.has_running_var)?;
    let problem = spec::batch_norm_problem(
        layout.shape(),
        weight.map(|(_, layout)| layout.shape()),
        bias.map(|(_, layout)| layout.shape()),
        mean.map(|(_, layout)| layout.shape()),
        variance.map(|(_, layout)| layout.shape()),
        "cuda generated batch normalization",
    )?;
    let dimensions = layout.dimensions();
    let kinds = vec![AxisKind::Parallel; dimensions.len()];
    // Parameters vary only along the channel dimension.
    let channel_strides = |(_, layout): (DataType, &Layout)| {
        let mut strides = vec![0; dimensions.len()];
        if dimensions.len() > 1 {
            strides[1] = layout.strides()[0];
        }
        strides
    };
    let parameters = [weight, bias, mean, variance];
    let mut strides = vec![layout.strides().to_vec()];
    strides.extend(
        parameters
            .iter()
            .flatten()
            .map(|&parameter| channel_strides(parameter)),
    );
    strides.push(Layout::contiguous(dimensions)?.strides().to_vec());
    let mut builder = DomainKernel::create("singe_batch_norm", dimensions, &kinds, &strides)?;
    let element = element_type(data_type)?;
    let compute = element.compute();
    let input = builder.input(element, 0);
    let mut value = builder.kernel.load(input);
    let mut column = 1;
    let mut load =
        |builder: &mut DomainKernel, parameter: Option<(DataType, &Layout)>| -> Result<Option<_>> {
            let Some((parameter_type, _)) = parameter else {
                return Ok(None);
            };
            let access = builder.input(element_type(parameter_type)?, column);
            column += 1;
            let node = builder.kernel.load(access);
            Ok(Some(builder.convert(node, compute)))
        };
    let weight = load(&mut builder, weight)?;
    let bias = load(&mut builder, bias)?;
    let mean = load(&mut builder, mean)?;
    let variance = load(&mut builder, variance)?;
    let eps = builder.scalar(compute, f64::from(attributes.eps), "epsilon");
    if let Some(mean) = mean {
        value = builder.kernel.binary(BinaryOp::Sub, value, mean);
    }

    let variance = variance.unwrap_or_else(|| builder.constant(compute, 1.0));
    let shifted = builder.kernel.binary(BinaryOp::Add, variance, eps);
    let scale = builder.kernel.unary(UnaryOp::Rsqrt, shifted);
    value = builder.kernel.binary(BinaryOp::Mul, value, scale);
    if let Some(weight) = weight {
        value = builder.kernel.binary(BinaryOp::Mul, value, weight);
    }

    if let Some(bias) = bias {
        value = builder.kernel.binary(BinaryOp::Add, value, bias);
    }

    let output = builder.output(element, column);
    builder.kernel.store(output, value);
    let schedule = element_schedule(&builder.kernel, 128);
    Ok(builder.finish(ValueSpec::new(data_type, problem.output_shape), schedule))
}

// Lowers `cumsum` or `cumprod` of `group`'s value along `dim` as an inclusive scan, writing every prefix; integers and booleans accumulate in `I64`.
pub fn scan(operation: Operation, group: &ElementGroup<'_>, dim: i64) -> Result<LoweredKernel> {
    let combiner = match operation {
        Operation::Cumsum => Combiner::Sum,
        Operation::Cumprod => Combiner::Product,
        _ => return Err(Error::unsupported_operation("cuda generated scan")),
    };
    let output_type = if group.data_type.is_float() {
        group.data_type
    } else {
        DataType::I64
    };
    let mut strides = group_strides(group)?;
    // A scalar is one row of one point.
    let (dimensions, axis) = if group.shape.rank() == 0 {
        for strides in &mut strides {
            *strides = vec![0];
        }
        (vec![1], 0)
    } else {
        let rank = group.shape.rank();
        (
            group.shape.dimensions().to_vec(),
            normalize_dim_usize(dim, rank)?,
        )
    };
    let kinds = (0..dimensions.len())
        .map(|index| {
            if index == axis {
                AxisKind::Reduction
            } else {
                AxisKind::Parallel
            }
        })
        .collect::<Vec<_>>();
    let contiguous = Layout::contiguous(dimensions.as_slice())?
        .strides()
        .to_vec();
    strides.push(contiguous.clone());
    if group.stored {
        strides.push(contiguous);
    }

    let mut builder = DomainKernel::create(
        &kernel_name(operation, group),
        &dimensions,
        &kinds,
        &strides,
    )?;
    let value = builder.group_value(group)?;
    let element = element_type(output_type)?;
    let widened = builder.convert(value, element.compute());
    // A scan along a dimension of one point is the value itself, and an empty one has no points to scan.
    let scanned = if dimensions[axis] == 1 || dimensions.contains(&0) {
        widened
    } else {
        let scanned_axis = *builder
            .reduction_axes()
            .first()
            .ok_or_else(|| Error::missing_node("scan"))?;
        builder.kernel.scan(combiner, scanned_axis, widened)
    };
    let column = group.inputs.len();
    let output = builder.output(element, column);
    builder.kernel.store(output, scanned);
    builder.store_group(group, value, column + 1)?;
    // Many or short rows take a thread each, which walks its row in order; fewer long ones take a block each, whose threads scan contiguous chunks.
    let (rows, width) = (
        builder.count(AxisKind::Parallel),
        builder.count(AxisKind::Reduction),
    );
    let schedule = if width <= 64.0 || rows >= 2048.0 {
        Schedule::Elements {
            threads: 128,
            vector: 1,
        }
    } else {
        Schedule::Rows {
            threads: row_threads(width / 16.0).clamp(32, 512),
            vector: 1,
            tiles: [1, 1],
        }
    };
    Ok(builder.finish(ValueSpec::new(output_type, group.shape.clone()), schedule))
}
