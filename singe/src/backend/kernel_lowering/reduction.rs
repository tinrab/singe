use std::collections::HashMap;

use singe_codegen::{
    ir::{AxisKind, BinaryOp, Combiner, ElementType, Literal},
    schedule::{Schedule, reduces_contiguously},
};

use crate::{
    backend::{
        kernel_lowering::{
            domain::{DomainKernel, LoweredKernel, group_strides, kernel_name},
            element_type,
            epilogue::{EpilogueValue, EpilogueWriter, RowEpilogue, RowPoint},
            group::ElementGroup,
            row::{row_schedule, row_threads},
        },
        registry::Operation,
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
