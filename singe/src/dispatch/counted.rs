use crate::{
    backend::registry::{
        LayerNormalizationAttributes, Operation, OperationAttributes, ReductionAttributes,
        SliceAttributes, SortAttributes, TopKAttributes, TransposeAttributes,
    },
    data_type::DataType,
    dispatch::{DispatchKey, Kernel, KernelCall, RegistryBuilder, dispatch_checked},
    error::{Error, Result},
    graph::capture,
    scalar::Scalar,
    shape::normalize_dim_usize,
    tensor::{Tensor, count::Count},
    utility::to_i64,
};

pub fn register(builder: &mut RegistryBuilder) {
    builder.fallback(DispatchKey::Counted, Kernel::new(counted));
}

#[derive(Debug, Default)]
struct Plan {
    fills: Vec<(usize, Scalar)>,
    count: Option<Count>,
}

impl Plan {
    fn carry(count: Count) -> Self {
        Self {
            fills: Vec::new(),
            count: Some(count),
        }
    }

    fn fill(index: usize, value: Scalar) -> Self {
        Self {
            fills: vec![(index, value)],
            count: None,
        }
    }
}

fn counted(call: &KernelCall<'_>) -> Result<Vec<Tensor>> {
    if call.inputs().iter().any(Tensor::requires_grad) {
        return Err(refused(
            call.operation(),
            "gradients do not flow through counted tensors",
        ));
    }

    if let (Operation::Mean, OperationAttributes::Reduction(reduction)) =
        (call.operation(), call.attributes())
        && let Some(mean) = mean(call, reduction)?
    {
        return Ok(vec![mean]);
    }

    let plan = plan(call)?;
    let mut inputs = call.inputs().to_vec();
    for (index, value) in plan.fills {
        let input = inputs
            .get_mut(index)
            .ok_or_else(|| Error::missing_input("counted fill"))?;
        *input = input.fill_padding(value)?;
    }

    let outputs = dispatch_checked(call.request(), &inputs)?;
    let Some(count) = plan.count else {
        return Ok(outputs);
    };
    let count = if call.operation() == Operation::CopyTo {
        let device = outputs
            .first()
            .ok_or_else(|| Error::missing_output("counted copy"))?
            .device();
        Count::create(count.axis(), count.rows().to_device(device)?)?
    } else {
        count
    };
    outputs
        .into_iter()
        .map(|output| output.with_count(count.clone()))
        .collect()
}

fn plan(call: &KernelCall<'_>) -> Result<Plan> {
    let operation = call.operation();
    let inputs = call.inputs();
    let output = output_dimensions(call)?;
    match (operation, call.attributes()) {
        (
            Operation::Sum
            | Operation::Max
            | Operation::Min
            | Operation::Product
            | Operation::LogSumExp
            | Operation::Mean,
            OperationAttributes::Reduction(reduction),
        ) => reduce(call, reduction),
        (Operation::Softmax | Operation::LogSoftmax, &OperationAttributes::Dimension(dim)) => {
            let (index, count) = single(call)?;
            let fills = if count.axis() == axis(dim, &inputs[index])? {
                vec![(index, extreme(inputs[index].data_type(), false)?)]
            } else {
                Vec::new()
            };
            Ok(Plan {
                fills,
                count: Some(count.clone()),
            })
        }
        (Operation::ArgMax | Operation::ArgMin, &OperationAttributes::Dimension(dim)) => {
            let (index, count) = single(call)?;
            let input = &inputs[index];
            let dim = axis(dim, input)?;
            if count.axis() == dim {
                let highest = operation == Operation::ArgMin;
                return Ok(Plan::fill(index, extreme(input.data_type(), highest)?));
            }

            let removed = input.shape().dimensions().len() - output.len();
            Ok(Plan::carry(count.moved_to(if count.axis() > dim {
                count.axis() - removed
            } else {
                count.axis()
            })))
        }
        (Operation::Cumsum | Operation::Cumprod, OperationAttributes::Dimension(_))
        | (Operation::Copy | Operation::Contiguous | Operation::CopyTo, _) => {
            Ok(Plan::carry(single(call)?.1.clone()))
        }
        (Operation::Argsort, &OperationAttributes::Sort(SortAttributes { dim, .. }))
        | (
            Operation::TopK | Operation::TopKValues | Operation::TopKIndices,
            &OperationAttributes::TopK(TopKAttributes { dim, .. }),
        ) => {
            let (index, count) = single(call)?;
            if count.axis() == axis(dim, &inputs[index])? {
                return Err(refused(operation, "it orders along the counted axis"));
            }
            Ok(Plan::carry(count.clone()))
        }
        (
            Operation::Transpose,
            &OperationAttributes::Transpose(TransposeAttributes { dim0, dim1 }),
        ) => {
            let (index, count) = single(call)?;
            let (first, second) = (axis(dim0, &inputs[index])?, axis(dim1, &inputs[index])?);
            Ok(Plan::carry(count.moved_to(match count.axis() {
                axis if axis == first => second,
                axis if axis == second => first,
                axis => axis,
            })))
        }
        (Operation::Permute, OperationAttributes::Permute(order)) => {
            let (index, count) = single(call)?;
            let input = &inputs[index];
            for (position, &dim) in order.iter().enumerate() {
                if axis(dim, input)? == count.axis() {
                    return Ok(Plan::carry(count.moved_to(position)));
                }
            }
            Err(refused(operation, "its order leaves out the counted axis"))
        }
        (
            Operation::Reshape | Operation::Flatten | Operation::Squeeze | Operation::Unsqueeze,
            _,
        ) => {
            let (index, count) = single(call)?;
            reshaped(inputs[index].shape().dimensions(), &output, count.axis())
                .map(|axis| Plan::carry(count.moved_to(axis)))
                .ok_or_else(|| refused(operation, "it merges or splits the counted axis"))
        }
        (Operation::Slice, &OperationAttributes::Slice(slice)) => sliced(call, slice),
        (Operation::Concat, &OperationAttributes::Concat(dim)) => {
            let count = shared(call, |_, axis| Ok(axis))?
                .ok_or_else(|| Error::missing_input("counted concat"))?;
            if count.axis() == axis(dim, &inputs[0])? {
                return Err(refused(operation, "it joins along the counted axis"));
            }
            Ok(Plan::carry(count))
        }
        (Operation::IndexSelect | Operation::Take, &OperationAttributes::IndexSelect(dim)) => {
            index_select(call, dim)
        }
        (Operation::Embedding, _) => {
            if inputs[0].count().is_some() {
                return Err(refused(operation, "its weight is counted"));
            }

            let count = inputs[1]
                .count()
                .ok_or_else(|| Error::missing_input("counted embedding"))?;
            Ok(Plan {
                fills: vec![(1, Scalar::Signed(0))],
                count: Some(count.clone()),
            })
        }
        (Operation::IndexWrite | Operation::IndexPut, _) => {
            let (index, count) = single(call)?;
            if index != 0 {
                return Err(refused(operation, "its source or indices are counted"));
            }

            if let OperationAttributes::IndexPut(attributes) = call.attributes()
                && axis(attributes.dim, &inputs[0])? == count.axis()
            {
                return Err(refused(operation, "it writes along the counted axis"));
            }
            Ok(Plan::carry(count.clone()))
        }
        (
            Operation::LayerNorm | Operation::RmsNorm,
            OperationAttributes::LayerNormalization(LayerNormalizationAttributes {
                normalized_shape,
                ..
            }),
        ) => {
            let (index, count) = single(call)?;
            if index != 0 {
                return Err(refused(operation, "its weight or bias is counted"));
            }

            let leading = inputs[0].shape().dimensions().len() - normalized_shape.len();
            if count.axis() >= leading {
                return Err(refused(operation, "it normalizes along the counted axis"));
            }
            Ok(Plan::carry(count.clone()))
        }
        (Operation::MatMul, _) => matmul(call, &output),
        _ if matches!(
            operation,
            Operation::Where
                | Operation::Clamp
                | Operation::Lerp
                | Operation::Cast
                | Operation::Expand
                | Operation::Bernoulli
        ) || operation.is_unary_elementwise()
            || operation.is_binary_elementwise()
            || operation.is_scalar_elementwise() =>
        {
            broadcast(call, &output)
        }
        _ => Err(refused(operation, "it has no rule for counted operands")),
    }
}

fn mean(call: &KernelCall<'_>, reduction: &ReductionAttributes) -> Result<Option<Tensor>> {
    let input = call.input(0)?;
    let Some(count) = input.count() else {
        return Ok(None);
    };
    let dimensions = input.shape().dimensions();
    let reduced = reduced_axes(reduction, dimensions.len())?;
    if !reduced.contains(&count.axis()) {
        return Ok(None);
    }

    let sum = input
        .fill_padding(Scalar::Signed(0))?
        .sum(&reduction.dimensions, reduction.keepdim)?;
    let others = reduced
        .iter()
        .filter(|&&axis| axis != count.axis())
        .map(|&axis| dimensions[axis])
        .product::<i64>();
    let denominator = count.rows().cast(sum.data_type())?.mul_scalar(others)?;
    sum.div(&denominator).map(Some)
}

fn reduce(call: &KernelCall<'_>, reduction: &ReductionAttributes) -> Result<Plan> {
    let (index, count) = single(call)?;
    let input = &call.inputs()[index];
    let reduced = reduced_axes(reduction, input.shape().dimensions().len())?;
    if !reduced.contains(&count.axis()) {
        let removed = if reduction.keepdim {
            0
        } else {
            reduced.iter().filter(|&&axis| axis < count.axis()).count()
        };
        return Ok(Plan::carry(count.moved_to(count.axis() - removed)));
    }

    let data_type = input.data_type();
    let identity = match call.operation() {
        Operation::Sum => Scalar::Signed(0),
        Operation::Product => Scalar::Signed(1),
        Operation::Max | Operation::LogSumExp => extreme(data_type, false)?,
        Operation::Min => extreme(data_type, true)?,
        operation => return Err(refused(operation, "it has no identity to fill with")),
    };
    Ok(Plan::fill(index, identity))
}

fn sliced(call: &KernelCall<'_>, slice: SliceAttributes) -> Result<Plan> {
    let (index, count) = single(call)?;
    let input = &call.inputs()[index];
    if axis(slice.dim, input)? != count.axis() {
        return Ok(Plan::carry(count.clone()));
    }

    if slice.start != 0 || slice.step != 1 {
        return Err(refused(
            call.operation(),
            "it starts past the first row or steps along the counted axis",
        ));
    }

    if slice.length == input.shape().dimensions()[count.axis()] {
        return Ok(Plan::carry(count.clone()));
    }

    let length = capture::literal(input.device(), |device| {
        Tensor::scalar(device, slice.length)
    })?;
    Ok(Plan::carry(Count::create(
        count.axis(),
        count.rows().minimum(&length)?,
    )?))
}

fn index_select(call: &KernelCall<'_>, dim: i64) -> Result<Plan> {
    let inputs = call.inputs();
    let (input, indices) = (&inputs[0], &inputs[1]);
    let dim = axis(dim, input)?;
    let index_rank = indices.shape().dimensions().len();
    let carried = input
        .count()
        .filter(|count| count.axis() != dim)
        .map(|count| {
            count.moved_to(if count.axis() > dim {
                count.axis() + index_rank - 1
            } else {
                count.axis()
            })
        });
    let Some(count) = indices.count() else {
        return Ok(Plan {
            fills: Vec::new(),
            count: carried,
        });
    };
    if carried.is_some() {
        return Err(refused(
            call.operation(),
            "both its input and its indices are counted",
        ));
    }

    Ok(Plan {
        fills: vec![(1, Scalar::Signed(0))],
        count: Some(count.moved_to(dim + count.axis())),
    })
}

fn matmul(call: &KernelCall<'_>, output: &[i64]) -> Result<Plan> {
    let mut plan = Plan::default();
    let rank = output.len();
    for (index, input) in call.inputs().iter().enumerate() {
        let Some(count) = input.count() else {
            continue;
        };
        let dimensions = input.shape().dimensions();
        let own = dimensions.len();
        if own < 2 || rank < 2 {
            return Err(refused(call.operation(), "a counted operand is a vector"));
        }

        let contracted = if index == 0 { own - 1 } else { own - 2 };
        let axis = if count.axis() == contracted {
            plan.fills.push((index, Scalar::Signed(0)));
            continue;
        } else if count.axis() >= own - 2 {
            rank - (own - count.axis())
        } else {
            let axis = count.axis() + rank - own;
            if output[axis] != dimensions[count.axis()] {
                return Err(refused(call.operation(), "it broadcasts the counted axis"));
            }
            axis
        };
        merge(&mut plan.count, count.moved_to(axis), call.operation())?;
    }
    Ok(plan)
}

fn broadcast(call: &KernelCall<'_>, output: &[i64]) -> Result<Plan> {
    let operation = call.operation();
    let count = shared(call, |index, axis| {
        let dimensions = call.inputs()[index].shape().dimensions();
        let moved = axis + output.len() - dimensions.len();
        if output[moved] != dimensions[axis] {
            return Err(refused(operation, "it broadcasts the counted axis"));
        }
        Ok(moved)
    })?
    .ok_or_else(|| Error::missing_input("counted broadcast"))?;
    Ok(Plan::carry(count))
}

fn shared(
    call: &KernelCall<'_>,
    moved: impl Fn(usize, usize) -> Result<usize>,
) -> Result<Option<Count>> {
    let mut shared = None;
    for (index, input) in call.inputs().iter().enumerate() {
        let Some(count) = input.count() else {
            continue;
        };
        let axis = moved(index, count.axis())?;
        merge(&mut shared, count.moved_to(axis), call.operation())?;
    }
    Ok(shared)
}

fn merge(shared: &mut Option<Count>, count: Count, operation: Operation) -> Result<()> {
    match shared {
        Some(existing) if !existing.is_same(&count) => Err(refused(
            operation,
            "its operands hold different counts or count different axes",
        )),
        Some(_) => Ok(()),
        None => {
            *shared = Some(count);
            Ok(())
        }
    }
}

fn single<'a>(call: &KernelCall<'a>) -> Result<(usize, &'a Count)> {
    let mut counted = call
        .inputs()
        .iter()
        .enumerate()
        .filter_map(|(index, input)| input.count().map(|count| (index, count)));
    let first = counted
        .next()
        .ok_or_else(|| Error::missing_input("counted operand"))?;
    if counted.next().is_some() {
        return Err(refused(
            call.operation(),
            "more than one operand is counted",
        ));
    }
    Ok(first)
}

fn reshaped(input: &[i64], output: &[i64], counted: usize) -> Option<usize> {
    let before = input[..counted].iter().product::<i64>();
    let mut product = 1;
    for (axis, &extent) in output.iter().enumerate() {
        if product == before && extent == input[counted] {
            return Some(axis);
        }

        product *= extent;
        if product > before && before != 0 {
            return None;
        }
    }
    None
}

fn reduced_axes(reduction: &ReductionAttributes, rank: usize) -> Result<Vec<usize>> {
    if reduction.dimensions.is_empty() {
        return Ok((0..rank).collect());
    }

    let rank = to_i64(rank, "reduction rank")?;
    reduction
        .dimensions
        .iter()
        .map(|&dim| normalize_dim_usize(dim, rank))
        .collect()
}

fn axis(dim: i64, input: &Tensor) -> Result<usize> {
    normalize_dim_usize(dim, input.shape().rank())
}

fn output_dimensions(call: &KernelCall<'_>) -> Result<Vec<i64>> {
    call.request()
        .inference()
        .outputs
        .first()
        .map(|output| output.shape().dimensions().to_vec())
        .ok_or_else(|| Error::missing_output("counted operation"))
}

fn extreme(data_type: DataType, highest: bool) -> Result<Scalar> {
    let signed = |low: i64, high: i64| Scalar::Signed(if highest { high } else { low });
    let unsigned = |high: u64| Scalar::Unsigned(if highest { high } else { 0 });
    Ok(match data_type {
        DataType::Bool => Scalar::Bool(highest),
        DataType::U8 => unsigned(u8::MAX.into()),
        DataType::U16 => unsigned(u16::MAX.into()),
        DataType::U32 => unsigned(u32::MAX.into()),
        DataType::U64 => unsigned(u64::MAX),
        DataType::I8 => signed(i8::MIN.into(), i8::MAX.into()),
        DataType::I16 => signed(i16::MIN.into(), i16::MAX.into()),
        DataType::I32 => signed(i32::MIN.into(), i32::MAX.into()),
        DataType::I64 => signed(i64::MIN, i64::MAX),
        _ if data_type.is_float() => Scalar::Float(if highest {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        }),
        _ => return Err(Error::unsupported_data_type("counted fill")),
    })
}

fn refused(operation: Operation, reason: &str) -> Error {
    Error::unsupported_operation_with_details(
        operation.name(),
        format!("a counted operand, since {reason}"),
    )
}
