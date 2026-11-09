use crate::{
    backend::{
        registry::{
            Operation, OperationAttributes, OperationInference, ResultAlias,
            infer::InferenceContext,
        },
        spec,
    },
    data_type::DataType,
    error::{Error, Result},
    scalar::Scalar,
    shape::Shape,
    value::ValueSpec,
};

/// Keeps the first input's shape and data type.
#[expect(
    clippy::unnecessary_wraps,
    reason = "every inference rule has the fallible `InferenceRule` signature"
)]
pub fn same_as_input(context: &InferenceContext<'_>) -> Result<OperationInference> {
    Ok(context.same_type(context.shape()))
}

/// Computes a floating-point function, which promotes integer and Boolean operands to `F32` as `PyTorch` does.
pub fn floating_unary(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let data_type = floating(context.first().data_type())?;
    Ok(context.converted(context.shape(), data_type))
}

/// Computes a function only floating types define, such as an activation.
pub fn float_only(context: &InferenceContext<'_>) -> Result<OperationInference> {
    for input in context.inputs {
        require_float(input.data_type())?;
    }
    broadcast(context)
}

/// Negates, takes the absolute value of, or rectifies the operand, which has no Boolean meaning.
pub fn signed_unary(context: &InferenceContext<'_>) -> Result<OperationInference> {
    reject_bool(
        context,
        "boolean negation, absolute value, or rectification",
    )?;
    same_as_input(context)
}

/// Rounds the operand, which leaves integers unchanged and has no Boolean meaning.
pub fn rounding(context: &InferenceContext<'_>) -> Result<OperationInference> {
    reject_bool(context, "boolean rounding")?;
    same_as_input(context)
}

/// Complements an integer operand's bits or a Boolean operand's truth values; floating operands have no bits to complement.
pub fn bitwise_not(context: &InferenceContext<'_>) -> Result<OperationInference> {
    if context.first().data_type().is_float() {
        return Err(Error::unsupported_data_type(
            "bitwise complement of floating values",
        ));
    }
    same_as_input(context)
}

/// Squares the operand, computing Boolean operands as `I64`.
#[expect(
    clippy::unnecessary_wraps,
    reason = "every inference rule has the fallible `InferenceRule` signature"
)]
pub fn square(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let data_type = match context.first().data_type() {
        DataType::Bool => DataType::I64,
        data_type => data_type,
    };
    Ok(context.converted(context.shape(), data_type))
}

/// Returns the floating type a floating-point function of `data_type` computes in: `data_type` itself, or `F32` for integers and Booleans.
fn floating(data_type: DataType) -> Result<DataType> {
    match data_type {
        DataType::F4E2M1 => Err(Error::unsupported_data_type("fp4 arithmetic")),
        data_type if data_type.is_float() => Ok(data_type),
        _ => Ok(DataType::F32),
    }
}

fn require_float(data_type: DataType) -> Result<()> {
    if data_type.is_float() && data_type != DataType::F4E2M1 {
        Ok(())
    } else {
        Err(Error::unsupported_data_type(
            "floating-point function operand",
        ))
    }
}

pub fn reject_bool(context: &InferenceContext<'_>, what: &'static str) -> Result<()> {
    if context.first().data_type() == DataType::Bool {
        Err(Error::invalid_argument(what))
    } else {
        Ok(())
    }
}

/// Broadcasts operands of one data type.
pub fn broadcast(context: &InferenceContext<'_>) -> Result<OperationInference> {
    context.require_same_types()?;
    let mut shape = context.shape().clone();
    for input in &context.inputs[1..] {
        shape = shape.broadcast_with(input.shape())?;
    }
    Ok(context.same_type(&shape))
}

/// Broadcasts two operands and applies arithmetic, true-division, or Boolean type promotion.
pub fn promoted_binary(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let lhs = context.input(0)?;
    let rhs = context.input(1)?;
    let binary = context
        .schema
        .binary_request(lhs.clone(), rhs.clone())
        .ok_or_else(|| Error::unsupported_operation("binary promotion"))?;
    let inference = binary.infer()?;
    Ok(OperationInference {
        outputs: inference.outputs().to_vec(),
        aliases: vec![ResultAlias::Independent],
        computation_data_type: inference.computation_data_type(),
        kernel_input_data_types: vec![inference.computation_data_type(); context.inputs.len()],
    })
}

/// Tests each element for NaN in the operand's own type, returning `Bool`.
#[expect(
    clippy::unnecessary_wraps,
    reason = "every inference rule has the fallible `InferenceRule` signature"
)]
pub fn is_nan(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let mut inference = context.converted(context.shape(), context.first().data_type());
    inference.outputs = vec![ValueSpec::new(DataType::Bool, context.shape().clone())];
    Ok(inference)
}

/// Negates the operand's truth value.
#[expect(
    clippy::unnecessary_wraps,
    reason = "every inference rule has the fallible `InferenceRule` signature"
)]
pub fn logical_not(context: &InferenceContext<'_>) -> Result<OperationInference> {
    Ok(context.converted(context.shape(), DataType::Bool))
}

/// Selects between two operands of one data type with a Boolean condition.
pub fn where_select(context: &InferenceContext<'_>) -> Result<OperationInference> {
    if context.first().data_type() != DataType::Bool {
        return Err(Error::unsupported_data_type("where condition"));
    }

    let on_true = context.input(1)?;
    let on_false = context.input(2)?;
    if on_true.data_type() != on_false.data_type() {
        return Err(Error::data_type_mismatch(
            "operation operands",
            on_true.data_type(),
            on_false.data_type(),
        ));
    }

    let problem =
        spec::where_elementwise_problem(context.shape(), on_true.shape(), on_false.shape())?;
    Ok(context.finish(&problem.output_shape, on_true.data_type()))
}

/// Combines the operand with a scalar, promoting its data type.
pub fn scalar(context: &InferenceContext<'_>) -> Result<OperationInference> {
    scalar_with(context, false)
}

/// Divides a scalar by the operand, promoting integer operands to floating point.
pub fn reverse_divide_scalar(context: &InferenceContext<'_>) -> Result<OperationInference> {
    scalar_with(context, true)
}

fn scalar_with(context: &InferenceContext<'_>, true_division: bool) -> Result<OperationInference> {
    let value = context.value(OperationAttributes::scalar)?;
    let subtraction = matches!(
        context.operation,
        Operation::SubScalar | Operation::RSubScalar
    );
    if subtraction
        && (context.first().data_type() == DataType::Bool || matches!(value, Scalar::Bool(_)))
    {
        return Err(Error::invalid_argument("boolean subtraction"));
    }

    let data_type = value.promoted_type(context.first().data_type(), true_division)?;
    Ok(context.finish(context.shape(), data_type))
}

/// Clamps the operand to a valid range.
pub fn clamp(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let clamp = context.attributes(OperationAttributes::clamp)?;
    let problem = spec::clamp_problem(context.shape(), clamp.min, clamp.max)?;
    // The bounds are floating, so they promote integer and Boolean operands to `F32`.
    let data_type = floating(context.first().data_type())?;
    Ok(context.converted(&problem.output_shape, data_type))
}

/// Interpolates between two operands of one data type.
pub fn lerp(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let t = context.value(OperationAttributes::lerp_t)?;
    context.require_same_types()?;
    require_float(context.first().data_type())?;
    let problem = spec::lerp_problem(context.shape(), context.input(1)?.shape(), t)?;
    Ok(context.same_type(&problem.output_shape))
}

/// Converts the operand to the attribute's data type.
pub fn cast(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let data_type = context.value(OperationAttributes::cast_data_type)?;
    Ok(context.finish(context.shape(), data_type))
}

/// Quantizes to the attribute's data type, or dequantizes to `F32`, with per-tensor or per-channel parameters.
pub fn quantization(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let quantization = context.attributes(OperationAttributes::quantization)?;
    let shape = spec::unpacked_shape(context.shape(), quantization.packed)?;
    check_quantization_parameters(
        &shape,
        context.input(1)?,
        context.input(2)?,
        quantization.axis,
        quantization.block_size,
    )?;
    Ok(context.finish(&shape, quantization.data_type.unwrap_or(DataType::F32)))
}

/// Checks that a scale and zero point have the parameter shape of quantizing a tensor of shape `input` per tensor, per channel, or per block.
pub fn check_quantization_parameters(
    input: &Shape,
    scale: &ValueSpec,
    zero_point: &ValueSpec,
    axis: Option<i64>,
    block_size: Option<i64>,
) -> Result<()> {
    let expected = spec::quantization_problem(input, axis, block_size)?.parameter_shape;
    for (name, parameter) in [
        ("quantization scale", scale),
        ("quantization zero point", zero_point),
    ] {
        if parameter.shape() != &expected {
            return Err(Error::shape_mismatch(
                name,
                expected.dimensions(),
                parameter.shape().dimensions(),
            ));
        }
    }
    Ok(())
}

/// Keeps the operand's shape and data type on another device.
pub fn copy_to(context: &InferenceContext<'_>) -> Result<OperationInference> {
    context.value(OperationAttributes::copy_device)?;
    same_as_input(context)
}

/// Runs the inference function of a custom operation.
pub fn custom(context: &InferenceContext<'_>) -> Result<OperationInference> {
    let Operation::Custom(operation) = context.operation else {
        return Err(Error::invalid_argument(
            "custom inference for a built-in operation",
        ));
    };
    let values = context.attributes(OperationAttributes::custom)?;
    let outputs = operation.infer(context.inputs, values)?;
    Ok(OperationInference {
        aliases: vec![ResultAlias::Independent; outputs.len()],
        computation_data_type: outputs
            .first()
            .map_or_else(|| context.first().data_type(), ValueSpec::data_type),
        outputs,
        kernel_input_data_types: context.inputs.iter().map(ValueSpec::data_type).collect(),
    })
}
