//! Quantization and dequantization with per-tensor, per-channel, or per-block scales and zero points.

use singe_codegen::{
    ir::{AxisKind, BinaryOp, ElementType, Expr, Literal, NodeId, UnaryOp},
    schedule::element_schedule,
};

use crate::{
    backend::{
        kernel_lowering::{
            STATUS_INVALID_QUANTIZATION_SCALE,
            domain::{DomainKernel, LoweredKernel},
        },
        registry::{Operation, PackedFormat, QuantizationAttributes},
        spec::{QuantizationGranularity, quantization_problem, unpacked_shape},
    },
    data_type::DataType,
    error::{Error, Result},
    layout::Layout,
    shape::normalize_dim_usize,
    utility::to_i64,
    value::ValueSpec,
};

/// Lowers `quantize` or `dequantize` of `input` with its `F32` `scale` and `I64` `zero_point` tensors, laid out as the attributes' granularity requires.
///
/// Quantizing rounds `input / scale` to nearest with ties to even, adds the zero point, and saturates, mapping NaN to the saturated zero point; dequantizing computes `(input - zero_point) × scale`.
/// A scale that is not finite and positive reports [`STATUS_INVALID_QUANTIZATION_SCALE`].
///
/// # Errors
///
/// Returns an error if the data types are not ones the operation takes or the parameters do not fit the input.
pub fn quantization(
    operation: Operation,
    attributes: QuantizationAttributes,
    input: (DataType, &Layout),
    scale: (DataType, &Layout),
    zero_point: (DataType, &Layout),
) -> Result<LoweredKernel> {
    if scale.0 != DataType::F32 || zero_point.0 != DataType::I64 {
        return Err(Error::unsupported_data_type(
            "cuda quantization parameters must be f32 scales and i64 zero points",
        ));
    }

    let (data_type, layout) = input;
    let packed = attributes
        .packed
        .filter(|_| operation == Operation::Dequantize);
    let logical = unpacked_shape(layout.shape(), packed)?;
    let shape = &logical;
    let problem = quantization_problem(shape, attributes.axis, attributes.block_size)?;
    if packed.is_some() && matches!(problem.granularity, QuantizationGranularity::Block { .. }) {
        return Err(Error::unsupported_operation(
            "cuda packed dequantize with per-block scales",
        ));
    }

    for parameter in [scale.1, zero_point.1] {
        if parameter.shape() != &problem.parameter_shape {
            return Err(Error::shape_mismatch(
                "cuda quantization parameters",
                problem.parameter_shape.dimensions().to_vec(),
                parameter.dimensions().to_vec(),
            ));
        }
    }

    let (output_type, element) = match operation {
        Operation::Quantize if data_type == DataType::F32 => {
            let output = attributes.data_type.unwrap_or(DataType::I8);
            if !matches!(output, DataType::I8 | DataType::U8) {
                return Err(Error::unsupported_data_type("cuda quantize output"));
            }
            (output, ElementType::F32)
        }
        Operation::Quantize => return Err(Error::unsupported_data_type("cuda quantize input")),
        Operation::Dequantize if matches!(data_type, DataType::I8 | DataType::U8) => {
            (DataType::F32, ElementType::I64)
        }
        Operation::Dequantize => {
            return Err(Error::unsupported_data_type("cuda dequantize input"));
        }
        _ => return Err(Error::unsupported_operation("cuda quantization kernel")),
    };
    // The domain is the input, with a blocked axis split into its blocks and the positions inside one.
    let axis = attributes
        .axis
        .map(|axis| normalize_dim_usize(axis, shape.rank()))
        .transpose()?
        .unwrap_or(0);
    let mut dimensions = shape.dimensions().to_vec();
    let mut input_strides = layout.strides().to_vec();
    if packed.is_some() {
        // The bytes of a packed input: half the last extent, and no stride along the nibble.
        dimensions.pop();
        dimensions.extend(layout.dimensions().last());
    }

    let parameter_strides = |parameter: &Layout| -> Vec<i64> {
        match problem.granularity {
            QuantizationGranularity::Tensor => vec![0; dimensions.len()],
            QuantizationGranularity::Channel { .. } => {
                let mut strides = vec![0; dimensions.len()];
                strides[axis] = parameter.strides()[0];
                strides
            }
            QuantizationGranularity::Block { .. } => parameter.strides().to_vec(),
        }
    };
    let mut scale_strides = parameter_strides(scale.1);
    let mut zero_point_strides = parameter_strides(zero_point.1);
    if let QuantizationGranularity::Block { block_size, .. } = problem.granularity {
        let block = to_i64(block_size, "cuda quantization block size")?;
        dimensions[axis] /= block;
        dimensions.insert(axis + 1, block);
        input_strides[axis] *= block;
        input_strides.insert(axis + 1, layout.strides()[axis]);
        scale_strides.insert(axis + 1, 0);
        zero_point_strides.insert(axis + 1, 0);
    }

    let mut output_strides = Layout::contiguous(shape.dimensions())?.strides().to_vec();
    if packed.is_some() {
        // The last axis splits into the bytes and the two nibbles of each, which the output holds in turn.
        let split = |strides: &mut Vec<i64>| {
            let inner = strides.last().copied().unwrap_or(0);
            if let Some(last) = strides.last_mut() {
                *last = 2 * inner;
            }

            strides.push(inner);
        };
        split(&mut output_strides);
        split(&mut scale_strides);
        split(&mut zero_point_strides);
        dimensions.push(2);
        input_strides.push(0);
    } else if output_strides.len() != dimensions.len() {
        output_strides = Layout::contiguous(dimensions.as_slice())?
            .strides()
            .to_vec();
    }

    let kinds = vec![AxisKind::Parallel; dimensions.len()];
    let mut builder = DomainKernel::create(
        &format!("singe_{}", operation.name()),
        &dimensions,
        &kinds,
        &[
            input_strides,
            scale_strides,
            zero_point_strides,
            output_strides,
        ],
    )?;
    let value = builder.input(storage_type(data_type), 0);
    let value = builder.kernel.load(value);
    let scale = builder.input(ElementType::F32, 1);
    let scale = builder.kernel.load(scale);
    let zero_point = builder.input(ElementType::I64, 2);
    let zero_point = builder.kernel.load(zero_point);
    // A scale is valid when it is positive and finite, which NaN is not.
    let zero = builder.constant(ElementType::F32, 0.0);
    let infinity = builder.constant(ElementType::F32, f64::INFINITY);
    let positive = builder.kernel.binary(BinaryOp::Greater, scale, zero);
    let finite = builder.kernel.binary(BinaryOp::Less, scale, infinity);
    let valid = builder.kernel.binary(BinaryOp::And, positive, finite);
    let scale = builder
        .kernel
        .require(scale, valid, STATUS_INVALID_QUANTIZATION_SCALE);
    let result = if operation == Operation::Quantize {
        let quotient = builder.kernel.binary(BinaryOp::Div, value, scale);
        let rounded = builder.kernel.unary(UnaryOp::Round, quotient);
        let nan = builder.kernel.unary(UnaryOp::IsNan, value);
        let rounded = builder.kernel.select(nan, zero, rounded);
        let rounded = builder.kernel.cast(ElementType::F64, rounded);
        let zero_point = builder.kernel.cast(ElementType::F64, zero_point);
        let shifted = builder.kernel.binary(BinaryOp::Add, rounded, zero_point);
        let (low, high) = if output_type == DataType::I8 {
            (-128.0, 127.0)
        } else {
            (0.0, 255.0)
        };
        let low = builder
            .kernel
            .literal(ElementType::F64, Literal::Float(low));
        let high = builder
            .kernel
            .literal(ElementType::F64, Literal::Float(high));
        let clamped = builder.kernel.binary(BinaryOp::Max, shifted, low);
        let clamped = builder.kernel.binary(BinaryOp::Min, clamped, high);
        builder.kernel.cast(storage_type(output_type), clamped)
    } else {
        let centered = if let Some(format) = packed {
            let value = unpacked(&mut builder, value, data_type, format)?;
            let zero_point = builder.kernel.cast(ElementType::F32, zero_point);
            builder.kernel.binary(BinaryOp::Sub, value, zero_point)
        } else {
            let value = builder.convert(value, element);
            let centered = builder.kernel.binary(BinaryOp::Sub, value, zero_point);
            builder.kernel.cast(ElementType::F32, centered)
        };
        builder.kernel.binary(BinaryOp::Mul, centered, scale)
    };
    let output = builder.output(storage_type(output_type), 3);
    builder.kernel.store(output, result);
    let schedule = element_schedule(&builder.kernel, 128);
    Ok(builder.finish(ValueSpec::new(output_type, shape.clone()), schedule))
}

/// Returns the kernel element type that `F32`, `I8`, and `U8` values are stored as.
const fn storage_type(data_type: DataType) -> ElementType {
    match data_type {
        DataType::I8 => ElementType::I8,
        DataType::U8 => ElementType::U8,
        _ => ElementType::F32,
    }
}

/// Returns the `F32` value of the 4-bit code in `byte`, a loaded element of a `container` tensor packed as `format`, at the nibble the domain's innermost axis indexes, 0 for the low one.
///
/// The decoding matches [`unpacked_value`](crate::backend::spec::unpacked_value), with integer division and remainder for the bit fields.
///
/// # Errors
///
/// Returns an error if the domain has no axes.
pub fn unpacked(
    builder: &mut DomainKernel,
    byte: NodeId,
    container: DataType,
    format: PackedFormat,
) -> Result<NodeId> {
    let integer = |builder: &mut DomainKernel, value: i64| {
        builder
            .kernel
            .literal(ElementType::I32, Literal::Int(value))
    };
    let nibble = builder
        .innermost_axis()
        .ok_or_else(|| Error::invalid_argument("packed dequantize domain"))?;
    let nibble = builder
        .kernel
        .add_node(ElementType::I32, Expr::AxisIndex(nibble));
    // The byte's unsigned bits, in 32-bit arithmetic: a negative `I8` wraps to its value plus 256.
    let byte = builder.convert(byte, ElementType::I32);
    let byte = if container == DataType::I8 {
        let wrap = integer(builder, 256);
        let shifted = builder.kernel.binary(BinaryOp::Add, byte, wrap);
        builder.kernel.binary(BinaryOp::Rem, shifted, wrap)
    } else {
        byte
    };
    let sixteen = integer(builder, 16);
    let low = builder.kernel.binary(BinaryOp::Rem, byte, sixteen);
    let high = builder.kernel.binary(BinaryOp::Div, byte, sixteen);
    let zero = integer(builder, 0);
    let first = builder.kernel.binary(BinaryOp::Equal, nibble, zero);
    let code = builder.kernel.select(first, low, high);
    let eight = integer(builder, 8);
    let negative = builder.kernel.binary(BinaryOp::GreaterEqual, code, eight);
    Ok(match format {
        PackedFormat::Int4 if container == DataType::I8 => {
            let wrapped = builder.kernel.binary(BinaryOp::Sub, code, sixteen);
            let signed = builder.kernel.select(negative, wrapped, code);
            builder.kernel.cast(ElementType::F32, signed)
        }
        PackedFormat::Int4 => builder.kernel.cast(ElementType::F32, code),
        PackedFormat::Float4E2M1 => {
            // Bits `s e e m`: zero exponents are subnormal `m / 2`, others `(1 + m / 2) · 2^(e - 1)`.
            let two = integer(builder, 2);
            let magnitude_code = builder.kernel.binary(BinaryOp::Rem, code, eight);
            let exponent = builder.kernel.binary(BinaryOp::Div, magnitude_code, two);
            let mantissa = builder.kernel.binary(BinaryOp::Rem, magnitude_code, two);
            let mantissa = builder.kernel.cast(ElementType::F32, mantissa);
            let half = builder.constant(ElementType::F32, 0.5);
            let one = builder.constant(ElementType::F32, 1.0);
            let fraction = builder.kernel.binary(BinaryOp::Mul, mantissa, half);
            let normal = builder.kernel.binary(BinaryOp::Add, one, fraction);
            let is_two = builder.kernel.binary(BinaryOp::Equal, exponent, two);
            let three = integer(builder, 3);
            let is_three = builder
                .kernel
                .binary(BinaryOp::GreaterEqual, exponent, three);
            let two_float = builder.constant(ElementType::F32, 2.0);
            let four_float = builder.constant(ElementType::F32, 4.0);
            let scale = builder.kernel.select(is_two, two_float, one);
            let scale = builder.kernel.select(is_three, four_float, scale);
            let normal = builder.kernel.binary(BinaryOp::Mul, normal, scale);
            let subnormal = builder.kernel.binary(BinaryOp::Equal, exponent, zero);
            let magnitude = builder.kernel.select(subnormal, fraction, normal);
            let negated = builder.kernel.unary(UnaryOp::Neg, magnitude);
            builder.kernel.select(negative, negated, magnitude)
        }
    })
}
