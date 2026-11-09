use num_traits::ToPrimitive;

use crate::{
    backend::{
        cpu::{
            iteration::{ElementwisePlan, typed_values},
            utility::cpu_device_mismatch,
        },
        registry::{PackedFormat, QuantizationAttributes},
        spec::{QuantizationProblem, quantization_problem, unpacked_shape, unpacked_value},
    },
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    scalar::Scalar,
    shape::Shape,
    tensor::{Tensor, TensorDataType, tensor_data_type_cases},
};

/// Host copies of quantization scales and zero points with the mapping from input elements to them.
pub struct QuantizationParameters {
    problem: QuantizationProblem,
    scales: Vec<f32>,
    zero_points: Vec<i64>,
}

impl QuantizationParameters {
    /// Reads the scale and zero point tensors of quantizing a tensor of shape `shape` as `attributes` describe.
    pub fn create(
        shape: &Shape,
        scale: &Tensor,
        zero_point: &Tensor,
        attributes: QuantizationAttributes,
    ) -> Result<Self> {
        for parameter in [scale, zero_point] {
            if parameter.device() != Device::Cpu {
                return Err(cpu_device_mismatch(
                    "cpu quantization parameter",
                    parameter.device(),
                ));
            }
        }
        Ok(Self {
            problem: quantization_problem(shape, attributes.axis, attributes.block_size)?,
            scales: scale.to_vec::<f32>()?,
            zero_points: zero_point.to_vec::<i64>()?,
        })
    }

    /// Returns the checked scale and the zero point of the input element at row-major index `linear`.
    pub fn at(&self, linear: usize) -> Result<(f32, i64)> {
        let index = self.problem.parameter_index(linear);
        let scale = *self
            .scales
            .get(index)
            .ok_or_else(|| Error::missing_input("at"))?;
        validate_quantization_scale(scale)?;
        Ok((
            scale,
            *self
                .zero_points
                .get(index)
                .ok_or_else(|| Error::missing_input("at"))?,
        ))
    }
}

pub fn validate_quantization_scale(scale: f32) -> Result<()> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(Error::invalid_argument("quantization scale"));
    }
    Ok(())
}

pub fn f32_to_i64_saturating(value: f32) -> i64 {
    if value.is_nan() {
        return 0;
    }
    value.to_i64().unwrap_or_else(|| {
        if value.is_sign_negative() {
            i64::MIN
        } else {
            i64::MAX
        }
    })
}

pub fn quantized_value(value: f32, scale: f32, zero_point: i64, min: i64, max: i64) -> i64 {
    if value.is_nan() {
        return zero_point.clamp(min, max);
    }

    let scaled = f32_to_i64_saturating((value / scale).round_ties_even());
    let quantized = scaled.saturating_add(zero_point);
    quantized.clamp(min, max)
}

pub fn dequantized_value(value: i64, scale: f32, zero_point: i64) -> Result<f32> {
    let adjusted = i128::from(value) - i128::from(zero_point);
    let adjusted = adjusted
        .to_f32()
        .ok_or_else(|| Error::out_of_range("cpu dequantize adjusted value"))?;
    Ok(adjusted * scale)
}

pub fn cast(input: &Tensor, data_type: DataType) -> Result<Tensor> {
    if input.data_type() == data_type {
        return Ok(input.clone());
    }

    let plan = ElementwisePlan::create(&[input])?;
    macro_rules! read_type {
        ($($ty:ty => $variant:ident = $one:expr),+ $(,)?) => {
            match input.data_type() {
                $(DataType::$variant => {
                    let input = typed_values::<$ty>(input)?;
                    plan.collect(|linear| input[plan.offset(0, linear)].into_scalar())
                },)+
                _ => return Err(Error::unsupported_data_type("cpu cast input")),
            }
        };
    }
    let values: Vec<Scalar> = tensor_data_type_cases!(read_type);
    macro_rules! write_type {
        ($($ty:ty => $variant:ident = $one:expr),+ $(,)?) => {
            match data_type {
                $(DataType::$variant => Tensor::from_vec(Device::Cpu, plan.shape(), values.into_iter().map(<$ty>::from_scalar).collect()),)+
                _ => Err(Error::unsupported_data_type("cpu cast output")),
            }
        };
    }
    tensor_data_type_cases!(write_type)
}

/// Quantizes an `F32` tensor with scales and zero points per tensor, channel, or block.
pub fn quantize(
    input: &Tensor,
    scale: &Tensor,
    zero_point: &Tensor,
    attributes: QuantizationAttributes,
) -> Result<Tensor> {
    if input.device() != Device::Cpu {
        return Err(cpu_device_mismatch("cpu quantize input", input.device()));
    }

    if input.data_type() != DataType::F32 {
        return Err(Error::unsupported_data_type("cpu quantize"));
    }

    let parameters = QuantizationParameters::create(input.shape(), scale, zero_point, attributes)?;
    let values = input.to_vec::<f32>()?;
    let (min, max) = match attributes.data_type {
        Some(DataType::I8) => (i64::from(i8::MIN), i64::from(i8::MAX)),
        Some(DataType::U8) => (i64::from(u8::MIN), i64::from(u8::MAX)),
        _ => return Err(Error::unsupported_data_type("cpu quantize")),
    };
    let quantized = values
        .iter()
        .enumerate()
        .map(|(linear, &value)| {
            let (scale, zero_point) = parameters.at(linear)?;
            Ok(quantized_value(value, scale, zero_point, min, max))
        })
        .collect::<Result<Vec<_>>>()?;
    let shape = input.shape().clone();
    if attributes.data_type == Some(DataType::I8) {
        let values = quantized
            .into_iter()
            .map(|value| i8::try_from(value).expect("quantized i8 value is clamped to i8 range"))
            .collect::<Vec<_>>();
        Tensor::from_vec(Device::Cpu, shape, values)
    } else {
        let values = quantized
            .into_iter()
            .map(|value| u8::try_from(value).expect("quantized u8 value is clamped to u8 range"))
            .collect::<Vec<_>>();
        Tensor::from_vec(Device::Cpu, shape, values)
    }
}

/// Dequantizes an `I8` or `U8` tensor to `F32` with scales and zero points per tensor, channel, or block.
pub fn dequantize(
    input: &Tensor,
    scale: &Tensor,
    zero_point: &Tensor,
    attributes: QuantizationAttributes,
) -> Result<Tensor> {
    if input.device() != Device::Cpu {
        return Err(cpu_device_mismatch("cpu dequantize input", input.device()));
    }

    if let Some(format) = attributes.packed {
        return dequantize_packed(input, scale, zero_point, attributes, format);
    }

    let parameters = QuantizationParameters::create(input.shape(), scale, zero_point, attributes)?;
    let values = match input.data_type() {
        DataType::I8 => input
            .to_vec::<i8>()?
            .into_iter()
            .map(i64::from)
            .collect::<Vec<_>>(),
        DataType::U8 => input
            .to_vec::<u8>()?
            .into_iter()
            .map(i64::from)
            .collect::<Vec<_>>(),
        _ => return Err(Error::unsupported_data_type("cpu dequantize")),
    };
    let output = values
        .iter()
        .enumerate()
        .map(|(linear, &value)| {
            let (scale, zero_point) = parameters.at(linear)?;
            dequantized_value(value, scale, zero_point)
        })
        .collect::<Result<Vec<_>>>()?;
    Tensor::from_vec(Device::Cpu, input.shape().clone(), output)
}

/// Dequantizes an `I8` or `U8` tensor whose bytes each pack two 4-bit values along the last axis, low nibble first, into an `F32` tensor with that axis doubled.
fn dequantize_packed(
    input: &Tensor,
    scale: &Tensor,
    zero_point: &Tensor,
    attributes: QuantizationAttributes,
    format: PackedFormat,
) -> Result<Tensor> {
    let bytes = match input.data_type() {
        DataType::I8 => input
            .to_vec::<i8>()?
            .into_iter()
            .map(i8::cast_unsigned)
            .collect::<Vec<_>>(),
        DataType::U8 => input.to_vec::<u8>()?,
        _ => return Err(Error::unsupported_data_type("cpu packed dequantize")),
    };
    let shape = unpacked_shape(input.shape(), Some(format))?;
    let parameters = QuantizationParameters::create(&shape, scale, zero_point, attributes)?;
    let output = (0..shape.element_count())
        .map(|linear| {
            let (scale, zero_point) = parameters.at(linear)?;
            let value = unpacked_value(bytes[linear / 2], linear % 2, input.data_type(), format);
            #[expect(
                clippy::cast_precision_loss,
                reason = "8-bit zero points are exact in F32"
            )]
            Ok((value - zero_point as f32) * scale)
        })
        .collect::<Result<Vec<_>>>()?;
    Tensor::from_vec(Device::Cpu, shape, output)
}
