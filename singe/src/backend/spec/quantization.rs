use crate::{
    backend::{registry::PackedFormat, spec::dimension_product_usize},
    data_type::DataType,
    error::{Error, Result},
    shape::{Shape, normalize_dim_usize},
    utility::to_usize,
};

/// How quantization scales and zero points map onto input elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationGranularity {
    /// One scale and zero point for every element.
    Tensor,
    /// One scale and zero point per position along an axis of extent `axis_size`, with `inner_size` elements after it.
    Channel { axis_size: usize, inner_size: usize },
    /// One scale and zero point per `block_size` consecutive positions along an axis, with the parameters shaped like the input with that axis divided by `block_size`.
    Block {
        axis_size: usize,
        inner_size: usize,
        block_size: usize,
    },
}

/// The parameter shape and element mapping of a quantization operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizationProblem {
    /// The shape every scale and zero point tensor must have.
    pub parameter_shape: Shape,
    /// How parameters map onto input elements.
    pub granularity: QuantizationGranularity,
}

impl QuantizationProblem {
    /// Returns the row-major index into the parameters of the input element at row-major index `linear`.
    #[must_use]
    pub const fn parameter_index(&self, linear: usize) -> usize {
        match self.granularity {
            QuantizationGranularity::Tensor => 0,
            QuantizationGranularity::Channel {
                axis_size,
                inner_size,
            } => (linear / inner_size) % axis_size,
            QuantizationGranularity::Block {
                axis_size,
                inner_size,
                block_size,
            } => {
                let outer = linear / (inner_size * axis_size);
                let position = (linear / inner_size) % axis_size;
                (outer * (axis_size / block_size) + position / block_size) * inner_size
                    + linear % inner_size
            }
        }
    }
}

/// Infers the parameter shape of quantizing `input` per tensor, per channel along `axis`, or per block of `block_size` positions along `axis`.
///
/// # Errors
///
/// Returns an error if `block_size` is given without `axis`, is not positive, or does not divide the axis extent, or if `axis` is out of range.
pub fn quantization_problem(
    input: &Shape,
    axis: Option<i64>,
    block_size: Option<i64>,
) -> Result<QuantizationProblem> {
    let Some(axis) = axis else {
        if block_size.is_some() {
            return Err(Error::invalid_argument(
                "block-scaled quantization needs an axis",
            ));
        }
        return Ok(QuantizationProblem {
            parameter_shape: Shape::scalar(),
            granularity: QuantizationGranularity::Tensor,
        });
    };
    let axis = normalize_dim_usize(axis, input.rank())?;
    let dimensions = input.dimensions();
    let extent = dimensions[axis];
    let axis_size = to_usize(extent, "quantization axis extent")?;
    let inner_size =
        dimension_product_usize("quantization inner size", &dimensions[axis + 1..], true)?;
    let Some(block_size) = block_size else {
        return Ok(QuantizationProblem {
            parameter_shape: Shape::create([extent])?,
            granularity: QuantizationGranularity::Channel {
                axis_size,
                inner_size,
            },
        });
    };
    if block_size <= 0 || extent % block_size != 0 {
        return Err(Error::invalid_argument(
            "quantization block size must be positive and divide the axis extent",
        ));
    }

    let mut parameter_dimensions = dimensions.to_vec();
    parameter_dimensions[axis] = extent / block_size;
    Ok(QuantizationProblem {
        parameter_shape: Shape::create(parameter_dimensions)?,
        granularity: QuantizationGranularity::Block {
            axis_size,
            inner_size,
            block_size: to_usize(block_size, "quantization block size")?,
        },
    })
}

/// Returns the shape a dequantization of an `input`-shaped tensor produces: `input` itself, or with its last axis doubled when two 4-bit values pack in each byte.
///
/// # Errors
///
/// Returns an error if a packed input has no axes.
pub fn unpacked_shape(input: &Shape, packed: Option<PackedFormat>) -> Result<Shape> {
    if packed.is_none() {
        return Ok(input.clone());
    }

    let mut dimensions = input.dimensions().to_vec();
    let last = dimensions
        .last_mut()
        .ok_or_else(|| Error::invalid_argument("packed dequantize operand rank"))?;
    *last = last
        .checked_mul(2)
        .ok_or_else(|| Error::out_of_range("packed dequantize extent"))?;
    Shape::create(dimensions)
}

/// Returns the 4-bit value in nibble `nibble`, 0 for the low one, of `byte`, an element of a `container` tensor packed as `format`.
#[must_use]
pub fn unpacked_value(byte: u8, nibble: usize, container: DataType, format: PackedFormat) -> f32 {
    let code = if nibble == 0 { byte & 0xF } else { byte >> 4 };
    match format {
        PackedFormat::Int4 if container == DataType::I8 => {
            f32::from(i8::try_from(code).unwrap_or(0) - if code >= 8 { 16 } else { 0 })
        }
        PackedFormat::Int4 => f32::from(code),
        PackedFormat::Float4E2M1 => {
            let (exponent, mantissa) = ((code >> 1) & 3, code & 1);
            let magnitude = if exponent == 0 {
                0.5 * f32::from(mantissa)
            } else {
                f32::from((2 + mantissa) << (exponent - 1)) * 0.5
            };
            if code & 8 == 0 { magnitude } else { -magnitude }
        }
    }
}
