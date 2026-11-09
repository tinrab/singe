use crate::{
    backend::registry::{Operation, OperationAttributes, PackedFormat, QuantizationAttributes},
    data_type::DataType,
    device::Device,
    dispatch::call_one,
    error::Result,
    graph::capture,
    tensor::Tensor,
};

impl Tensor {
    /// Affine-quantizes an F32 tensor to `data_type` using `round_ties_even(input / scale) + zero_point`, saturated to the output range.
    /// NaNs map to the saturated zero point; infinities map to the corresponding range endpoint.
    /// The scale must be finite and positive.
    ///
    /// # Errors
    ///
    /// Returns an error if `scale` is invalid, `data_type` is unsupported, or the backend fails.
    pub fn quantize(&self, scale: f32, zero_point: i64, data_type: DataType) -> Result<Self> {
        let (scale, zero_point) = scalar_parameters(self.device(), scale, zero_point)?;
        self.quantize_with(&scale, &zero_point, None, None, data_type)
    }

    /// Affine-quantizes an F32 tensor with an `F32` scale and `I64` zero point per tensor, per channel along `axis`, or per block of `block_size` consecutive positions along `axis`.
    ///
    /// Per-tensor parameters are scalars, per-channel parameters are vectors with one value per position along `axis`, and per-block parameters have the input's shape with `axis` divided by `block_size`.
    ///
    /// # Errors
    ///
    /// Returns an error if a parameter has the wrong shape or data type, a scale is invalid, `data_type` is unsupported, or the backend fails.
    pub fn quantize_with(
        &self,
        scale: &Self,
        zero_point: &Self,
        axis: Option<i64>,
        block_size: Option<i64>,
        data_type: DataType,
    ) -> Result<Self> {
        call_one(
            Operation::Quantize,
            OperationAttributes::Quantization(QuantizationAttributes {
                axis,
                block_size,
                data_type: Some(data_type),
                packed: None,
            }),
            &[self, scale, zero_point],
        )
    }

    /// Dequantizes an I8/U8 tensor to F32 using `(input - zero_point) * scale`.
    ///
    /// # Errors
    ///
    /// Returns an error if `scale` is invalid, the input dtype is unsupported, or the backend fails.
    pub fn dequantize(&self, scale: f32, zero_point: i64) -> Result<Self> {
        let (scale, zero_point) = scalar_parameters(self.device(), scale, zero_point)?;
        self.dequantize_with(&scale, &zero_point, None, None)
    }

    /// Dequantizes an I8/U8 tensor with an `F32` scale and `I64` zero point per tensor, per channel along `axis`, or per block of `block_size` consecutive positions along `axis`, with parameter shapes as for [`Tensor::quantize_with`].
    ///
    /// # Errors
    ///
    /// Returns an error if a parameter has the wrong shape or data type, a scale is invalid, the input dtype is unsupported, or the backend fails.
    pub fn dequantize_with(
        &self,
        scale: &Self,
        zero_point: &Self,
        axis: Option<i64>,
        block_size: Option<i64>,
    ) -> Result<Self> {
        call_one(
            Operation::Dequantize,
            OperationAttributes::Quantization(QuantizationAttributes {
                axis,
                block_size,
                data_type: None,
                packed: None,
            }),
            &[self, scale, zero_point],
        )
    }

    /// Dequantizes an `I8` or `U8` tensor whose bytes each hold two 4-bit values of `format` along the last axis, the low nibble first, into an `F32` tensor with that axis doubled, with an `F32` scale and `I64` zero point per tensor or per channel along `axis` of the unpacked shape.
    ///
    /// # Errors
    ///
    /// Returns an error if a parameter has the wrong shape or data type, a scale is invalid, the input dtype is unsupported, or the backend fails.
    pub fn dequantize_packed(
        &self,
        scale: &Self,
        zero_point: &Self,
        axis: Option<i64>,
        format: PackedFormat,
    ) -> Result<Self> {
        call_one(
            Operation::Dequantize,
            OperationAttributes::Quantization(QuantizationAttributes {
                axis,
                block_size: None,
                data_type: None,
                packed: Some(format),
            }),
            &[self, scale, zero_point],
        )
    }
}

/// Creates scalar scale and zero-point tensors, recorded as literals during graph capture.
pub fn scalar_parameters(device: Device, scale: f32, zero_point: i64) -> Result<(Tensor, Tensor)> {
    Ok((
        capture::literal(device, |device| Tensor::scalar(device, scale))?,
        capture::literal(device, |device| Tensor::scalar(device, zero_point))?,
    ))
}
