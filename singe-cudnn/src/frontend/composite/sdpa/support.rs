use crate::{
    data_type::DataType,
    error::{Error, Result},
    frontend::graph::Graph,
    pointwise::PointwiseMode,
    scalar::ScalarValue,
    tensor::{Shape, TensorId, TensorSpec},
};

pub(super) fn sdpa_mask_scalar(data_type: DataType, value: f32) -> Result<TensorSpec> {
    let tensor = TensorSpec::new(data_type, Shape::contiguous([1, 1, 1, 1])?);
    match data_type {
        DataType::F32 => tensor.with_scalar_value(ScalarValue::F32(value)),
        DataType::F16 => tensor.with_scalar_value(ScalarValue::f16_bits(if value == 0.0 {
            0x0000
        } else {
            0xfc00
        })),
        DataType::BF16 => tensor.with_scalar_value(ScalarValue::bf16_bits(if value == 0.0 {
            0x0000
        } else {
            0xff80
        })),
        _ => TensorSpec::scalar_f32(value),
    }
}

pub(super) fn sdpa_scalar_tensor(data_type: DataType, value: f32) -> Result<TensorSpec> {
    let tensor = TensorSpec::new(data_type, Shape::contiguous([1, 1, 1, 1])?);
    match data_type {
        DataType::F32 => tensor.with_scalar_value(ScalarValue::F32(value)),
        DataType::F16 => {
            tensor.with_scalar_value(ScalarValue::f16_bits(half::f16::from_f32(value).to_bits()))
        }
        DataType::BF16 => tensor.with_scalar_value(ScalarValue::bf16_bits(
            half::bf16::from_f32(value).to_bits(),
        )),
        _ => Err(Error::DescriptorMismatch {
            name: "sdpa scalar data type".into(),
        }),
    }
}

impl Graph {
    pub(super) fn sdpa_effective_scale(
        &mut self,
        scale: TensorId,
        attention_scale: Option<f32>,
    ) -> Result<TensorId> {
        let Some(attention_scale) = attention_scale else {
            return Ok(scale);
        };

        let scale_data_type = self.tensor_config(scale)?.data_type;
        if let Some(scalar_value) = self.tensor_config(scale)?.scalar_value {
            let combined = match scalar_value {
                ScalarValue::F32(value) => Some(ScalarValue::F32(value * attention_scale)),
                ScalarValue::F16(bits) => {
                    let value = half::f16::from_bits(bits).to_f32() * attention_scale;
                    Some(ScalarValue::f16_bits(half::f16::from_f32(value).to_bits()))
                }
                ScalarValue::Bf16(bits) => {
                    let value = half::bf16::from_bits(bits).to_f32() * attention_scale;
                    Some(ScalarValue::bf16_bits(
                        half::bf16::from_f32(value).to_bits(),
                    ))
                }
                _ => None,
            };
            if let Some(combined) = combined {
                return Ok(self.tensor(
                    TensorSpec::new(scale_data_type, Shape::contiguous([1, 1, 1, 1])?)
                        .with_scalar_value(combined)?,
                ));
            }
        }
        let inline_scale = self.tensor(sdpa_scalar_tensor(scale_data_type, attention_scale)?);
        self.pointwise_binary_infer_as(
            scale,
            inline_scale,
            PointwiseMode::Mul,
            self.effective_compute_data_type(DataType::F32),
            scale_data_type,
        )
    }

    pub(super) fn bind_sdpa_inferred_tensor(
        &mut self,
        inferred: TensorId,
        output: TensorId,
        error_name: &str,
    ) -> Result<()> {
        let inferred_tensor = self.tensor_config(inferred)?.clone();
        let output_tensor = self.tensor_config(output)?.clone();
        if inferred_tensor.data_type != output_tensor.data_type {
            return Err(Error::DescriptorMismatch {
                name: error_name.into(),
            });
        }
        if inferred_tensor.shape.dimensions() != output_tensor.shape.dimensions() {
            return Err(Error::DescriptorMismatch {
                name: error_name.into(),
            });
        }
        self.reshape(inferred, output)
    }

    pub(super) fn bind_sdpa_inferred_optional_tensor(
        &mut self,
        inferred: Option<TensorId>,
        output: Option<TensorId>,
        error_name: &str,
    ) -> Result<()> {
        match (inferred, output) {
            (Some(inferred), Some(output)) => {
                self.bind_sdpa_inferred_tensor(inferred, output, error_name)
            }
            (None, None) => Ok(()),
            _ => Err(Error::DescriptorMismatch {
                name: error_name.into(),
            }),
        }
    }
}
