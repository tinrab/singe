use std::collections::HashMap;

use serde_json::Value;
use singe::{
    DataType, Device, Shape, Tensor,
    nn::{
        Activation, BatchNorm2d, Conv2d, Conv2dConfig, EncoderBlock, Flatten, LayerNorm, Linear,
        Model, Pool2d, Pool2dConfig, Pool2dKind, SelfAttention, Sequential, StateBuilder,
    },
};

pub fn example_model(
    name: &str,
    reference: &Value,
    device: Device,
) -> anyhow::Result<Model<Sequential>> {
    let checkpoint = match name {
        "mlp" => include_bytes!("../../tests/references/layers/mlp.safetensors").as_slice(),
        "cnn" => include_bytes!("../../tests/references/layers/cnn.safetensors").as_slice(),
        "encoder" => include_bytes!("../../tests/references/layers/encoder.safetensors").as_slice(),
        _ => anyhow::bail!("unknown example model `{name}`"),
    };
    model_from_checkpoint(name, reference, device, checkpoint)
}

pub fn model_from_checkpoint(
    name: &str,
    reference: &Value,
    device: Device,
    checkpoint: &[u8],
) -> anyhow::Result<Model<Sequential>> {
    let mut builder = StateBuilder::new();
    let mut keys = HashMap::new();
    let entries = reference["state"]
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("example state schema"))?;
    for (name, specification) in entries {
        let shape = dimensions(specification)?;
        let tensor = Tensor::zeroed_on(device, DataType::F32, Shape::create(shape)?)?;
        let key = if name.contains("running_") {
            builder.buffer(name, tensor)?
        } else {
            builder.parameter(name, tensor)?
        };
        keys.insert(name.clone(), key);
    }

    let key = |name: &str| {
        keys.get(name)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("missing example state `{name}`"))
    };
    let linear = |name: &str| -> anyhow::Result<Linear> {
        Ok(Linear::new(
            key(&format!("{name}.weight"))?,
            Some(key(&format!("{name}.bias"))?),
        ))
    };
    let norm = |name: &str| -> anyhow::Result<LayerNorm> {
        Ok(LayerNorm::new(
            dimensions(&reference["state"][format!("{name}.weight")])?,
            Some(key(&format!("{name}.weight"))?),
            Some(key(&format!("{name}.bias"))?),
            1e-5,
        ))
    };
    let module = match name {
        "mlp" => Sequential::new()
            .with(linear("first")?)
            .with(Activation::Relu)
            .with(linear("second")?),
        "cnn" => Sequential::new()
            .with(Conv2d::new(
                key("conv.weight")?,
                Some(key("conv.bias")?),
                Conv2dConfig::default(),
            ))
            .with(BatchNorm2d::new(
                Some(key("norm.weight")?),
                Some(key("norm.bias")?),
                key("norm.running_mean")?,
                key("norm.running_var")?,
                1e-5,
            ))
            .with(Activation::Relu)
            .with(Pool2d::new(
                Pool2dKind::Average {
                    count_include_pad: true,
                },
                Pool2dConfig {
                    stride: (1, 1),
                    ..Pool2dConfig::new((2, 2))
                },
            ))
            .with(Flatten::new(1, -1))
            .with(linear("head")?),
        "encoder" => Sequential::new().with(EncoderBlock::new(
            SelfAttention::create(
                linear("query")?,
                linear("key")?,
                linear("value")?,
                linear("output")?,
                2,
                4,
            )?,
            norm("norm1")?,
            linear("expand")?,
            linear("contract")?,
            norm("norm2")?,
        )),
        _ => {
            anyhow::bail!("example model must be mlp, cnn, or encoder");
        }
    };
    Ok(Model::new(
        module,
        builder.finish()?.load_safetensors_bytes(checkpoint)?,
    ))
}

pub fn dimensions(value: &Value) -> anyhow::Result<Vec<i64>> {
    value["shape"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("example dimensions"))?
        .iter()
        .map(|dimension| {
            dimension
                .as_i64()
                .ok_or_else(|| anyhow::anyhow!("example dimension"))
        })
        .collect()
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "example values are float32 values stored as f64 JSON numbers"
)]
pub fn values(value: &Value) -> anyhow::Result<Vec<f32>> {
    value["values"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("example values"))?
        .iter()
        .map(|value| {
            value
                .as_f64()
                .map(|value| value as f32)
                .ok_or_else(|| anyhow::anyhow!("example value"))
        })
        .collect()
}

pub fn verify(actual: &Tensor, expected: &Value) -> anyhow::Result<()> {
    let values = values(expected)?;
    let actual_values = actual.to_vec::<f32>()?;
    if actual.shape().dimensions() != dimensions(expected)?
        || actual_values.len() != values.len()
        || actual_values.iter().zip(values).any(|(actual, expected)| {
            !actual.is_finite()
                || !expected.is_finite()
                || (actual - expected).abs() > f32::mul_add(expected.abs(), 2e-5, 2e-5)
        })
    {
        anyhow::bail!("model output differs from the pinned PyTorch reference");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn test_verification_rejects_incomplete_or_nonfinite_reference_values() {
        let actual =
            Tensor::from_vec(Device::Cpu, Shape::create([2]).unwrap(), vec![1_f32, 2.]).unwrap();
        assert!(verify(&actual, &json!({"shape": [2], "values": [1.]})).is_err());
        assert!(verify(&actual, &json!({"shape": [2], "values": [1e100, 2.]})).is_err());
    }
}
