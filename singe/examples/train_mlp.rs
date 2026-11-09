//! Fits a two-layer MLP to a noisy sine curve with `AdamW`, recomputing the hidden layer through gradient checkpointing.

use singe::{
    DataType, Device, Shape, Tensor, autograd::checkpoint, error::Result, optim::AdamW,
    random::Generator,
};

fn main() -> Result<()> {
    let device = Device::Cpu;
    let generator = Generator::new(7);
    let samples = 64;
    let x = Tensor::from_vec(
        device,
        Shape::create([samples, 1])?,
        (0..samples)
            .map(|index| (index as f32 / samples as f32).mul_add(6.0, -3.0))
            .collect(),
    )?;
    let y = x.sin()?;
    let parameter = |dimensions: Vec<i64>| -> Result<Tensor> {
        let value = Tensor::randn(
            device,
            Shape::create(dimensions)?,
            DataType::F32,
            Some(&generator),
        )?
        .mul_scalar(0.5)?;
        value.set_requires_grad(true)?;
        Ok(value)
    };
    let parameters = vec![
        parameter(vec![1, 32])?,
        parameter(vec![32])?,
        parameter(vec![32, 1])?,
    ];
    let mut optimizer = AdamW::new(parameters.clone(), 0.02);
    for step in 0..=500 {
        let hidden = checkpoint(
            &[x.clone(), parameters[0].clone(), parameters[1].clone()],
            |inputs| Ok(vec![inputs[0].matmul(&inputs[1])?.add(&inputs[2])?.tanh()?]),
        )?;
        let loss = hidden[0]
            .matmul(&parameters[2])?
            .sub(&y)?
            .square()?
            .mean(&[], false)?;
        optimizer.zero_grad();
        loss.backward()?;
        optimizer.step()?;
        if step % 100 == 0 {
            println!("step {step:>3}: loss {:.5}", loss.to_vec::<f32>()?[0]);
        }
    }
    Ok(())
}
