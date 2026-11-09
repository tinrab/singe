//! Captures once and reuses an MLP with completed, independent outputs.

use std::{collections::HashMap, env};

use singe::{
    Device, Result as SingeResult, Shape, Tensor,
    graph::{CaptureBindings, Graph, GraphPrepareOptions},
};

fn tensor(dimensions: &[i64], values: &[f32]) -> SingeResult<Tensor> {
    Tensor::from_vec(
        Device::Cpu,
        Shape::create(dimensions.to_vec())?,
        values.to_vec(),
    )
}

fn main() -> anyhow::Result<()> {
    let device = if env::args().any(|argument| argument == "--cuda") {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };
    let weight = tensor(&[3, 2], &[1., -1., 2., 1., 0.5, 2.])?;
    let bias = tensor(&[2], &[1., -2.])?;
    let (captured, _) = Graph::capture_with_bindings(
        [("x", tensor(&[2, 3], &[1., 2., 3., 4., 5., 6.])?)],
        CaptureBindings::new()
            .with_parameter("weight", weight.clone())
            .with_buffer("bias", bias.clone()),
        |inputs| {
            let hidden = inputs[0].matmul(&weight)?.add(&bias)?.relu()?;
            Ok(vec![hidden.transpose(0, 1)?.reshape(vec![4])?])
        },
    )?;
    let mut executor = captured.prepare(GraphPrepareOptions::new(device))?;
    for values in [[1., 2., 3., 4., 5., 6.], [-1., 2., -3., 4., -5., 6.]] {
        let input = tensor(&[2, 3], &values)?.to_device(device)?;
        let outputs = executor.run(&HashMap::from([("x".into(), input)]))?;
        println!("{:?}", outputs[0].to_vec::<f32>()?);
    }

    println!("{}", executor.dump()?);
    Ok(())
}
