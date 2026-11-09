use std::{collections::HashMap, env, process};

use singe::{
    device::Device,
    error::Error,
    onnx::{Error as OnnxError, OnnxModel, OnnxOptions, OnnxValueInfo, Result},
    tensor::Tensor,
};

fn main() -> Result<()> {
    let path = env::args().nth(1).unwrap_or_else(|| {
        eprintln!(
            "usage: cargo run -p singe --features onnx --example onnx_inference -- <model.onnx>"
        );
        process::exit(2);
    });

    let model = OnnxModel::create(path, OnnxOptions::new())?;
    println!("inputs");
    for input in model.inputs() {
        println!(
            "  {}: {:?} {:?}{}",
            input.name,
            input.data_type,
            input.shape.dimensions(),
            if input.has_initializer {
                " initializer"
            } else {
                ""
            }
        );
    }

    let inputs = zero_runtime_inputs(model.inputs())?;
    let outputs = model.run(&inputs)?;

    println!("outputs");
    for output in model.outputs() {
        let tensor = outputs
            .get(&output.name)
            .ok_or_else(|| OnnxError::Singe(Error::missing_output(output.name.clone())))?;
        println!(
            "  {}: {:?} {:?}",
            output.name,
            tensor.data_type(),
            tensor.shape().dimensions()
        );
    }

    Ok(())
}

fn zero_runtime_inputs(inputs: &[OnnxValueInfo]) -> Result<HashMap<String, Tensor>> {
    inputs
        .iter()
        .filter(|input| !input.has_initializer)
        .map(|input| {
            let tensor = Tensor::zeroed_on(Device::Cpu, input.data_type, &input.shape)?;
            Ok((input.name.clone(), tensor))
        })
        .collect()
}
