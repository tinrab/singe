//! Load a tiny checkpoint and run the same model eagerly and through preparation.
//! `cargo run -p singe --example reference_models -- --model cnn` selects the CNN; `--model encoder` selects the encoder and `--cuda` selects CUDA.
//! Add `--capture` to record eager execution instead of metadata construction.

mod support;

use std::{collections::HashMap, env};

use serde_json::Value;
use singe::{DataType, Device, Shape, Tensor, graph::GraphPrepareOptions, value::ValueSpec};
use support::models::{dimensions, example_model, values, verify};

fn main() -> anyhow::Result<()> {
    let mut arguments = env::args().skip(1);
    let mut name = "mlp".to_owned();
    let mut device = Device::Cpu;
    let mut capture = false;
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--cuda" => device = Device::Cuda(0),
            "--capture" => capture = true,
            "--model" => {
                name = arguments
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("missing model name"))?;
            }
            _ => {
                anyhow::bail!("unknown argument `{argument}`");
            }
        }
    }

    let references: Value = singe_testing::reference!("layers").resolved();
    let reference = &references["models"][&name];
    let model = example_model(&name, reference, device)?;
    let cases = reference["cases"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("example cases"))?;
    let input = |value: &Value| -> anyhow::Result<Tensor> {
        Ok(Tensor::from_vec(
            device,
            Shape::create(dimensions(value)?)?,
            values(value)?,
        )?)
    };
    let program = if capture {
        model.capture("x", input(&cases[0]["input"])?)?
    } else {
        model.build(
            "x",
            ValueSpec::new(
                DataType::F32,
                Shape::create(dimensions(&cases[0]["input"])?)?,
            ),
        )?
    };
    let mut executor = program.prepare(GraphPrepareOptions::new(device))?;
    for case in cases {
        let input = input(&case["input"])?;
        verify(&model.forward(&input)?, &case["output"])?;
        let output = executor.run(&HashMap::from([("x".into(), input)]))?;
        verify(&output[0], &case["output"])?;
        println!("{name} on {device}: {:?}", output[0].to_vec::<f32>()?);
    }

    println!(
        "analysis builds: {}, completed runs: {}",
        executor.cache_stats().analysis_builds(),
        executor.runs()
    );
    Ok(())
}
