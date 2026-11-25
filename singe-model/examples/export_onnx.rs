//! Exports a model as an ONNX file for one task, `model.onnx`, which ONNX Runtime runs: `--snapshot DIR --directory OUT`, `--task ID` for other than the task the snapshot's weights default to, `--type f16|bf16` for other than F32 weights, and `--weights inline|beside` to keep the weights in `model.onnx` or put them in `model.onnx_data` whatever their size, and `--shape 1,3,800,800` to export at one input shape with every axis fixed, for a model whose sizes do not follow its input at run time.
//! Exporting builds the model on metadata, so it runs no kernel and needs no GPU.
//! The export holds the model alone; copy the snapshot's tokenizer or processor files beside it to preprocess its inputs.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use singe::{DataType, onnx::WeightPlacement};
use singe_model::{
    Source,
    export::onnx::{ExportOptions, export},
    task::Task,
};

#[derive(Parser)]
struct Arguments {
    #[arg(long)]
    snapshot: PathBuf,
    #[arg(long)]
    directory: PathBuf,
    #[arg(long)]
    task: Option<String>,
    #[arg(long = "type", default_value = "f32")]
    data_type: String,
    #[arg(long)]
    weights: Option<String>,
    #[arg(long, value_delimiter = ',')]
    shape: Option<Vec<i64>>,
}

fn main() -> Result<()> {
    let arguments = Arguments::parse();
    let source = Source::open(&arguments.snapshot)?;
    let task = match &arguments.task {
        Some(task) => Task::from_id(task)?,
        None => Task::default_for(&source)?,
    };
    let data_type = match arguments.data_type.as_str() {
        "f32" => DataType::F32,
        "f16" => DataType::F16,
        "bf16" => DataType::Bf16,
        other => anyhow::bail!("weights of type {other}; name f32, f16, or bf16"),
    };
    let weights = match arguments.weights.as_deref() {
        None => WeightPlacement::Automatic,
        Some("inline") => WeightPlacement::Inline,
        Some("beside") => WeightPlacement::Beside,
        Some(other) => anyhow::bail!("weights placed {other}; name inline or beside"),
    };
    let options = ExportOptions {
        data_type,
        weights,
        input_shape: arguments.shape,
    };
    export(&source, task, &arguments.directory, &options)?;
    println!(
        "wrote model.onnx for {} to {}",
        task.id(),
        arguments.directory.display()
    );
    Ok(())
}
