//! Runs any task's pipeline on JSON inputs and prints one JSON output per line, as `transformers.pipeline(task)(inputs, **parameters)` returns them.
//!
//! `--task ID --snapshot DIR` with the inputs as positional JSON values, or `--inputs JSON` for one value or an array; a positional value that is not JSON is read as a string.
//! A string naming an existing file or directory is replaced by its contents, as [`file_value`](singe_model::processor::base64::file_value) reads them, so images, clips, and frames are given as paths; chat messages take `{"role", "content"}` with text or `[{"type": "image", "image": PATH}, {"type": "text", "text": …}]`.
//! `--parameters JSON` sets the pipeline's parameters, `--type f32|f16|bf16` the weights' type, F32 by default, and `--cpu` runs on the host.
//! For example: `--task text-generation --snapshot DIR "The capital of France is" --parameters '{"max_new_tokens": 24}'`.

mod support;

use std::path::Path;

use anyhow::Result;
use serde_json::Value;
use singe::DataType;
use singe_model::{
    LoadOptions, Source,
    processor::base64::file_value,
    task::{self, Task},
};

use crate::support::Arguments;

fn main() -> Result<()> {
    let arguments = Arguments::parse(&[])?;
    let task = Task::from_id(arguments.required("task")?)?;
    let data_type = arguments.data_type(DataType::F32)?;
    let inputs = match arguments.value("inputs") {
        Some(inputs) => serde_json::from_str(inputs)?,
        None => Value::Array(
            arguments
                .positional()
                .iter()
                .map(|input| {
                    serde_json::from_str(input).unwrap_or_else(|_| Value::String(input.clone()))
                })
                .collect(),
        ),
    };
    let parameters = arguments
        .value("parameters")
        .map(serde_json::from_str)
        .transpose()?
        .unwrap_or(Value::Null);
    let inputs = files(inputs)?;
    let source = Source::open(arguments.snapshot())?;
    let options = LoadOptions::new(arguments.device(), data_type);
    match task::load(task, &source, options)?.run_json(&inputs, &parameters)? {
        Value::Array(outputs) if inputs.is_array() => {
            for output in outputs {
                println!("{output}");
            }
        }
        output => println!("{output}"),
    }
    Ok(())
}

/// Replaces each string naming an existing file or directory with its contents.
fn files(value: Value) -> Result<Value> {
    Ok(match value {
        Value::String(text) if Path::new(&text).exists() => file_value(Path::new(&text))?,
        Value::Array(items) => Value::Array(items.into_iter().map(files).collect::<Result<_>>()?),
        Value::Object(fields) => Value::Object(
            fields
                .into_iter()
                .map(|(key, value)| Ok((key, files(value)?)))
                .collect::<Result<_>>()?,
        ),
        other => other,
    })
}
