// Runs any task's pipeline on JSON inputs and prints one JSON output per line, as `transformers.pipeline(task)(inputs, **parameters)` returns them.
//
// `--task ID --snapshot DIR` with the inputs as positional JSON values, or `--inputs JSON` for one value or an array; a positional value that is not JSON is read as a string.
// Images and clips are file paths, and chat messages take `{"role", "content"}` with text or `[{"type": "image", "image": PATH}, {"type": "text", "text": …}]`.
// `--parameters JSON` sets the pipeline's parameters, `--type f32|f16|bf16` the weights' type, F32 by default, and `--cpu` runs on the host.
// For example: `--task text-generation --snapshot DIR "The capital of France is" --parameters '{"max_new_tokens": 24}'`.

mod support;

use anyhow::Result;
use serde_json::Value;
use singe::DataType;
use singe_model::{
    LoadOptions, Source,
    task::{Task, run_json},
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
                .positional
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
    let source = Source::open(&arguments.snapshot)?;
    let options = LoadOptions::new(arguments.device, data_type);
    for output in run_json(task, &source, options, &inputs, &parameters)? {
        println!("{output}");
    }
    Ok(())
}
