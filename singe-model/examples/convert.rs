// Saves a model in Singe's format: `--snapshot DIR --task ID --directory OUT`, reading the weights a task's pipeline loads, in F32 unless `--type bf16|f16` asks otherwise.
// The saved directory loads with no import rules and runs every task the source ran.

mod support;

use anyhow::{Context as _, Result};
use singe::DataType;
use singe_model::{LoadOptions, Source, model, task::Task};

use crate::support::Arguments;

fn main() -> Result<()> {
    let arguments = Arguments::parse(&[])?;
    let task = Task::from_id(arguments.required("task")?)?;
    let head = task.head().with_context(|| {
        format!(
            "{}'s pipeline loads `diffusers` components itself, not a model's head to convert",
            task.id()
        )
    })?;
    let data_type = arguments.data_type(DataType::F32)?;
    let source = Source::open(&arguments.snapshot)?;
    let model = model::find(&source)?;
    let directory = arguments.required("directory")?;
    model::convert(
        &source,
        head,
        directory,
        LoadOptions::new(arguments.device, data_type),
    )?;
    println!("saved {} in Singe's format to {directory}", model.name);
    Ok(())
}
