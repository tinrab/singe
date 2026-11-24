//! Saves a model in Singe's format: `--snapshot DIR --task ID --directory OUT`, reading the weights a task's pipeline loads, in F32 unless `--type bf16|f16` asks otherwise.
//! The saved directory records the head that pipeline loads, loads with no import rules, and runs `--task` and the other tasks of that head.

mod support;

use std::path::Path;

use anyhow::{Context as _, Result};
use singe::DataType;
use singe_model::{LoadOptions, Source, format::WeightFormat, model, task::Task};

use crate::support::Arguments;

fn main() -> Result<()> {
    let arguments = Arguments::parse(&[])?;
    let task = Task::from_id(arguments.required("task")?)?;
    let data_type = arguments.data_type(DataType::F32)?;
    let source = Source::open(arguments.snapshot())?;
    let model = model::find(&source)?;
    let head = task
        .heads()
        .iter()
        .copied()
        .find(|head| model.heads().any(|served| served == *head))
        .with_context(|| {
            format!(
                "{} serves none of the heads {}'s pipeline loads",
                model.name,
                task.id()
            )
        })?;
    let directory = arguments.required("directory")?;
    model::convert(
        &source,
        head,
        Path::new(directory),
        LoadOptions::new(arguments.device(), data_type),
        WeightFormat::Safetensors,
    )?;
    println!("saved {} in Singe's format to {directory}", model.name);
    Ok(())
}
