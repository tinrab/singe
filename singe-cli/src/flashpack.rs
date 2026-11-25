// `singe flashpack`: writes a model in Singe's format with each part's state in a flashpack file.

use std::{fs, path::PathBuf};

use anyhow::{Context as _, Result};
use bytesize::ByteSize;
use clap::{Args, Subcommand};
use singe::{Device, nn::Checkpoint};
use singe_model::{LoadOptions, Source, format::WeightFormat, head::Head, model, task::Task};
use singe_server::config::parse_device;

use crate::arguments::TypeChoice;

#[derive(Subcommand)]
pub enum FlashpackCommand {
    /// Loads a model and writes it in Singe's format, with each part's state in a flashpack file.
    Build(BuildArguments),
}

impl FlashpackCommand {
    pub fn run(self) -> Result<()> {
        match self {
            Self::Build(arguments) => build(&arguments),
        }
    }
}

#[derive(Args)]
pub struct BuildArguments {
    /// A model directory, or a Hugging Face repository the local cache holds, as `owner/name` or `owner/name@revision`.
    model: String,
    /// The directory to write, which must not be inside the model's.
    output: PathBuf,
    /// The task whose head to keep, or else the head the model's weights hold.
    #[arg(long)]
    task: Option<String>,
    /// The type of the written floats; `auto` keeps BF16 or F16 weights as stored, and writes others as F32.
    #[arg(long = "type", value_enum, default_value_t)]
    data_type: TypeChoice,
    /// Where the weights load while converting: `cpu`, `cuda`, or `cuda:N`.
    #[arg(long, default_value = "cpu", value_parser = parse_device)]
    device: Device,
}

fn build(arguments: &BuildArguments) -> Result<()> {
    let source = Source::find(&arguments.model)?;
    let head = head(&source, arguments.task.as_deref())?;
    let data_type = arguments.data_type.choice().resolve(&source)?;
    let parts = model::convert(
        &source,
        head,
        &arguments.output,
        LoadOptions::new(arguments.device, data_type),
        WeightFormat::Flashpack,
    )?;

    println!(
        "wrote {} as {} in {data_type} to {}",
        model::find(&source)?.name,
        head.id(),
        arguments.output.display()
    );
    for (name, path) in parts {
        let tensors = Checkpoint::open([&path])?.names().len();
        let bytes = fs::metadata(&path)
            .with_context(|| format!("reading {}", path.display()))?
            .len();
        println!("  {name}: {tensors} tensors, {}", ByteSize(bytes));
    }
    Ok(())
}

fn head(source: &Source, task: Option<&str>) -> Result<Head> {
    let Some(task) = task else {
        return Ok(model::default_head(source)?);
    };
    let task = Task::from_id(task)?;
    let model = model::find(source)?;
    task.heads()
        .iter()
        .copied()
        .find(|head| model.heads().any(|served| served == *head))
        .with_context(|| {
            format!(
                "{} serves none of the heads {}'s pipeline loads",
                model.name,
                task.id()
            )
        })
}
