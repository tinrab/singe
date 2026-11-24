// `singe flashpack`: writes a model in Singe's format with each part's state in a flashpack file.

use std::{fs, path::PathBuf};

use anyhow::{Context as _, Result, bail};
use bytesize::ByteSize;
use clap::{Args, Subcommand, ValueEnum};
use singe::{DataType, Device, nn::Checkpoint};
use singe_model::{LoadOptions, Source, format::WeightFormat, head::Head, model, task::Task};

#[derive(Subcommand)]
pub enum FlashpackCommand {
    #[command(
        about = "Loads a model and writes it in Singe's format, with each part's state in a flashpack file."
    )]
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
    #[arg(
        help = "A model directory, or a Hugging Face repository the local cache holds, as `owner/name` or `owner/name@revision`."
    )]
    model: String,
    #[arg(help = "The directory to write, which must not be inside the model's.")]
    output: PathBuf,
    #[arg(
        long,
        help = "The task whose head to keep, or else the head the model's weights hold."
    )]
    task: Option<String>,
    #[arg(
        long = "type",
        value_enum,
        default_value_t,
        help = "The type of the written floats; `auto` keeps BF16 or F16 weights as stored, and writes others as F32."
    )]
    data_type: TypeChoice,
    #[arg(
        long,
        default_value = "cpu",
        value_parser = parse_device,
        help = "Where the weights load while converting: `cpu`, `cuda`, or `cuda:N`."
    )]
    device: Device,
}

#[derive(Clone, Copy, Default, ValueEnum)]
enum TypeChoice {
    #[default]
    Auto,
    F32,
    F16,
    Bf16,
}

fn build(arguments: &BuildArguments) -> Result<()> {
    let source = Source::find(&arguments.model)?;
    let head = head(&source, arguments.task.as_deref())?;
    let data_type = match arguments.data_type {
        TypeChoice::Auto => source.default_data_type()?,
        TypeChoice::F32 => DataType::F32,
        TypeChoice::F16 => DataType::F16,
        TypeChoice::Bf16 => DataType::Bf16,
    };
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

// The head `task`'s pipeline loads that `source`'s model serves, or the model's default head.
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

fn parse_device(device: &str) -> Result<Device> {
    match device {
        "cpu" => Ok(Device::Cpu),
        "cuda" => Ok(Device::Cuda(0)),
        other => match other.strip_prefix("cuda:").map(str::parse) {
            Some(Ok(index)) => Ok(Device::Cuda(index)),
            _ => bail!("device `{other}` is not `cpu`, `cuda`, or `cuda:N`"),
        },
    }
}
