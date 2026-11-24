//! Exports a model as CUDA C++ programs for one task that run with nothing of Singe: `--snapshot DIR --directory OUT`, and `--task ID` for other than the task the snapshot's weights default to.
//!
//! A generating model writes `model_generate.cu` beside its program set, a transcriber's reading a file of F32 samples, and any other model `model_run.cu`, which maps input files to output files; `model.json` says how to build each.
//! `--architecture sm_120` for other than this machine's GPU, `--library NAME` for each library a program may call, every one if none is named, `--type f32|f16` for other than BF16, and the sizes a program fixes: `--capacity` tokens, `--rows` sequences at once, `--seconds` of audio, `--length` tokens of a sentence, `--image-size` of an autoencoder's square image, and `--image-patches`, for `--task image-text-to-text`, the most patches of one, each 16 pixels square.
//! A generating model's loop takes the most likely token unless `--temperature`, `--top-k`, or `--top-p` asks it to sample, and its `--seed` then starts the draws.

mod support;

use anyhow::{Context, Result};
use singe::{DataType, backend::cuda::export::compiled::program_target, tensor::SamplingOptions};
use singe_model::{
    Source,
    export::{ExportOptions, export},
    task::Task,
};

use crate::support::Arguments;

fn main() -> Result<()> {
    let arguments = Arguments::parse(&[])?;
    let directory = arguments.required("directory")?;
    let architecture = match arguments.value("architecture") {
        Some(architecture) => architecture.parse()?,
        None => program_target(arguments.device()).context(
            "name an --architecture, since this machine has no GPU and nvcc to read it from",
        )?,
    };
    let mut options = ExportOptions::new(architecture);
    options.device = arguments.device();
    options.libraries = arguments.libraries()?;
    options.data_type = arguments.data_type(DataType::Bf16)?;
    if let Some(rows) = arguments.parsed("rows")? {
        options.rows = rows;
    }

    let temperature = arguments.parsed("temperature")?;
    let top_k = arguments.parsed("top-k")?;
    let top_p = arguments.parsed("top-p")?;
    if temperature.is_some() || top_k.is_some() || top_p.is_some() {
        options.sampling = Some(
            SamplingOptions::new()
                .with_temperature(temperature.unwrap_or(1.0))
                .with_top_k(top_k)
                .with_top_p(top_p.unwrap_or(1.0)),
        );
    }

    for (name, field) in [
        ("capacity", &mut options.capacity),
        ("seconds", &mut options.audio_seconds),
        ("length", &mut options.sequence_length),
        ("image-size", &mut options.image_size),
    ] {
        if let Some(value) = arguments.parsed(name)? {
            *field = value;
        }
    }

    options.image_patches = arguments.parsed("image-patches")?;
    let source = Source::open(arguments.snapshot())?;
    let task = match arguments.value("task") {
        Some(task) => Task::from_id(task)?,
        None => Task::default_for(&source)?,
    };
    let exported = export(&source, task, directory, &options)?;
    println!(
        "wrote {exported:?} programs for {} to {directory}; model.json says how to build them",
        task.id()
    );
    Ok(())
}
