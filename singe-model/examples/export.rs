// Exports a snapshot as CUDA C++ programs that run with nothing of Singe: `--snapshot DIR --directory OUT`.
//
// A generating model writes `model_generate.cu` beside its program set, a transcriber's reading a file of F32 samples, and any other model `model_run.cu`, which maps input files to output files; `model.json` says how to build each.
// `--architecture sm_120` for other than this machine's GPU, `--library NAME` for each library a program may call, every one if none is named, `--type f32|f16` for other than BF16, and the sizes a program fixes: `--capacity` tokens, `--seconds` of audio, `--length` tokens of a sentence, and `--image-size`.

mod support;

use anyhow::{Context, Result, bail};
use singe::{DataType, backend::cuda::export::compiled::program_target};
use singe_model::export::{ExportOptions, export_snapshot};

use crate::support::Arguments;

fn main() -> Result<()> {
    let arguments = Arguments::parse(&[])?;
    let directory = arguments.required("directory")?;
    let architecture = match arguments.value("architecture") {
        Some(architecture) => architecture.parse()?,
        None => program_target(arguments.device).context(
            "name an --architecture, since this machine has no GPU and nvcc to read it from",
        )?,
    };
    let mut options = ExportOptions::new(architecture);
    options.device = arguments.device;
    let libraries = arguments.values("library");
    if !libraries.is_empty() {
        options.libraries = libraries
            .into_iter()
            .map(str::parse)
            .collect::<Result<_, _>>()?;
    }

    options.data_type = match arguments.value("type") {
        None | Some("bf16") => DataType::Bf16,
        Some("f16") => DataType::F16,
        Some("f32") => DataType::F32,
        Some(other) => bail!("unknown --type {other}"),
    };
    for (name, field) in [
        ("capacity", &mut options.capacity),
        ("seconds", &mut options.audio_seconds),
        ("length", &mut options.sequence_length),
        ("image-size", &mut options.image_size),
    ] {
        if let Some(value) = arguments.value(name) {
            *field = value.parse()?;
        }
    }

    let exported = export_snapshot(&arguments.snapshot, directory, &options)?;
    println!("wrote {exported:?} programs to {directory}; model.json says how to build them");
    Ok(())
}
