// `singe`: command-line tools for Singe models.

mod arguments;
mod export;
mod flashpack;
mod inspect;
mod serve;

use std::process::ExitCode;

use clap::{Parser, Subcommand};

use crate::{export::ExportArguments, flashpack::FlashpackCommand, serve::ServeArguments};

/// Command-line tools for Singe models.
#[derive(Parser)]
#[command(name = "singe", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Serves models over HTTP, from the command line or a config file.
    Serve(Box<ServeArguments>),
    /// Writes a model as a CUDA C++ program, autotuned with `--tune`, or a snapshot as an ONNX file.
    Export(Box<ExportArguments>),
    /// Describes what a model would serve, without loading its weights.
    Inspect {
        /// A model directory, a Hugging Face repository the local cache holds, an export directory, or an ONNX file.
        model: String,
    },
    /// Lists every task and the endpoints that serve it.
    Tasks,
    /// Writes models with flashpack weights.
    #[command(subcommand)]
    Flashpack(FlashpackCommand),
}

fn main() -> ExitCode {
    let result = match Cli::parse().command {
        Command::Serve(arguments) => serve::serve(*arguments),
        Command::Export(arguments) => export::export(&arguments),
        Command::Inspect { model } => inspect::inspect(&model),
        Command::Tasks => {
            inspect::tasks();
            Ok(())
        }
        Command::Flashpack(command) => command.run(),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}
