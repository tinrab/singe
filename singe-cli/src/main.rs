// `singe`: command-line tools for Singe models.

mod arguments;
mod flashpack;
mod inspect;
mod serve;

use std::process::ExitCode;

use clap::{Parser, Subcommand};

use crate::{flashpack::FlashpackCommand, serve::ServeArguments};

#[derive(Parser)]
#[command(name = "singe", version, about = "Command-line tools for Singe models")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    #[command(about = "Serves models over HTTP, from the command line or a config file.")]
    Serve(Box<ServeArguments>),
    #[command(about = "Describes what a model would serve, without loading its weights.")]
    Inspect {
        #[arg(help = "A model directory, or a Hugging Face repository the local cache holds.")]
        model: String,
    },
    #[command(about = "Lists every task and the endpoints that serve it.")]
    Tasks,
    #[command(subcommand, about = "Writes models with flashpack weights.")]
    Flashpack(FlashpackCommand),
}

fn main() -> ExitCode {
    let result = match Cli::parse().command {
        Command::Serve(arguments) => serve::serve(*arguments),
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
