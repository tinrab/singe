// `singe`: command-line tools for Singe models.

mod flashpack;

use std::process::ExitCode;

use clap::{Parser, Subcommand};

use crate::flashpack::FlashpackCommand;

#[derive(Parser)]
#[command(name = "singe", version, about = "Command-line tools for Singe models")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    #[command(subcommand, about = "Writes models with flashpack weights.")]
    Flashpack(FlashpackCommand),
}

fn main() -> ExitCode {
    let result = match Cli::parse().command {
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
