//! Workspace maintenance commands for Singe.
//!
//! The xtask binary injects generated rustdoc into checked-in bindings and refreshes
//! generated PTX instruction metadata.

use std::env;

use anyhow::{Result, bail};

use xtask::document::document;
use xtask::ptx::generate_ptx_instruction_parser;

fn main() {
    if let Err(error) = run() {
        eprintln!("{error:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let mut args = env::args().skip(1);
    let Some(command) = args.next() else {
        print_help();
        bail!("missing xtask command");
    };

    match command.as_str() {
        "document" => document(),
        "gen-ptx-instructions" => generate_ptx_instruction_parser(),
        "help" | "--help" | "-h" => {
            print_help();
            Ok(())
        }
        other => bail!("unknown xtask command: {other}"),
    }
}

fn print_help() {
    println!("cargo xtask document");
    println!("cargo xtask gen-ptx-instructions");
}
