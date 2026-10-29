//! Workspace maintenance commands for Singe.

mod no_slop;

use std::{env, path::PathBuf, process::ExitCode};

use anyhow::{Result, bail};

use xtask::{
    cutile_dtype_kernels::generate_cutile_dtype_kernels, cutile_fused::generate_cutile_fused,
    document::document,
};

fn main() -> ExitCode {
    match run() {
        Ok(status) => status,
        Err(error) => {
            eprintln!("{error:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<ExitCode> {
    let mut args = env::args_os().skip(1);
    let Some(command) = args.next() else {
        print_help();
        bail!("missing xtask command");
    };

    match command.to_str() {
        Some("no-slop") => return Ok(no_slop::run(&args.collect::<Vec<_>>())),
        Some("document") => {
            let args = args.collect::<Vec<_>>();
            let comments_only = args.iter().any(|arg| arg == "--comments-only");
            let paths = args
                .into_iter()
                .filter(|arg| arg != "--comments-only")
                .map(PathBuf::from)
                .collect::<Vec<_>>();
            document(&paths, comments_only)
        }
        Some("gen-cutile-dtype-kernels") => generate_cutile_dtype_kernels(),
        Some("gen-cutile-fused") => generate_cutile_fused(),
        Some("help" | "--help" | "-h") => {
            print_help();
            Ok(())
        }
        _ => bail!("unknown xtask command: {}", command.to_string_lossy()),
    }?;
    Ok(ExitCode::SUCCESS)
}

fn print_help() {
    println!("cargo xtask no-slop [options]");
    println!("cargo xtask document [--comments-only] [binding-path ...]");
    println!("cargo xtask gen-cutile-dtype-kernels");
    println!("cargo xtask gen-cutile-fused");
}
