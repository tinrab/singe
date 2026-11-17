mod common;

use std::{env, path::PathBuf};

use singe_cudss::{context::Context, error::Result as CudssResult};

fn main() -> anyhow::Result<()> {
    let Some((_backend, layer_path)) = communication_layer_args() else {
        println!(
            "Example SKIPPED: distributed MG/MN matrices need MPI or NCCL communicators plus a cuDSS communication layer"
        );
        return Ok(());
    };

    let (_cuda_context, _stream, mut context) = common::create_context()?;
    set_communication_layer(&mut context, layer_path.as_ref())?;

    println!(
        "Example SKIPPED: communication layer configured, but this Rust example does not initialize MPI/NCCL communicators"
    );
    Ok(())
}

fn communication_layer_args() -> Option<(String, Option<PathBuf>)> {
    let mut args = env::args_os().skip(1);
    if let Some(backend) = args.next() {
        return Some((
            backend.to_string_lossy().into_owned(),
            args.next().map(Into::into),
        ));
    }

    env::var_os("CUDSS_COMM_BACKEND").map(|backend| {
        (
            backend.to_string_lossy().into_owned(),
            env::var_os("CUDSS_COMM_LIB").map(Into::into),
        )
    })
}

fn set_communication_layer(context: &mut Context, path: Option<&PathBuf>) -> CudssResult<()> {
    match path {
        // SAFETY: the path names the cuDSS communication layer library the user selected for this cuDSS installation.
        Some(path) => unsafe { context.set_communication_layer(path) },
        None => context.set_default_communication_layer(),
    }
}
