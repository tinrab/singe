mod common;

use std::{env, error::Error, path::PathBuf};

use singe_cudss::{
    context::Context,
    types::{DataParameter, MatrixFormat},
};

fn main() -> Result<(), Box<dyn Error>> {
    let Some((_backend, layer_path)) = communication_layer_args() else {
        println!(
            "Example SKIPPED: distributed MG/MN matrices need MPI or NCCL communicators plus a cuDSS communication layer"
        );
        return Ok(());
    };

    let (_cuda_context, _stream, context) = common::create_context()?;
    set_communication_layer(&context, layer_path.as_ref())?;

    let _distributed_matrix_format = MatrixFormat::DISTRIBUTED;
    let _host_comm_parameter = DataParameter::CommunicationHost;
    let _device_comm_parameter = DataParameter::CommunicationDevice;

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

fn set_communication_layer(
    context: &Context,
    path: Option<&PathBuf>,
) -> singe_cudss::error::Result<()> {
    match path {
        Some(path) => context.set_communication_layer(path),
        None => context.set_default_communication_layer(),
    }
}
