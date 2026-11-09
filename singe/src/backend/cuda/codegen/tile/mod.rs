//! Generated kernels from the kernel IR, written as CUDA Tile IR by `singe-codegen`'s tile emitter and compiled to cubin by the toolkit's `tileiras`.

use std::{
    env, fs,
    path::PathBuf,
    process::{self, Command},
    sync::{
        OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
};

use cutile_ir::bytecode::{BytecodeVersion, write_bytecode_version};
use singe_codegen::{emit::tile, ir::Kernel, schedule::TileSchedule};

use crate::{
    backend::cuda::codegen::kernel::KernelCode,
    error::{CompileError, Result},
};

/// Returns `kernel` emitted as Tile IR bytecode under `schedule`, in the bytecode version of the CUDA 13.4 toolkit.
///
/// # Errors
///
/// Returns an error if the emitter does not support the kernel or schedule, or the bytecode cannot be written.
pub fn emit(kernel: &Kernel, schedule: TileSchedule) -> Result<KernelCode> {
    let module = tile::emit(kernel, schedule)?;
    Ok(KernelCode::TileIr(write_bytecode_version(
        &module,
        BytecodeVersion::V13_4,
    )?))
}

/// Returns the cubin `tileiras` compiles from `bytecode`, kernel `name`'s Tile IR, for `architecture`.
///
/// The bytecode and cubin pass through files in the system's temporary directory, removed afterwards.
///
/// # Errors
///
/// Returns an error if `tileiras` cannot run or rejects the bytecode, or its files cannot be written or read.
pub fn compile(name: &str, bytecode: &[u8], architecture: &str) -> Result<Vec<u8>> {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let directory = env::temp_dir().join(format!("singe-tileiras-{}", process::id()));
    fs::create_dir_all(&directory).map_err(CompileError::TileirasIo)?;
    let stem = format!("{name}-{}", NEXT.fetch_add(1, Ordering::Relaxed));
    let input = directory.join(format!("{stem}.tilebc"));
    let output = directory.join(format!("{stem}.cubin"));
    let result = fs::write(&input, bytecode)
        .and_then(|()| {
            Command::new(tileiras())
                .args(["--gpu-name", architecture, "--opt-level", "3", "-o"])
                .arg(&output)
                .arg(&input)
                .output()
        })
        .map_err(CompileError::TileirasIo)
        .and_then(|run| {
            if run.status.success() {
                fs::read(&output).map_err(CompileError::TileirasIo)
            } else {
                Err(CompileError::Tileiras {
                    kernel: name.to_owned(),
                    message: String::from_utf8_lossy(&run.stderr).trim().to_owned(),
                })
            }
        });
    // The files live in the temporary directory, so a failed removal leaves only litter.
    fs::remove_file(&input).ok();
    fs::remove_file(&output).ok();
    Ok(result?)
}

/// Returns `tileiras --version`'s output, which keys cached cubins.
///
/// # Errors
///
/// Returns an error if `tileiras` cannot run.
pub fn version() -> Result<String> {
    static VERSION: OnceLock<String> = OnceLock::new();
    if let Some(version) = VERSION.get() {
        return Ok(version.clone());
    }

    let run = Command::new(tileiras())
        .arg("--version")
        .output()
        .map_err(CompileError::TileirasIo)?;
    Ok(VERSION
        .get_or_init(|| String::from_utf8_lossy(&run.stdout).trim().to_owned())
        .clone())
}

/// Returns the toolkit's `tileiras`: the one under `CUDA_PATH`, or the one the `PATH` finds.
fn tileiras() -> PathBuf {
    env::var_os("CUDA_PATH").map_or_else(
        || PathBuf::from("tileiras"),
        |root| PathBuf::from(root).join("bin").join("tileiras"),
    )
}

#[cfg(test)]
mod tests;
