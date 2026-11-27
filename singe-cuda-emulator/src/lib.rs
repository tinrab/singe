//! Emulated CUDA and NCCL on the CPU.
//!
//! Programs built on `singe-cuda` and `singe-nccl` run unchanged on emulated GPUs, with no GPU, CUDA, or NCCL installed.
//! Call [`install`] before the first CUDA or NCCL call.
//!
//! # Example
//!
//! ```rust,no_run
//! use singe_ptx::emulator::{EmulatorOptions, Topology};
//!
//! fn main() -> anyhow::Result<()> {
//!     singe_cuda_emulator::install(EmulatorOptions {
//!         topology: Topology::node(4),
//!         ..Default::default()
//!     })?;
//!     // singe-cuda and singe-nccl now see four devices in one node.
//!     Ok(())
//! }
//! ```

mod driver;
mod machine;
mod memory;
mod nccl;
mod runtime;

use std::{
    io,
    sync::{Mutex, PoisonError},
};

use singe_cuda_sys::LoadError;
use singe_ptx::emulator::{Emulator, EmulatorOptions, ExecutionError};

use crate::machine::{Machine, State};

/// Why [`install`] failed.
#[derive(Debug, thiserror::Error)]
pub enum InstallError {
    /// A CUDA or NCCL function ran before the emulator was installed, so the real library is already in use.
    #[error("a CUDA or NCCL library is already in use")]
    AlreadyLoaded(#[from] LoadError),
    /// The emulator is already installed with another topology.
    #[error("emulator is already installed with another topology")]
    Installed,
    /// The emulator would not start.
    #[error("failed to start the emulator")]
    Start(#[from] io::Error),
}

/// Starts an emulated machine with `options` and serves CUDA and NCCL from it for the rest of the process.
///
/// Installing again with the same topology does nothing, so every test or example may call it.
/// Raises `options.queued_commands` to at least 4096, since CUDA programs queue work without waiting.
/// Also sets `options.zero_uninitialized_registers`, since NCCL's device code reads registers it never wrote.
///
/// # Errors
///
/// Returns an error if a CUDA or NCCL function already ran, the emulator is installed with another topology, or the emulator would not start.
pub fn install(mut options: EmulatorOptions) -> Result<(), InstallError> {
    static INSTALLING: Mutex<()> = Mutex::new(());
    let _guard = INSTALLING.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(machine) = machine::installed() {
        return if machine.topology == options.topology {
            Ok(())
        } else {
            Err(InstallError::Installed)
        };
    }

    options.queued_commands = options.queued_commands.max(4096);
    options.zero_uninitialized_registers = true;
    let topology = options.topology;
    let device_bytes = options.device_storage_bytes;
    let emulator = Emulator::create(options)?;
    let devices = emulator.devices();
    let count = devices.len();
    singe_cuda_sys::provide_driver(driver::provide)?;
    singe_cuda_sys::provide_runtime(runtime::provide)?;
    singe_nccl_sys::provide(nccl::provide)?;
    machine::install(Machine {
        emulator,
        devices,
        topology,
        device_bytes,
        state: Mutex::new(State::new(count)),
    })
}

/// Returns the installed emulator, for host work such as reading device memory in a check.
#[must_use]
pub fn emulator() -> Option<Emulator> {
    machine::installed().map(|machine| machine.emulator.clone())
}

/// Returns the first execution failure the emulated CUDA calls reported, which CUDA's status codes only summarize.
///
/// A fault stops only the device it happened on, as a sticky error poisons only its CUDA context.
#[must_use]
pub fn last_fault() -> Option<ExecutionError> {
    let machine = machine::installed()?;
    machine
        .emulator
        .fault()
        .or_else(|| machine.state().fault.clone())
}
