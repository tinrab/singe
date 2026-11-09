//! Generated kernels from the kernel IR, written as CUDA C++ by `singe-codegen`'s emitter and compiled by NVRTC.

use singe_codegen::{
    emit::{IndexWidth, cuda_c},
    ir::Kernel,
    schedule::ThreadSchedule,
};

use crate::{backend::cuda::codegen::kernel::KernelCode, error::Result};

/// Returns `kernel` emitted as CUDA C++ under `schedule`, computing indices in 32 bits when argument sizes `sizes` let every offset fit.
///
/// # Errors
///
/// Returns an error if the emitter does not support the kernel or schedule.
pub fn emit(kernel: &Kernel, schedule: ThreadSchedule, sizes: &[u64]) -> Result<KernelCode> {
    Ok(KernelCode::Source(cuda_c::emit(
        kernel,
        schedule,
        IndexWidth::of(kernel, sizes),
    )?))
}

#[cfg(test)]
mod tests;
