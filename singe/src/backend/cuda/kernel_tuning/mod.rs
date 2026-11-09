//! Schedules for generated kernels chosen by measuring them.
//!
//! [`tune`] times a kernel's candidate schedules on buffers shaped for its launch, after checking each candidate's outputs against the default schedule's, and `tuning::tune_graph` keeps the winners in records that later compiling and exporting read.
//! A [`KernelRecorder`](crate::backend::cuda::KernelRecorder) collects the kernels planned under its context, so an offline tuner can search all of them, and [`PlannedKernel::emit`] returns the artifact each one compiles from.

mod measure;
mod search;

use singe_codegen::{abi::Bindings, ir::Kernel, schedule::Schedule};
use singe_cuda::architecture::GpuArchitecture;

use crate::{
    backend::cuda::codegen::{KernelArtifact, KernelDescription, emit_kernel},
    error::Result,
};

pub use self::{
    measure::{KernelBench, Timing},
    search::{Outcome, TuneOptions, TuningResult, tune},
};

/// A kernel a process planned: its IR, the schedule its lowering chose, its launch values, and what its producer described.
#[derive(Debug, Clone, PartialEq)]
pub struct PlannedKernel {
    pub kernel: Kernel,
    pub schedule: Schedule,
    pub bindings: Bindings,
    pub description: KernelDescription,
    pub target: Option<GpuArchitecture>,
}

impl PlannedKernel {
    /// Emits this kernel under `schedule`, with the launch values it was planned with, as the runtime would compile it.
    ///
    /// # Errors
    ///
    /// Returns an error if the emitter does not support the kernel under `schedule`.
    pub fn emit(&self, schedule: Schedule) -> Result<KernelArtifact> {
        let artifact = emit_kernel(&self.kernel, schedule, &self.bindings, &self.description)?;
        Ok(KernelArtifact {
            target: self.target,
            ..artifact
        })
    }
}

#[cfg(test)]
mod tests;
