use std::{
    mem,
    sync::{Arc, Mutex},
};

use singe_codegen::{abi::Bindings, ir::Kernel, schedule::Schedule};
use singe_cuda::architecture::GpuArchitecture;

use crate::{
    backend::cuda::{
        codegen::{description::KernelDescription, emission::KernelLanguage},
        kernel_tuning::PlannedKernel,
        tuning::{ForcedLibrary, Measurement, TuningRecords, library_key},
    },
    error::{Error, Result},
    graph::ir::{Graph, ValueId},
    utility::Locked,
};

// Where a planned kernel goes, which decides the forms it may take.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KernelDestination {
    // Singe compiles and launches it, with NVRTC or `tileiras`, so any recorded or tuned schedule applies.
    #[default]
    Singe,
    // A program's source, which the user compiles with `nvcc`, so it stays CUDA C++.
    Program,
}

// Everything planning a generated kernel reads besides the kernel itself, so a plan depends on its arguments alone.
// The default emits CUDA C++, tunes nothing, and collects nothing, which is what eager calls and feasibility checks want.
#[derive(Debug, Clone, Default)]
pub struct PlanningContext {
    pub language: KernelLanguage,
    // What the tuning step chose, which planning reads and never adds to.
    pub records: Option<Arc<TuningRecords>>,
    pub recorder: Option<KernelRecorder>,
    // Plans a generated kernel for a convolution that is not depthwise, which a library runs otherwise.
    pub generated_convolutions: bool,
    pub destination: KernelDestination,
    // The architecture kernels are planned and compiled for, or `None` for the device that runs them; a recorded or tuned schedule applies only to the architecture it was measured on.
    pub target: Option<GpuArchitecture>,
}

impl PlanningContext {
    // The records that hold for this context's target, which are none when they were measured on another architecture, which `check_records` refuses first.
    pub(crate) fn records(&self) -> Option<&TuningRecords> {
        self.records.as_deref().filter(|_| self.records_hold())
    }

    // What records say a library should run the node producing `id` with.
    pub fn forced_library(&self, ir: &Graph, id: ValueId) -> Result<Option<ForcedLibrary>> {
        let Some(records) = self
            .records
            .as_ref()
            .filter(|records| !records.is_empty() && self.records_hold())
        else {
            return Ok(None);
        };
        let key = library_key(ir, id)?;
        Ok(records.get(&key).map(|choice| ForcedLibrary {
            records: Arc::clone(records),
            key,
            choice,
        }))
    }

    // Fails when the records were made under another environment than this context plans for, saying what differs, since what they chose may not run here.
    //
    // # Errors
    //
    // Returns an error naming what differs.
    pub(crate) fn check_records(&self) -> Result<()> {
        let measurement = match self.destination {
            KernelDestination::Singe => Measurement::GraphReplay,
            KernelDestination::Program => Measurement::Program,
        };
        let target = self.target.map(|target| target.to_string());
        self.records
            .as_deref()
            .and_then(|records| records.mismatch(target.as_deref(), measurement))
            .map_or_else(
                || Ok(()),
                |what| {
                    Err(Error::invalid_argument(format!(
                        "tuning records were made {what}; tune again here, or plan without records"
                    )))
                },
            )
    }

    // Whether the records were measured on this context's target, the way what it plans runs.
    pub(crate) fn records_hold(&self) -> bool {
        let measurement = match self.destination {
            KernelDestination::Singe => Measurement::GraphReplay,
            KernelDestination::Program => Measurement::Program,
        };
        let target = self.target.map(|target| target.to_string());
        self.records
            .as_deref()
            .is_none_or(|records| records.holds_for(target.as_deref(), measurement))
    }
}

// Collects every distinct kernel planned under the contexts that hold it, with the first description each was planned with.
#[derive(Debug, Clone, Default)]
pub struct KernelRecorder {
    planned: Arc<Mutex<Vec<PlannedKernel>>>,
}

impl KernelRecorder {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    // The kernels collected so far, each kernel and launch once; collecting goes on.
    pub fn take(&self) -> Result<Vec<PlannedKernel>> {
        Ok(mem::take(&mut *self.planned.locked("kernel recorder")?))
    }

    pub(crate) fn note(
        &self,
        kernel: &Kernel,
        schedule: Schedule,
        bindings: &Bindings,
        description: &KernelDescription,
        target: Option<GpuArchitecture>,
    ) -> Result<()> {
        let mut planned = self.planned.locked("kernel recorder")?;
        if !planned
            .iter()
            .any(|entry| entry.kernel == *kernel && entry.bindings == *bindings)
        {
            planned.push(PlannedKernel {
                kernel: kernel.clone(),
                schedule,
                bindings: bindings.clone(),
                description: description.clone(),
                target,
            });
        }

        drop(planned);
        Ok(())
    }
}
