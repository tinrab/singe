use std::{
    fmt::{Debug, Formatter, Result as FormatResult},
    path::Path,
    sync::{Arc, Mutex},
};

use singe::{
    Device, Tensor,
    backend::cuda::export::{
        ProgramValue,
        compiled::{CompiledProgram, ProgramInstance},
    },
};

use crate::{
    error::{Error, Result},
    function::{Bucketed, ExportManifest, StepRunner, TensorFunction, TensorSpec},
};

struct ProgramRunner {
    instance: Mutex<ProgramInstance>,
    names: Vec<Option<String>>,
}

impl Debug for ProgramRunner {
    fn fmt(&self, f: &mut Formatter<'_>) -> FormatResult {
        f.debug_struct("ProgramRunner")
            .field("steps", &self.names)
            .finish_non_exhaustive()
    }
}

impl StepRunner for ProgramRunner {
    fn device(&self) -> Device {
        Device::Cuda(0)
    }

    fn run_step(&self, step: usize, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        let instance = self
            .instance
            .lock()
            .map_err(|_| Error::invalid("a program whose last run panicked"))?;
        let outputs = self.names[step].as_deref().map_or_else(
            || instance.run(&inputs, 1),
            |name| instance.run_step(name, &inputs, 1),
        );
        drop(instance);
        Ok(outputs?)
    }
}

pub fn open(directory: &Path) -> Result<Arc<dyn TensorFunction>> {
    let manifest = ExportManifest::read(directory)?;
    let program = CompiledProgram::open(directory.to_owned())?;
    if program.steps().len() != manifest.steps.len() {
        return Err(Error::invalid(format!(
            "a program of {} steps, where {} lists {}",
            program.steps().len(),
            ExportManifest::FILE,
            manifest.steps.len()
        )));
    }

    let single = program.steps().len() == 1;
    let mut names = Vec::new();
    for step in &manifest.steps {
        let name = (!single).then(|| step.name.clone());
        let interface = program
            .steps()
            .iter()
            .find(|interface| interface.name == name.as_deref().unwrap_or_default())
            .ok_or_else(|| Error::invalid(format!("a program without the step {}", step.name)))?;
        if !matches(&step.inputs, &interface.inputs) || !matches(&step.outputs, &interface.outputs)
        {
            return Err(Error::invalid(format!(
                "a program whose step {} differs from what {} lists",
                step.name,
                ExportManifest::FILE
            )));
        }

        names.push(name);
    }

    let runner = ProgramRunner {
        instance: Mutex::new(program.instance()?),
        names,
    };
    Ok(Arc::new(Bucketed::create(manifest.steps, runner)?))
}

fn matches(specs: &[TensorSpec], values: &[ProgramValue]) -> bool {
    specs.len() == values.len()
        && specs.iter().zip(values).all(|(spec, value)| {
            (spec.name.as_str(), spec.data_type, spec.shape.as_slice())
                == (
                    value.graph_name.as_str(),
                    value.data_type,
                    value.layout.dimensions(),
                )
        })
}
