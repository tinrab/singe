use std::{
    collections::HashMap,
    env, fs,
    path::{self, Path},
    process,
};

use singe::{
    DataType, Device, Tensor,
    backend::cuda::{
        export::{
            CudaExport, GpuArchitecture, Library, ProgramValue,
            compiled::{build_written_runner, program_target},
        },
        tuning::{Measurement, TuningRecords},
    },
    onnx::WeightPlacement,
};

use crate::{
    error::{Error, Result},
    export::{
        Compared, compare, matched_lines,
        onnx::{self, ExportOptions},
    },
    function::{
        ExportManifest, Shapes, Step, TensorFunction, TensorSpec,
        onnx::{Imported, import, tune_steps},
        program, step_inputs,
    },
    source::Source,
    task::Task,
    tune::{Tuning, read_records},
};

#[derive(Debug, Clone)]
pub struct FileExport {
    pub architecture: GpuArchitecture,
    pub device: Device,
    pub shapes: Shapes,
    pub libraries: Vec<Library>,
    pub fusion: bool,
    pub state_functions: bool,
    pub tuning: Option<Tuning>,
    pub build: Build,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Build {
    #[default]
    Write,
    Library,
    Check,
}

#[derive(Debug)]
pub struct FileExported {
    pub manifest: ExportManifest,
    pub records: Option<TuningRecords>,
    pub checks: Vec<String>,
}

pub fn export_file(path: &Path, output: &Path, options: &FileExport) -> Result<FileExported> {
    export_imported(path, output, options, path.display().to_string(), None)
}

pub fn export_snapshot(
    name: &str,
    task: Task,
    data_type: DataType,
    output: &Path,
    options: &FileExport,
) -> Result<FileExported> {
    let source = Source::find(name)?;
    // A directory is recorded absolute, so serving the program from elsewhere finds the same snapshot.
    let recorded = if Path::new(name).is_dir() {
        path::absolute(name)
            .map_err(Error::io(Path::new(name)))?
            .to_string_lossy()
            .into_owned()
    } else {
        name.to_owned()
    };
    let scratch = env::temp_dir().join(format!("singe-export-{}", process::id()));
    let written = onnx::program(
        &source,
        task,
        &ExportOptions {
            data_type,
            weights: WeightPlacement::Beside,
            input_shape: None,
        },
    )
    .and_then(|(program, state)| onnx::write(&program, &state, &scratch, WeightPlacement::Beside));
    let exported = written.and_then(|()| {
        export_imported(
            &scratch.join("model.onnx"),
            output,
            options,
            recorded,
            Some(task),
        )
    });
    let removed = fs::remove_dir_all(&scratch).map_err(Error::io(&scratch));
    let exported = exported?;
    removed?;
    Ok(exported)
}

fn export_imported(
    path: &Path,
    output: &Path,
    options: &FileExport,
    source: String,
    task: Option<Task>,
) -> Result<FileExported> {
    let imported = import(path, &options.shapes, options.device)?;
    let mut export = CudaExport::new(options.architecture)
        .with_fusion(options.fusion)
        .with_state_functions(options.state_functions);
    export = if options.libraries.is_empty() {
        export.without_libraries()
    } else {
        export.with_libraries(options.libraries.clone())
    };

    let tune_inputs = options
        .tuning
        .as_ref()
        .and_then(|tuning| tuning.inputs.as_deref());
    let inputs = imported
        .iter()
        .map(|imported| step_inputs(&imported.step.inputs, tune_inputs, options.device))
        .collect::<Result<Vec<_>>>()?;
    let records = options
        .tuning
        .as_ref()
        .map(|tuning| records(&imported, &inputs, &export, tuning, options, output))
        .transpose()?;
    if let Some(records) = &records {
        export = export.with_tuning_records(records.clone());
    }

    fs::create_dir_all(output).map_err(Error::io(output))?;
    let steps = write_program(&imported, &export, output)?;
    let manifest = ExportManifest {
        source,
        task: task.map(|task| task.id().to_owned()),
        architecture: options.architecture.to_string(),
        libraries: options
            .libraries
            .iter()
            .map(|library| library.name().to_owned())
            .collect(),
        tuned: records.is_some(),
        steps,
    };
    manifest.write(output)?;

    let checks = match options.build {
        Build::Write => Vec::new(),
        Build::Library | Build::Check => {
            let function = program::open(output)?;
            if imported.len() == 1 {
                build_written_runner(output, "model")?;
            }

            if options.build == Build::Check {
                let mut comparisons = Vec::new();
                for (imported, inputs) in imported.iter().zip(&inputs) {
                    comparisons.extend(check(imported, inputs, &*function)?);
                }
                matched_lines(comparisons)?
            } else {
                Vec::new()
            }
        }
    };
    Ok(FileExported {
        manifest,
        records,
        checks,
    })
}

fn write_program(imported: &[Imported], export: &CudaExport, output: &Path) -> Result<Vec<Step>> {
    let mut steps = imported
        .iter()
        .map(|imported| imported.step.clone())
        .collect::<Vec<_>>();
    let graphs = imported
        .iter()
        .map(|imported| imported.model.graph())
        .collect::<singe::Result<Vec<_>>>()?;
    if let [graph] = graphs.as_slice() {
        let program = export.export(graph)?;
        order(&mut steps[0], &program.inputs, &program.outputs)?;
        program.write_to(output, "model")?;
        return Ok(steps);
    }

    let named = steps
        .iter()
        .zip(&graphs)
        .map(|(step, graph)| (step.name.as_str(), graph))
        .collect::<Vec<_>>();
    let set = export.export_steps(&named)?;
    for step in &mut steps {
        let entry = set
            .entries
            .iter()
            .find(|entry| entry.name == step.name)
            .ok_or_else(|| Error::invalid(format!("an export without the step {}", step.name)))?;
        order(step, &entry.program.inputs, &entry.program.outputs)?;
    }

    set.write_to(output, "model")?;
    Ok(steps)
}

fn order(step: &mut Step, inputs: &[ProgramValue], outputs: &[ProgramValue]) -> Result<()> {
    if let Some(value) = outputs.iter().find(|value| value.count.is_some()) {
        return Err(Error::unsupported(format!(
            "output {}, whose size depends on the input's values, in an exported step",
            value.graph_name
        )));
    }

    let ordered = |specs: &[TensorSpec], values: &[ProgramValue]| {
        values
            .iter()
            .map(|value| {
                specs
                    .iter()
                    .find(|spec| spec.name == value.graph_name)
                    .cloned()
                    .ok_or_else(|| {
                        Error::invalid(format!(
                            "a program value {}, which the graph lacks",
                            value.graph_name
                        ))
                    })
            })
            .collect::<Result<Vec<_>>>()
    };
    step.inputs = ordered(&step.inputs, inputs)?;
    step.outputs = ordered(&step.outputs, outputs)?;
    Ok(())
}

fn records(
    imported: &[Imported],
    inputs: &[HashMap<String, Tensor>],
    export: &CudaExport,
    tuning: &Tuning,
    options: &FileExport,
    output: &Path,
) -> Result<TuningRecords> {
    let architecture = options.architecture.to_string();
    let records = if let Some(path) = &tuning.records {
        read_records(path, Some(&architecture), Measurement::Program)?
    } else {
        let local = program_target(options.device)
            .ok_or_else(|| Error::invalid("tuning without a GPU and nvcc to measure on"))?;
        if local.to_string() != architecture {
            return Err(Error::invalid(format!(
                "tuning for {architecture} on a {local} GPU, which measures only its own architecture"
            )));
        }

        let tune_options = tuning
            .settings
            .options(export.lowering_policy(), Measurement::Program);
        tune_steps(imported, inputs, options.device, &tune_options)?
    };
    let save = tuning
        .save
        .clone()
        .unwrap_or_else(|| output.join("tuning.json"));
    if let Some(directory) = save.parent() {
        fs::create_dir_all(directory).map_err(Error::io(directory))?;
    }

    records.write(&save)?;
    Ok(records)
}

fn check(
    imported: &Imported,
    inputs: &HashMap<String, Tensor>,
    function: &dyn TensorFunction,
) -> Result<Vec<Compared>> {
    let expected = imported.model.run(inputs)?;
    let got = function.run(inputs, &[])?;
    imported
        .step
        .outputs
        .iter()
        .map(|spec| {
            let (Some(want), Some(have)) = (expected.get(&spec.name), got.get(&spec.name)) else {
                return Err(Error::invalid(format!(
                    "a check of step {} without its output {}",
                    imported.step.name, spec.name
                )));
            };
            compare(&format!("{} {}", imported.step.name, spec.name), want, have)
        })
        .collect()
}
