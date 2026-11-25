// `singe inspect` and `singe tasks`: what a model would serve, and which endpoints serve each task.

use std::{error::Error, path::Path};

use anyhow::{Result, bail};
use singe_model::{
    Source,
    function::{
        ExportManifest, TensorSpec,
        onnx::{DeclaredValues, declared_values},
    },
    model,
    task::Task,
};
use singe_server::registry::endpoints;

pub fn inspect(name: &str) -> Result<()> {
    let path = Path::new(name);
    if path.join(ExportManifest::FILE).is_file() {
        return program(path);
    }

    if path.is_file()
        && path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("onnx"))
    {
        return onnx(path);
    }

    if path.join("model.onnx").is_file() {
        return onnx(&path.join("model.onnx"));
    }

    if path.join("model.json").is_file() {
        bail!(
            "{} holds a CUDA program without {}, as `singe export` writes for generation loops and autoencoders, which inspect does not describe yet",
            path.display(),
            ExportManifest::FILE
        );
    }

    let source = Source::find(name)?;
    println!("directory: {}", source.directory().display());
    println!("format: {:?}", source.format());
    println!("kind: {}", source.kind());
    let classes = source.classes()?;
    if !classes.is_empty() {
        println!("classes: {}", classes.join(", "));
    }

    match model::default_head(&source) {
        Ok(head) => println!("head: {}", head.id()),
        Err(error) => println!("head: none ({})", report(&error)),
    }

    match Task::default_for(&source) {
        Ok(task) => {
            println!("task: {}", task.id());
            println!("endpoints: {}", endpoints(task.id()).join(", "));
        }
        Err(error) => println!("task: none ({})", report(&error)),
    }

    if let Some(length) = source.context_length()? {
        println!("context length: {length}");
    }

    if let Some(data_type) = source.stored_data_type()? {
        println!("stored data type: {data_type}");
    }
    Ok(())
}

fn program(directory: &Path) -> Result<()> {
    let manifest = ExportManifest::read(directory)?;
    println!("exported from: {}", manifest.source);
    if let Some(task) = &manifest.task {
        println!("task: {task}");
        println!("endpoints: {}", endpoints(task).join(", "));
    }

    println!("architecture: {}", manifest.architecture);
    if manifest.libraries.is_empty() {
        println!("libraries: none");
    } else {
        println!("libraries: {}", manifest.libraries.join(", "));
    }

    println!("tuned: {}", manifest.tuned);
    if let Some(step) = manifest.steps.first() {
        let (padded, exact) = step
            .dimensions
            .keys()
            .partition::<Vec<_>, _>(|name| step.padded.contains(*name));
        let list = |names: Vec<&String>| {
            if names.is_empty() {
                "none".to_owned()
            } else {
                names
                    .iter()
                    .map(|name| name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        };
        println!("padded up to a step: {}", list(padded));
        println!("matching a step exactly: {}", list(exact));
    }

    for step in &manifest.steps {
        println!("step {}:", step.name);
        for spec in &step.inputs {
            println!("  input {}", describe(spec));
        }
        for spec in &step.outputs {
            println!("  output {}", describe(spec));
        }
    }
    Ok(())
}

fn describe(spec: &TensorSpec) -> String {
    let axes = spec
        .shape
        .iter()
        .zip(&spec.axes)
        .map(|(size, axis)| {
            axis.as_ref()
                .map_or_else(|| size.to_string(), |name| format!("{name}={size}"))
        })
        .collect::<Vec<_>>();
    format!("{} {} [{}]", spec.name, spec.data_type, axes.join(", "))
}

fn onnx(path: &Path) -> Result<()> {
    let DeclaredValues { inputs, outputs } = declared_values(path)?;
    for (kind, values) in [("input", inputs), ("output", outputs)] {
        for value in values {
            let data_type = value.data_type.map_or_else(
                || "unknown".to_owned(),
                |data_type| format!("{data_type:?}"),
            );
            let axes = value
                .axes
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>();
            println!("{kind} {} {data_type} [{}]", value.name, axes.join(", "));
        }
    }
    Ok(())
}

pub fn tasks() {
    for task in Task::ALL {
        if task.heads().is_empty() {
            println!("{}: no pipeline yet", task.id());
        } else {
            println!("{}: {}", task.id(), endpoints(task.id()).join(", "));
        }
    }
}

fn report(error: &dyn Error) -> String {
    let mut message = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        message.push_str(": ");
        message.push_str(&cause.to_string());
        source = cause.source();
    }
    message
}
