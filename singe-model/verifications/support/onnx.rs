// ONNX exports checked against the reference library: a case's snapshot exported as `model.onnx` for its task, then run by `onnx_check.py` in ONNX Runtime beside the reference library's model on the case's inputs, at shapes other than the one the export was built at.
// When an output disagrees, the program is built again at the shape of the inputs it disagreed on, which run through the native program and through the program's export in Singe's importer, and the first value where they differ is reported, as PyTorch's `_VerificationInterpreter` finds it.

use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context as _, Result, bail, ensure};
use serde::Deserialize;
use serde_json::Value;
use singe::{
    DataType, Device, Shape, Tensor,
    onnx::{WeightPlacement, first_divergence},
};
use singe_model::{
    Source,
    export::{
        onnx::{ExportOptions, program, write},
        rewrite::rewrite,
    },
    function::{
        Shapes,
        onnx::{DeclaredAxis, declared_values},
    },
    task::Task,
};

use crate::support::{Case, media, root};

#[derive(Deserialize)]
struct Report {
    matched: bool,
    runtime: String,
    details: Vec<String>,
    mismatch: Option<PathBuf>,
}

#[derive(Deserialize)]
struct Feed {
    dtype: String,
    shape: Vec<i64>,
    values: Vec<f64>,
}

pub fn run(
    case: &Case,
    snapshot: &Path,
    directory: &Path,
    input_shape: Option<Vec<i64>>,
) -> Result<(bool, Vec<String>)> {
    if directory.exists() {
        fs::remove_dir_all(directory)?;
    }
    let source = Source::open(snapshot)?;
    let task = Task::from_id(&case.task)?;
    let options = ExportOptions {
        data_type: case.data_type,
        input_shape,
        ..ExportOptions::default()
    };
    let (exported, state) = program(&source, task, &options)?;
    write(&exported, &state, directory, options.weights)?;
    let inputs = directory.join("inputs.json");
    fs::write(
        &inputs,
        serde_json::to_vec(&media(Value::Array(case.inputs.clone())))?,
    )?;
    let (data_type, tolerance) = match case.data_type {
        DataType::F16 => ("f16", 1e-2),
        DataType::Bf16 => ("bf16", 5e-2),
        _ => ("f32", 1e-4),
    };
    let path = directory.join("model.onnx");
    let report = run_onnx_check(&path, snapshot, case, &inputs, data_type)?;
    let mut matched = report.matched;
    let mut details = vec![format!("ran in {}", report.runtime)];
    details.extend(report.details);
    // The export rewritten through Singe's importer and exporter, as `singe export --target onnx --file` writes it, runs against the same reference.
    for (label, shapes) in rewrites(&path)? {
        let rewritten = directory.join(format!("rewritten-{label}"));
        rewrite(&path, &rewritten, &shapes, WeightPlacement::Automatic)?;
        let report = run_onnx_check(
            &rewritten.join("model.onnx"),
            snapshot,
            case,
            &inputs,
            data_type,
        )?;
        matched &= report.matched;
        if report.matched {
            details.push(format!("rewritten {label}: ok"));
        } else {
            details.push(format!("rewritten {label}: {}", report.details.join("; ")));
        }
    }
    if let Some(mismatch) = &report.mismatch {
        let feeds = feeds(mismatch)?;
        // Singe's importer runs a model at the shapes it was exported at, so the program is built again at the batch's.
        let input_shape = [
            "input_ids",
            "pixel_values",
            "input_values",
            "input_features",
        ]
        .iter()
        .find_map(|name| feeds.get(*name))
        .map(|tensor| tensor.shape().dimensions().to_vec());
        let (program, state) = program(
            &source,
            task,
            &ExportOptions {
                input_shape,
                ..options
            },
        )?;
        details.push(
            match first_divergence(&program, state.snapshot(), &feeds, tolerance)? {
                Some(divergence) => format!(
                    "first value where the export differs from the native program: {}, from {}, {}",
                    divergence.name,
                    divergence.operation,
                    divergence.difference.map_or_else(
                        || "of another shape".to_owned(),
                        |difference| format!(
                            "max difference {difference:.2e} of {:.2e} allowed",
                            divergence.allowed
                        ),
                    )
                ),
                None => "the export agrees with the native program at every value, so the native program differs from the reference library".to_owned(),
            },
        );
    }
    Ok((matched, details))
}

fn run_onnx_check(
    path: &Path,
    snapshot: &Path,
    case: &Case,
    inputs: &Path,
    data_type: &str,
) -> Result<Report> {
    let output = Command::new("uv")
        .args(["run", "--locked", "python"])
        .arg(root().join("onnx_check.py"))
        .arg(path)
        .arg(snapshot)
        .arg(&case.task)
        .arg(inputs)
        .arg(data_type)
        .current_dir(root().join("../.."))
        .output()
        .context("run onnx_check.py")?;
    ensure!(
        output.status.success(),
        "onnx_check.py failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout)
        .with_context(|| format!("parse {}", String::from_utf8_lossy(&output.stdout)))
}

fn rewrites(path: &Path) -> Result<Vec<(&'static str, Shapes)>> {
    // The axes left as the export declares them, and, for an export with dynamic axes, each fixed at a size `onnx_check.py` fits the case's inputs to.
    let mut dimensions = BTreeMap::new();
    for input in &declared_values(path)?.inputs {
        let named = input.axes.iter().filter_map(|axis| match axis {
            DeclaredAxis::Named(name) => Some(name),
            DeclaredAxis::Size(_) | DeclaredAxis::Unknown => None,
        });
        for name in named {
            let size = match (name.as_str(), input.name.as_str()) {
                ("batch_size", _) => 1,
                ("sequence_length", "input_values") => 80_000,
                ("sequence_length", _) => 32,
                _ => bail!("dimension {name} of {}, which no rewrite fixes", input.name),
            };
            dimensions.entry(name.clone()).or_insert_with(|| vec![size]);
        }
    }

    let mut rewrites = vec![("as exported", Shapes::default())];
    if !dimensions.is_empty() {
        rewrites.push((
            "at fixed shapes",
            Shapes {
                dimensions,
                inputs: BTreeMap::new(),
            },
        ));
    }
    Ok(rewrites)
}

fn feeds(path: &Path) -> Result<HashMap<String, Tensor>> {
    let feeds: HashMap<String, Feed> = serde_json::from_slice(&fs::read(path)?)?;
    feeds
        .into_iter()
        .map(|(name, feed)| {
            let shape = Shape::create(feed.shape)?;
            let tensor = match feed.dtype.as_str() {
                "float32" => Tensor::from_vec(
                    Device::Cpu,
                    shape,
                    feed.values
                        .iter()
                        .map(|&value| value as f32)
                        .collect::<Vec<_>>(),
                )?,
                "int64" => Tensor::from_vec(
                    Device::Cpu,
                    shape,
                    feed.values
                        .iter()
                        .map(|&value| value as i64)
                        .collect::<Vec<_>>(),
                )?,
                other => bail!("an input of type {other}"),
            };
            Ok((name, tensor))
        })
        .collect()
}
