// What every verification shares: its cases, the snapshot each reads, running a case through its pipeline and timing it, comparing the outputs with the expected ones, and the report.

mod compare;
mod export;
mod onnx;
mod snapshot;

use std::{
    collections::BTreeMap,
    fmt::{Display, Formatter, Result as FmtResult},
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{Context as _, Result, bail};
use serde::{
    Deserialize, Deserializer, Serialize,
    de::{DeserializeOwned, Error as _},
};
use serde_json::Value;
use singe::{
    DataType, Device,
    backend::cuda::{
        clear_cache,
        export::{Library, compiled::program_target},
    },
};
use singe_model::{
    LoadOptions, Source,
    export::{
        cuda::file::{Build, FileExport, export_snapshot},
        onnx::{self as onnx_export, ExportOptions},
    },
    function::{self, DimensionRange, Shapes, program},
    processor::base64::file_value,
    task::{self, Pipeline, Task},
    tune::{TuneLevel, TuneSettings, Tuning},
};
use singe_testing::{GpuWork, profiled};

// The runs a case is timed over, after one that warms it up.
const TIMED_RUNS: usize = 5;

// The device verifications load their models on, as the references run them on CUDA device 0.
const DEVICE: Device = Device::Cuda(0);

// One model's verification: its directory under `verifications/`, and the function that checks its cases.
pub struct Verification {
    pub name: &'static str,
    pub verify: fn(&mut Session),
}

macro_rules! verifications {
    ($category:literal, [$($model:ident),+ $(,)?]) => {
        pub const VERIFICATIONS: &[$crate::support::Verification] = &[$(
            $crate::support::Verification {
                name: concat!($category, "/", stringify!($model)),
                verify: $model::verify,
            }
        ),+];
    };
}

pub(crate) use verifications;

pub trait Runner: Sized {
    const TASK: &'static str;

    type Input: DeserializeOwned;

    type Parameters: DeserializeOwned + Default;

    type Output: Serialize;

    fn load(source: &Source, options: LoadOptions) -> Result<Self>;

    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>>;
}

impl<P: Pipeline> Runner for P {
    const TASK: &'static str = P::TASK.id();

    type Input = P::Input;

    type Parameters = P::Parameters;

    type Output = P::Output;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(<P as Pipeline>::load(source, options)?)
    }

    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>> {
        Ok(<P as Pipeline>::run(self, inputs, parameters)?)
    }
}

// A case of `cases.json`: the snapshot it reads, by the variable naming it, the task, the type its weights load in, its inputs and parameters, and the reference's output for each input.
#[derive(Debug, Deserialize)]
struct Case {
    name: String,
    snapshot: String,
    task: String,
    // A case in half precision matches looser, by `compare::loose`.
    #[serde(deserialize_with = "data_type")]
    data_type: DataType,
    inputs: Vec<Value>,
    #[serde(default)]
    parameters: Value,
    expected: Vec<Value>,
}

#[derive(Debug, Deserialize)]
struct CaseFile {
    cases: Vec<Case>,
}

// What checking a case found.
enum Outcome {
    Passed,
    Failed,
    Skipped,
}

// The cases of one model, and what checking them found.
pub struct Session {
    name: &'static str,
    // Only cases whose `model/case` labels contain it run.
    filter: String,
    cases: Vec<Case>,
    pub passed: usize,
    pub failed: usize,
    pub skipped: usize,
}

impl Session {
    pub fn open(name: &'static str, filter: &str) -> Result<Self> {
        let path = root().join(name).join("cases.json");
        let text = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        let cases: CaseFile =
            serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
        Ok(Self {
            name,
            filter: filter.to_owned(),
            cases: cases.cases,
            passed: 0,
            failed: 0,
            skipped: 0,
        })
    }

    // Loads `R` from case `name`'s snapshot in the case's data type, as its reference loads it, runs the case through it, compares each output with the expected one, and reports it with the median time of a run.
    pub fn check<R: Runner>(&mut self, name: &str) {
        let label = format!("{}/{name}", self.name);
        self.checked(&label, name, |case, snapshot, label| {
            run::<R>(case, Source::open(snapshot)?, label)
        });
    }

    // Exports case `name`'s snapshot as a CUDA C++ program, builds its runner with `nvcc`, runs it on the case's inputs, and compares what it gives with the reference's outputs, as `export::run` does.
    pub fn check_exported(&mut self, name: &str) {
        let label = format!("{}/{name}/exported", self.name);
        self.checked(&label, name, run_exported);
    }

    // Exports the case's snapshot as a CUDA program with a step per combination of `dimensions`' sizes, a dimension named more than once taking each size, and runs the case through it.
    pub fn check_cuda<R: Runner>(&mut self, name: &str, dimensions: &[(&str, i64)]) {
        let label = format!("{}/{name}/cuda", self.name);
        let mut sizes = BTreeMap::<String, Vec<i64>>::new();
        for &(name, size) in dimensions {
            sizes.entry(name.to_owned()).or_default().push(size);
        }
        let shapes = Shapes {
            dimensions: sizes,
            ..Shapes::default()
        };
        self.checked(&label, name, |case, snapshot, label| {
            run_cuda::<R>(case, snapshot, label, shapes)
        });
    }

    pub fn check_onnx(&mut self, name: &str) {
        let label = format!("{}/{name}/onnx", self.name);
        self.checked(&label, name, run_onnx);
    }

    // Exports case `name`'s snapshot as ONNX with dynamic axes, serves the file with `dimensions` fixed and `ranges` following each input from its least size up, as `singe serve --file --dim --range` does, and runs the case through it.
    pub fn check_served<R: Runner>(
        &mut self,
        name: &str,
        dimensions: &[(&str, i64)],
        ranges: &[(&str, i64)],
    ) {
        let label = format!("{}/{name}/served", self.name);
        let shapes = Shapes {
            dimensions: dimensions
                .iter()
                .map(|&(name, size)| (name.to_owned(), vec![size]))
                .collect(),
            ranges: ranges
                .iter()
                .map(|&(name, min)| (name.to_owned(), DimensionRange { min, max: None }))
                .collect(),
            ..Shapes::default()
        };
        self.checked(&label, name, |case, snapshot, label| {
            run_served::<R>(case, snapshot, label, &shapes)
        });
    }

    // A case whose snapshot is not downloaded is skipped, and one that fails to load or run fails, and the next case runs.
    fn checked(
        &mut self,
        label: &str,
        name: &str,
        check: impl FnOnce(&Case, &Path, &str) -> Result<Outcome>,
    ) {
        // A filter ending in `$` must end the label, so `model/case$` runs the case's own check without its exports.
        let selected = match self.filter.strip_suffix('$') {
            Some(end) => label.ends_with(end),
            None => label.contains(&self.filter),
        };
        if !selected {
            return;
        }

        let outcome = self
            .case(name)
            .and_then(|case| match snapshot(case, label)? {
                Some(snapshot) => check(case, &snapshot, label),
                None => Ok(Outcome::Skipped),
            });
        self.record(label, outcome);
    }

    fn case(&self, name: &str) -> Result<&Case> {
        self.cases
            .iter()
            .find(|case| case.name == name)
            .with_context(|| format!("cases.json has no case {name}"))
    }

    fn record(&mut self, label: &str, outcome: Result<Outcome>) {
        // The allocator keeps freed memory cached, which the next model needs on a small GPU, as `support.py` empties torch's.
        let cleared = clear_cache(DEVICE).context("clear the allocator's cache");
        match outcome.and_then(|outcome| cleared.map(|()| outcome)) {
            Ok(Outcome::Passed) => self.passed += 1,
            Ok(Outcome::Failed) => self.failed += 1,
            Ok(Outcome::Skipped) => self.skipped += 1,
            Err(error) => {
                println!("{label}: FAILED, {error:#}");
                self.failed += 1;
            }
        }
    }
}

fn snapshot(case: &Case, label: &str) -> Result<Option<PathBuf>> {
    let found = snapshot::find(&root(), &case.snapshot)?;
    if found.is_none() {
        println!(
            "{label}: skipped, snapshot {} is not downloaded (`just snapshot {}`)",
            case.snapshot, case.snapshot
        );
    }
    Ok(found)
}

fn verdict(label: &str, matched: bool, details: &[String]) -> Outcome {
    let verdict = if matched { "ok" } else { "MISMATCH" };
    println!("{label}: {verdict}; {}", details.join("; "));
    if matched {
        Outcome::Passed
    } else {
        Outcome::Failed
    }
}

fn run<R: Runner>(case: &Case, source: Source, label: &str) -> Result<Outcome> {
    if case.task != R::TASK {
        bail!("a {} case run as {}", case.task, R::TASK);
    }

    let (pipeline, load_time, load_gpu) =
        profiled(|| R::load(&source, LoadOptions::new(DEVICE, case.data_type)))?;
    let pipeline = pipeline?;
    let inputs = case
        .inputs
        .iter()
        .map(|input| Ok(serde_json::from_value(embedded(input.clone())?)?))
        .collect::<Result<Vec<R::Input>>>()?;
    let parameters = task::parameters::<R::Parameters>(&case.parameters)?;

    // The first run warms the pipeline up, and its outputs are the ones compared.
    let start = Instant::now();
    let outputs = pipeline.run(&inputs, &parameters)?;
    let first = start.elapsed();
    // The timed runs are not profiled, so profiling slows none of them; one more run is, for what the GPU did in a run.
    let mut times = (0..TIMED_RUNS)
        .map(|_| {
            let start = Instant::now();
            pipeline.run(&inputs, &parameters)?;
            Ok(start.elapsed())
        })
        .collect::<Result<Vec<Duration>>>()?;
    times.sort();
    let (ran, _, run_gpu) = profiled(|| pipeline.run(&inputs, &parameters))?;
    ran?;
    let timing = Timing {
        load: load_time,
        load_copies: load_gpu.copies,
        first,
        run: times[TIMED_RUNS / 2],
        run_gpu,
    };
    drop(pipeline);

    let compare = if case.data_type == DataType::F32 {
        compare::outputs
    } else {
        compare::loose
    };
    let mut matched = outputs.len() == case.expected.len();
    let mut details = Vec::new();
    for (want, got) in case.expected.iter().zip(&outputs) {
        let (same, detail) = compare(&case.task, want, &serde_json::to_value(got)?);
        matched &= same;
        details.push(detail);
    }
    if outputs.len() != case.expected.len() {
        details.push(format!(
            "{} outputs for {} expected",
            outputs.len(),
            case.expected.len()
        ));
    }

    let outcome = verdict(label, matched, &details);
    println!("    singe: {timing}");
    Ok(outcome)
}

fn run_exported(case: &Case, snapshot: &Path, label: &str) -> Result<Outcome> {
    let directory = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("exports")
        .join(label.replace('/', "-"));
    let (matched, details) = export::run(case, snapshot, &directory)?;
    Ok(verdict(label, matched, &details))
}

fn run_cuda<R: Runner>(
    case: &Case,
    snapshot: &Path,
    label: &str,
    shapes: Shapes,
) -> Result<Outcome> {
    if case.task != R::TASK {
        bail!("a {} case run as {}", case.task, R::TASK);
    }

    let directory = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("cuda")
        .join(label.replace('/', "-"));
    if directory.exists() {
        fs::remove_dir_all(&directory)?;
    }
    let options = FileExport {
        architecture: program_target(DEVICE).context("a GPU and nvcc to export for")?,
        device: DEVICE,
        shapes,
        libraries: Library::ALL.to_vec(),
        fusion: true,
        state_functions: false,
        tuning: Some(Tuning {
            settings: TuneSettings {
                level: TuneLevel::Quick,
                ..TuneSettings::default()
            },
            ..Tuning::default()
        }),
        build: Build::Library,
    };
    let name = snapshot
        .to_str()
        .context("a snapshot path that is not UTF-8")?;
    export_snapshot(
        name,
        Task::from_id(&case.task)?,
        case.data_type,
        &directory,
        &options,
    )?;
    let source = Source::open(snapshot)?.with_function(program::open(&directory)?);
    run::<R>(case, source, label)
}

fn run_served<R: Runner>(
    case: &Case,
    snapshot: &Path,
    label: &str,
    shapes: &Shapes,
) -> Result<Outcome> {
    if case.task != R::TASK {
        bail!("a {} case run as {}", case.task, R::TASK);
    }

    let directory = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("served")
        .join(label.replace('/', "-"));
    if directory.exists() {
        fs::remove_dir_all(&directory)?;
    }
    let source = Source::open(snapshot)?;
    let options = ExportOptions {
        data_type: case.data_type,
        ..ExportOptions::default()
    };
    let (program, state) = onnx_export::program(&source, Task::from_id(&case.task)?, &options)?;
    onnx_export::write(&program, &state, &directory, options.weights)?;
    let function = function::onnx::open(&directory.join("model.onnx"), shapes, DEVICE, None)?;
    run::<R>(case, source.with_function(function), label)
}

fn run_onnx(case: &Case, snapshot: &Path, label: &str) -> Result<Outcome> {
    let directory = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("onnx")
        .join(label.replace('/', "-"));
    let (matched, details) = onnx::run(case, snapshot, &directory)?;
    Ok(verdict(label, matched, &details))
}

fn data_type<'de, D: Deserializer<'de>>(deserializer: D) -> Result<DataType, D::Error> {
    match String::deserialize(deserializer)?.as_str() {
        "f32" => Ok(DataType::F32),
        "f16" => Ok(DataType::F16),
        "bf16" => Ok(DataType::Bf16),
        other => Err(D::Error::custom(format!("unknown data type {other}"))),
    }
}

// The directory verifications live in, which cases name media files relative to.
fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("verifications")
}

// Rewrites each string naming a file under `media/` to its path, as cases name the clips and images they read.
fn media(value: Value) -> Value {
    match value {
        Value::String(text) if text.starts_with("media/") => {
            Value::String(root().join(text).display().to_string())
        }
        Value::Array(items) => Value::Array(items.into_iter().map(media).collect()),
        Value::Object(fields) => Value::Object(
            fields
                .into_iter()
                .map(|(key, value)| (key, media(value)))
                .collect(),
        ),
        other => other,
    }
}

fn embedded(value: Value) -> Result<Value> {
    Ok(match value {
        Value::String(text) if text.starts_with("media/") => file_value(&root().join(text))?,
        Value::Array(items) => {
            Value::Array(items.into_iter().map(embedded).collect::<Result<_>>()?)
        }
        Value::Object(fields) => Value::Object(
            fields
                .into_iter()
                .map(|(key, value)| Ok((key, embedded(value)?)))
                .collect::<Result<_>>()?,
        ),
        other => other,
    })
}

// Where a case's time went: loading (with its copies to the GPU), the first run (with any compiling and tuning), and the median run, split into the GPU's work and the host's.
struct Timing {
    load: Duration,
    load_copies: Duration,
    first: Duration,
    run: Duration,
    run_gpu: GpuWork,
}

impl Display for Timing {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        let busy = self.run_gpu.busy();
        write!(
            formatter,
            "load {} ({} copying to the GPU), first run {}, run {}: GPU {} in {} kernels, host {}",
            time(self.load),
            time(self.load_copies),
            time(self.first),
            time(self.run),
            time(busy),
            self.run_gpu.launches(),
            time(self.run.saturating_sub(busy)),
        )
    }
}

// A duration in milliseconds below ten seconds, and in seconds above.
fn time(duration: Duration) -> String {
    if duration < Duration::from_secs(10) {
        format!("{:.1} ms", duration.as_secs_f64() * 1e3)
    } else {
        format!("{:.2} s", duration.as_secs_f64())
    }
}
