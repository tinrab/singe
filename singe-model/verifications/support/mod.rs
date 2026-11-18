// What every verification shares: its cases, the snapshot each reads, running a case through its pipeline and timing it, comparing the outputs with the expected ones, and the report.

mod compare;
mod export;
mod profile;
mod snapshot;

use std::{
    env,
    fmt::{Display, Formatter, Result as FmtResult},
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{Context as _, Result, bail};
use serde::Deserialize;
use serde_json::Value;
use singe::{DataType, Device};
use singe_model::{
    LoadOptions, Source,
    task::{Pipeline, text::sentence_similarity::SentenceSimilarityPipeline},
};

use self::profile::{GpuWork, profiled};

// The runs a case is timed over, after one that warms it up.
const TIMED_RUNS: usize = 5;

// The device verifications load their models on, as the references run them on CUDA device 0.
const DEVICE: Device = Device::Cuda(0);

// One model's verification: its directory under `verifications/`, and the function that checks its cases.
pub struct Verification {
    pub name: &'static str,
    pub verify: fn(&mut Session),
}

// A case of `cases.json`: the snapshot it reads, by the variable naming it, the task, its inputs and parameters, and the reference's output for each input.
#[derive(Debug, Deserialize)]
struct Case {
    name: String,
    snapshot: String,
    task: String,
    inputs: Vec<Value>,
    #[serde(default)]
    parameters: Value,
    expected: Vec<Value>,
    // Whether the case runs in half precision and matches looser, by `compare::loose`.
    #[serde(default)]
    loose: bool,
}

#[derive(Debug, Deserialize)]
struct Cases {
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

// The directory verifications live in, which cases name media files relative to.
fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("verifications")
}

impl Session {
    pub fn open(name: &'static str, filter: &str) -> Result<Self> {
        let path = root().join(name).join("cases.json");
        let text = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        let cases: Cases =
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

    // Loads pipeline `P` in `data_type` from case `name`'s snapshot, as its reference loads it, runs the case through it, compares each output with the expected one, and reports it with the median time of a run.
    // A case whose snapshot is not downloaded is skipped, and one that fails to load or run fails, and the next case runs.
    pub fn check<P: Pipeline>(&mut self, name: &str, data_type: DataType) {
        let label = format!("{}/{name}", self.name);
        if !label.contains(&self.filter) {
            return;
        }

        let outcome = self.run::<P>(name, &label, data_type);
        self.record(&label, outcome);
    }

    // Exports case `name`'s snapshot as a CUDA C++ program, builds its runner with `nvcc`, runs it on the case's inputs, and compares what it gives with the reference's outputs, as `export::run` does.
    pub fn check_exported(&mut self, name: &str) {
        let label = format!("{}/{name}/exported", self.name);
        if !label.contains(&self.filter) {
            return;
        }

        let outcome = self.run_exported(name, &label);
        self.record(&label, outcome);
    }

    // Embeds case `name`'s sentences with a `sentence-transformers` snapshot's pipeline and compares the pooled vectors with the reference's, whose scale a similarity would hide.
    pub fn check_embeddings(&mut self, name: &str) {
        let label = format!("{}/{name}", self.name);
        if !label.contains(&self.filter) {
            return;
        }

        let outcome = self.run_embeddings(name, &label);
        self.record(&label, outcome);
    }

    fn run_embeddings(&self, name: &str, label: &str) -> Result<Outcome> {
        let case = self
            .cases
            .iter()
            .find(|case| case.name == name)
            .with_context(|| format!("cases.json has no case {name}"))?;
        let Some(snapshot) = snapshot::find(&root(), &case.snapshot)? else {
            println!(
                "{label}: skipped, snapshot {} is not downloaded (`just snapshot {}`)",
                case.snapshot, case.snapshot
            );
            return Ok(Outcome::Skipped);
        };

        let pipeline = SentenceSimilarityPipeline::load(
            &Source::open(snapshot)?,
            LoadOptions::new(DEVICE, DataType::F32),
        )?;
        let texts = case
            .inputs
            .iter()
            .map(|input| input.as_str().context("a sentence to embed"))
            .collect::<Result<Vec<_>>>()?;
        let vectors = pipeline.embed(&texts, None)?;
        let width = usize::try_from(vectors.shape().dimensions()[1])?;
        let values = vectors.to_vec::<f32>()?;
        let mut matched = values.len() == width * case.expected.len();
        let mut details = Vec::new();
        for (want, got) in case.expected.iter().zip(values.chunks(width)) {
            let (same, detail) = compare::outputs(&case.task, want, &serde_json::json!(got));
            matched &= same;
            details.push(detail);
        }
        let verdict = if matched { "ok" } else { "MISMATCH" };
        println!("{label}: {verdict}; {}", details.join("; "));
        Ok(if matched {
            Outcome::Passed
        } else {
            Outcome::Failed
        })
    }

    fn record(&mut self, label: &str, outcome: Result<Outcome>) {
        // The allocator keeps freed memory cached, which the next model needs on a small GPU, as `support.py` empties torch's.
        let cleared =
            singe::backend::cuda::clear_cache(DEVICE).context("clear the allocator's cache");
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

    fn run_exported(&self, name: &str, label: &str) -> Result<Outcome> {
        let case = self
            .cases
            .iter()
            .find(|case| case.name == name)
            .with_context(|| format!("cases.json has no case {name}"))?;
        let Some(snapshot) = snapshot::find(&root(), &case.snapshot)? else {
            println!(
                "{label}: skipped, snapshot {} is not downloaded (`just snapshot {}`)",
                case.snapshot, case.snapshot
            );
            return Ok(Outcome::Skipped);
        };

        let directory = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join("exports")
            .join(label.replace('/', "-"));
        let (matched, details) = export::run(case, &snapshot, &directory)?;
        let verdict = if matched { "ok" } else { "MISMATCH" };
        println!("{label}: {verdict}; {}", details.join("; "));
        Ok(if matched {
            Outcome::Passed
        } else {
            Outcome::Failed
        })
    }

    fn run<P: Pipeline>(&self, name: &str, label: &str, data_type: DataType) -> Result<Outcome> {
        let case = self
            .cases
            .iter()
            .find(|case| case.name == name)
            .with_context(|| format!("cases.json has no case {name}"))?;
        if case.task != P::TASK.id() {
            bail!("a {} case run as {}", case.task, P::TASK.id());
        }

        let Some(snapshot) = snapshot::find(&root(), &case.snapshot)? else {
            println!(
                "{label}: skipped, snapshot {} is not downloaded (`just snapshot {}`)",
                case.snapshot, case.snapshot
            );
            return Ok(Outcome::Skipped);
        };

        let source = Source::open(snapshot)?;
        let (pipeline, load_time, load_gpu) =
            profiled(|| Ok(P::load(&source, LoadOptions::new(DEVICE, data_type))?))?;
        let inputs = case
            .inputs
            .iter()
            .map(|input| Ok(serde_json::from_value(media(input.clone()))?))
            .collect::<Result<Vec<P::Input>>>()?;
        let parameters = if case.parameters.is_null() {
            P::Parameters::default()
        } else {
            serde_json::from_value(case.parameters.clone())?
        };

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
        let (_, _, run_gpu) = profiled(|| Ok(pipeline.run(&inputs, &parameters)?))?;
        let timing = Timing {
            load: load_time,
            load_copies: load_gpu.copies,
            first,
            run: times[TIMED_RUNS / 2],
            run_gpu,
        };
        drop(pipeline);

        let mut failed = false;
        let mut details = Vec::new();
        for (want, got) in case.expected.iter().zip(&outputs) {
            let got = serde_json::to_value(got)?;
            let (matched, detail) = if case.loose {
                compare::loose(&case.task, want, &got)
            } else {
                compare::outputs(&case.task, want, &got)
            };
            failed |= !matched;
            details.push(detail);
        }
        if outputs.len() != case.expected.len() {
            failed = true;
            details.push(format!(
                "{} outputs for {} expected",
                outputs.len(),
                case.expected.len()
            ));
        }

        let verdict = if failed { "MISMATCH" } else { "ok" };
        println!(
            "{label}: {verdict}; {}\n    singe: {timing}",
            details.join("; ")
        );
        Ok(if failed {
            Outcome::Failed
        } else {
            Outcome::Passed
        })
    }
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
            self.run_gpu.launches,
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
