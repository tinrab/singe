use std::{
    collections::BTreeMap,
    env, fs,
    path::PathBuf,
    process::Command,
    time::{Duration, Instant},
};

use serde_json::{Value, json};

const WARM_UP: Duration = Duration::from_millis(300);
const MEASURE: Duration = Duration::from_secs(2);
const MIN_SAMPLES: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Summary {
    pub median: Duration,
    pub spread: f64,
    pub samples: usize,
}

impl Summary {
    fn from_samples(mut samples: Vec<Duration>) -> Self {
        samples.sort_unstable();
        let at = |share: usize| samples[(samples.len() - 1) * share / 100];
        let median = at(50);
        Self {
            median,
            spread: at(90).saturating_sub(at(10)).as_secs_f64() / median.as_secs_f64(),
            samples: samples.len(),
        }
    }

    fn to_json(self) -> Value {
        json!({
            "median_ns": u64::try_from(self.median.as_nanos()).unwrap(),
            "spread": self.spread,
            "samples": self.samples,
        })
    }

    fn from_json(value: &Value) -> Self {
        Self {
            median: Duration::from_nanos(value["median_ns"].as_u64().unwrap()),
            spread: value["spread"].as_f64().unwrap(),
            samples: usize::try_from(value["samples"].as_u64().unwrap()).unwrap(),
        }
    }
}

#[derive(Debug, Default)]
pub struct Benches {
    filters: Vec<String>,
    save: Option<String>,
    compare: Option<String>,
    results: BTreeMap<String, Summary>,
}

impl Benches {
    #[must_use]
    pub fn from_args() -> Self {
        let mut benches = Self::default();
        let mut arguments = env::args().skip(1);
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--bench" => {}
                "--save" => benches.save = Some(arguments.next().expect("--save takes a name")),
                "--compare" => {
                    benches.compare = Some(arguments.next().expect("--compare takes a name"));
                }
                _ => benches.filters.push(argument),
            }
        }
        benches
    }

    #[must_use]
    pub fn wants(&self, name: &str) -> bool {
        self.filters.is_empty() || self.filters.iter().any(|filter| name.contains(filter))
    }

    pub fn run(&mut self, name: &str, mut run: impl FnMut()) {
        if !self.wants(name) {
            return;
        }

        let started = Instant::now();
        while started.elapsed() < WARM_UP {
            run();
        }

        let mut samples = Vec::new();
        let started = Instant::now();
        while started.elapsed() < MEASURE || samples.len() < MIN_SAMPLES {
            let sample = Instant::now();
            run();
            samples.push(sample.elapsed());
        }

        let summary = Summary::from_samples(samples);
        println!(
            "{name:<44} {:>12.3?}  ±{:>5.1}%  ({} samples)",
            summary.median,
            summary.spread * 50.0,
            summary.samples
        );
        self.results.insert(name.to_owned(), summary);
    }

    pub fn finish(self) {
        if let Some(name) = &self.compare {
            let baseline = read_baseline(name);
            println!("\nagainst {name}, {}:", baseline["environment"]);
            for (bench, summary) in &self.results {
                let Some(before) = baseline["results"].get(bench).map(Summary::from_json) else {
                    println!("{bench:<44} new");
                    continue;
                };
                let change = summary.median.as_secs_f64() / before.median.as_secs_f64() - 1.0;
                let flag = if change.abs() > summary.spread.max(before.spread) {
                    if change > 0.0 { "  slower" } else { "  faster" }
                } else {
                    ""
                };
                println!(
                    "{bench:<44} {:>12.3?} -> {:>12.3?}  {:+.1}%{flag}",
                    before.median,
                    summary.median,
                    change * 100.0
                );
            }
        }
        if let Some(name) = &self.save {
            let mut baseline = read_baseline(name);
            baseline["environment"] = environment();
            if !baseline["results"].is_object() {
                baseline["results"] = json!({});
            }

            for (bench, summary) in &self.results {
                baseline["results"][bench] = summary.to_json();
            }

            let path = baseline_path(name);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(
                &path,
                serde_json::to_string_pretty(&baseline).unwrap() + "\n",
            )
            .unwrap();
            println!("\nsaved {}", path.display());
        }
    }
}

fn environment() -> Value {
    let output = |program: &str, arguments: &[&str]| {
        Command::new(program)
            .args(arguments)
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
    };
    json!({
        "commit": output("git", &["rev-parse", "--short", "HEAD"]),
        "dirty": output("git", &["status", "--porcelain", "--untracked-files=no"]).is_some_and(|status| !status.is_empty()),
        "gpu": output("nvidia-smi", &["--query-gpu=name,driver_version", "--format=csv,noheader"]),
    })
}

fn baseline_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../benchmarks")
        .join(format!("{name}.json"))
}

fn read_baseline(name: &str) -> Value {
    fs::read_to_string(baseline_path(name))
        .map_or_else(|_| json!({}), |text| serde_json::from_str(&text).unwrap())
}
