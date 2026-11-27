//! Singe's side of the generation comparison with vLLM: runs every case of `comparisons/cases.toml` for one model and prints one JSON line of timings per case, as `comparisons/vllm/generate.py` does.
//!
//! `--snapshot DIR --model NAME`, `--case NAME` for one case, `--cases PATH` for another cases file, and `--eager` for the decoder Singe runs without compiling, which `just compare` leaves out, since vLLM is compared as it runs in production, compiled, but which checks Singe's compiled decoder against its own eager one.
//! Each case generates from the same token ids vLLM's side does, greedily and with no stop token, so both sides do the same work.

mod support;

use std::{
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow};
use serde_json::json;
use singe::{
    DataType,
    generation::{Decoder, DecoderOptions, GenerationOptions, generate_batch},
};
use singe_model::{LoadOptions, Source, head::CausalLm, model};
use toml_edit::{DocumentMut, Item, Table};

use crate::support::{Arguments, start_runtime};

/// The cases file and what it sets for every case.
struct CaseFile {
    document: DocumentMut,
    first_id: u64,
    id_span: u64,
    new_tokens: usize,
    runs: usize,
}

impl CaseFile {
    fn read(path: &Path) -> Result<Self> {
        let document = fs::read_to_string(path)
            .with_context(|| format!("read {}", path.display()))?
            .parse::<DocumentMut>()?;
        let number = |key: &str| -> Result<u64> {
            let value = document
                .get(key)
                .and_then(Item::as_integer)
                .ok_or_else(|| anyhow!("cases.toml sets no integer `{key}`"))?;
            Ok(u64::try_from(value)?)
        };
        Ok(Self {
            first_id: number("first_id")?,
            id_span: number("id_span")?,
            new_tokens: usize::try_from(number("new_tokens")?)?,
            runs: usize::try_from(number("runs")?)?,
            document,
        })
    }

    /// Returns row `row`'s prompt of `length` ids, as `cases.toml` defines it and `generate.py` makes it.
    fn prompt(&self, row: u64, length: usize) -> Result<Vec<i64>> {
        let mut state = row;
        (0..length)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                Ok(i64::try_from(self.first_id + state % self.id_span)?)
            })
            .collect()
    }

    /// Returns the tables of the array of tables `name`.
    fn tables(&self, name: &str) -> Result<Vec<&Table>> {
        Ok(self
            .document
            .get(name)
            .and_then(Item::as_array_of_tables)
            .ok_or_else(|| anyhow!("cases.toml has no `[[{name}]]`"))?
            .iter()
            .collect())
    }
}

fn integer(table: &Table, key: &str) -> Result<usize> {
    let value = table
        .get(key)
        .and_then(Item::as_integer)
        .ok_or_else(|| anyhow!("a table of cases.toml sets no integer `{key}`"))?;
    Ok(usize::try_from(value)?)
}

fn string<'a>(table: &'a Table, key: &str) -> Result<&'a str> {
    table
        .get(key)
        .and_then(Item::as_str)
        .ok_or_else(|| anyhow!("a table of cases.toml sets no string `{key}`"))
}

/// Returns the median time of `runs` generations of `tokens` tokens from `rows`, after one untimed, each from an empty cache.
fn timed(
    decoder: &mut dyn Decoder,
    rows: &[&[i64]],
    tokens: usize,
    runs: usize,
) -> Result<Duration> {
    let options = GenerationOptions {
        max_new_tokens: tokens,
        ..GenerationOptions::default()
    };
    let mut generate = || -> Result<Duration> {
        decoder.reset();
        let started = Instant::now();
        generate_batch(
            decoder,
            rows,
            &options,
            None,
            |_, _, _| Ok(()),
            |_, _| Ok(()),
        )?;
        Ok(started.elapsed())
    };
    generate()?;
    let mut times = (0..runs).map(|_| generate()).collect::<Result<Vec<_>>>()?;
    times.sort();
    Ok(times[times.len() / 2])
}

fn main() -> Result<()> {
    // Startup counts from the process start, as vLLM's side counts from its script's.
    let started = Instant::now();
    let arguments = Arguments::parse(&["eager"])?;
    let name = arguments.required("model")?;
    let path = arguments.value("cases").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../comparisons/cases.toml"),
        PathBuf::from,
    );
    let cases = CaseFile::read(&path)?;
    let models = cases.tables("models")?;
    let entry = models
        .iter()
        .find(|table| string(table, "name").is_ok_and(|model| model == name))
        .ok_or_else(|| anyhow!("{} names no model {name}", path.display()))?;
    let context = integer(entry, "context")?;
    let compiled = !arguments.flag("eager");

    start_runtime(arguments.device())?;
    let runtime = started.elapsed();
    let source = Source::open(arguments.snapshot())?;
    let model = model::load::<Box<dyn CausalLm>>(
        &source,
        LoadOptions::new(arguments.device(), DataType::Bf16),
    )?;
    let load = started.elapsed().saturating_sub(runtime);

    let only = arguments.value("case");
    for case in cases.tables("cases")? {
        if only.is_some_and(|only| string(case, "name").is_ok_and(|name| name != only)) {
            continue;
        }

        let (batch, prompt_tokens) = (integer(case, "batch")?, integer(case, "prompt_tokens")?);
        let prompts = (0..batch)
            .map(|row| cases.prompt(u64::try_from(row)?, prompt_tokens))
            .collect::<Result<Vec<_>>>()?;
        let rows = prompts.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let preparing = Instant::now();
        let mut decoder = model.decoder(
            DecoderOptions::new(batch, i64::try_from(context)?),
            compiled,
        )?;
        let prepare = preparing.elapsed();
        // The first token is the prompt's prefill and one step; the later steps are what `new_tokens` adds to it.
        let first = timed(decoder.as_mut(), &rows, 1, cases.runs)?;
        let total = timed(decoder.as_mut(), &rows, cases.new_tokens, cases.runs)?;
        let later = total.saturating_sub(first).as_secs_f64() / (cases.new_tokens - 1) as f64;
        let line = json!({
            "side": "singe",
            "model": name,
            "mode": if compiled { "compiled" } else { "eager" },
            "case": string(case, "name")?,
            "batch": batch,
            "prompt_tokens": prompt_tokens,
            "new_tokens": cases.new_tokens,
            "startup_s": {
                "runtime": runtime.as_secs_f64(),
                "load": load.as_secs_f64(),
                "prepare": prepare.as_secs_f64(),
            },
            "first_token_ms": first.as_secs_f64() * 1e3,
            "step_ms": later * 1e3,
            "tokens_per_second": (batch * cases.new_tokens) as f64 / total.as_secs_f64(),
        });
        println!("{line}");
    }
    Ok(())
}
