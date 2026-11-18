// Exported programs checked against the references: a text classifier's snapshot exported as a CUDA C++ program with nothing of Singe, its runner built with `nvcc` as the program's manifest says, run on each input's tokens written to files, and its scores compared with the reference's.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context as _, Result, bail, ensure};
use serde_json::Value;
use singe::{DataType, Device, backend::cuda::export::compiled::build_written_runner};
use singe_model::{
    Source,
    export::{ExportOptions, export},
    processing::text::TextTokenizer,
    task::Task,
};

use super::{Case, DEVICE, compare};

// The tokens the exported program reads, which each input is padded to, more than any case's input holds.
const LENGTH: i64 = 64;

// Exports `snapshot` to `directory`, builds its runner, and runs it on each of `case`'s inputs, returning whether every output matched the reference's and what each comparison found.
pub fn run(case: &Case, snapshot: &Path, directory: &Path) -> Result<(bool, Vec<String>)> {
    if case.task != "text-classification" {
        bail!("no exported check for {}", case.task);
    }

    let architecture = singe::backend::cuda::export::compiled::program_target(DEVICE)
        .context("a GPU and nvcc to build the program for")?;
    let mut options = ExportOptions::new(architecture);
    options.data_type = DataType::F32;
    options.sequence_length = LENGTH;
    if directory.exists() {
        fs::remove_dir_all(directory)?;
    }
    fs::create_dir_all(directory)?;
    export(
        &Source::open(snapshot)?,
        Task::TextClassification,
        directory,
        &options,
    )?;
    let runner = build_written_runner(directory)?;
    let tokenizer = TextTokenizer::load(snapshot)?;

    let mut matched = true;
    let mut details = Vec::new();
    for (index, (input, want)) in case.inputs.iter().zip(&case.expected).enumerate() {
        let text = input
            .as_str()
            .context("a text classifier's input is a text")?;
        let scores = scores(&runner, directory, &tokenizer, text, index)?;
        let (best, score) = scores
            .iter()
            .copied()
            .enumerate()
            .fold(
                None,
                |best: Option<(usize, f32)>, (label, score)| match best {
                    Some((_, most)) if most >= score => best,
                    _ => Some((label, score)),
                },
            )
            .context("a classifier with no labels")?;
        let got =
            serde_json::json!({ "label": labels(snapshot, scores.len())?[best], "score": score });
        let (same, detail) = compare::outputs(&case.task, want, &got);
        matched &= same;
        details.push(detail);
    }
    Ok((matched, details))
}

// The softmaxed scores the runner gives `text`, its tokens padded to `LENGTH` and written to files named for input `index`.
fn scores(
    runner: &Path,
    directory: &Path,
    tokenizer: &TextTokenizer,
    text: &str,
    index: usize,
) -> Result<Vec<f32>> {
    let length = usize::try_from(LENGTH)?;
    let batch = tokenizer.encode_batch(&[text], None, Device::Cpu)?;
    ensure!(
        batch.tokens().shape().dimensions()[1] <= LENGTH,
        "an input of more tokens than the exported program reads"
    );
    let padded = |mut values: Vec<i64>| {
        values.resize(length, 0);
        values
            .into_iter()
            .flat_map(i64::to_le_bytes)
            .collect::<Vec<_>>()
    };
    let mut padding = batch
        .padding()
        .to_vec::<bool>()?
        .into_iter()
        .map(u8::from)
        .collect::<Vec<_>>();
    padding.resize(length, 0);
    let file = |name: &str| -> PathBuf { directory.join(format!("input-{index}-{name}.bin")) };
    let inputs = [
        (file("tokens"), padded(batch.tokens().to_vec::<i64>()?)),
        (
            file("token_types"),
            padded(batch.token_types().to_vec::<i64>()?),
        ),
        (file("padding"), padding),
    ];
    for (path, bytes) in &inputs {
        fs::write(path, bytes)?;
    }

    let output = file("scores");
    let status = Command::new(runner)
        .args(inputs.iter().map(|(path, _)| path))
        .arg(&output)
        .current_dir(directory)
        .status()
        .context("run the exported program")?;
    ensure!(
        status.success(),
        "the exported program exited with {status}"
    );
    let logits = fs::read(&output)?
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect::<Vec<_>>();
    let most = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponents = logits
        .iter()
        .map(|logit| (logit - most).exp())
        .collect::<Vec<_>>();
    let total = exponents.iter().sum::<f32>();
    Ok(exponents.iter().map(|value| value / total).collect())
}

// The snapshot's names of its classifier's `count` outputs, from `config.json`'s `id2label`, or `LABEL_N` without one, as `transformers` names them.
fn labels(snapshot: &Path, count: usize) -> Result<Vec<String>> {
    let config: Value = serde_json::from_str(&fs::read_to_string(snapshot.join("config.json"))?)?;
    (0..count)
        .map(|index| {
            Ok(match config["id2label"].get(index.to_string()) {
                Some(name) => name
                    .as_str()
                    .with_context(|| format!("id2label's label {index}"))?
                    .to_owned(),
                None => format!("LABEL_{index}"),
            })
        })
        .collect()
}
