// Exported programs checked against the references: a case's snapshot exported as CUDA C++ programs with nothing of Singe, built with `nvcc` as their manifests say, run on the case's inputs written to files, and what they give compared with the reference's outputs.
// A text classifier's runner reads tokens, a generation loop a prompt's tokens, a transcriber's loop a clip's samples and its prompt's tokens, and an autoencoder's two runners pixels and then latents.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context as _, Result, bail, ensure};
use serde_json::Value;
use singe::{
    DataType, Device,
    backend::cuda::export::compiled::{build_written_loop, build_written_runner},
};
use singe_model::{
    Source,
    export::{ExportOptions, export},
    model::audio::whisper::{self, WhisperTask},
    processing::{audio::Audio, image::Image, text::TextTokenizer},
    task::Task,
};

use super::{Case, DEVICE, compare};

// The tokens the exported program reads, which each input is padded to, more than any case's input holds.
const LENGTH: i64 = 64;

// Exports `snapshot` to `directory` for `case`'s task, builds what runs it, and runs it on each of the case's inputs, returning whether every output matched the reference's and what each comparison found.
pub fn run(case: &Case, snapshot: &Path, directory: &Path) -> Result<(bool, Vec<String>)> {
    let architecture = singe::backend::cuda::export::compiled::program_target(DEVICE)
        .context("a GPU and nvcc to build the program for")?;
    let mut options = ExportOptions::new(architecture);
    options.data_type = DataType::F32;
    let task = match case.task.as_str() {
        "text-classification" => {
            options.sequence_length = LENGTH;
            Task::TextClassification
        }
        "text-generation" => {
            options.capacity = LENGTH;
            Task::TextGeneration
        }
        "automatic-speech-recognition" => Task::AutomaticSpeechRecognition,
        "image-to-image" => Task::ImageToImage,
        other => bail!("no exported check for {other}"),
    };
    if directory.exists() {
        fs::remove_dir_all(directory)?;
    }
    fs::create_dir_all(directory)?;
    if task == Task::ImageToImage {
        let side = case
            .inputs
            .first()
            .and_then(Value::as_str)
            .map(|path| Image::open(super::root().join(path)))
            .transpose()?
            .context("an autoencoder's input is an image")?
            .width;
        options.image_size = i64::from(side);
    }
    export(&Source::open(snapshot)?, task, directory, &options)?;

    let mut matched = true;
    let mut details = Vec::new();
    for (index, (input, want)) in case.inputs.iter().zip(&case.expected).enumerate() {
        let got = match task {
            Task::TextClassification => classify(snapshot, directory, input, index)?,
            Task::TextGeneration => generate(snapshot, directory, input, &case.parameters)?,
            Task::AutomaticSpeechRecognition => {
                transcribe(snapshot, directory, input, &case.parameters, index)?
            }
            _ => reconstruct(directory, input, index)?,
        };
        let (same, detail) = compare::outputs(&case.task, want, &got);
        matched &= same;
        details.push(detail);
    }
    Ok((matched, details))
}

// A text classifier's best label and its score for `input`, from the runner's logits.
fn classify(snapshot: &Path, directory: &Path, input: &Value, index: usize) -> Result<Value> {
    let runner = build_written_runner(directory, "model")?;
    let tokenizer = TextTokenizer::load(snapshot)?;
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
    Ok(serde_json::json!({ "label": labels(snapshot, scores.len())?[best], "score": score }))
}

// The tokens the generation loop gives after `input`, a prompt encoded without special tokens, as the reference encodes it, since an exported loop's tokenizer may only decode.
fn generate(snapshot: &Path, directory: &Path, input: &Value, parameters: &Value) -> Result<Value> {
    let program = build_written_loop(directory)?;
    let prompt = input.as_str().context("a generation's input is a prompt")?;
    let prompt = TextTokenizer::load(snapshot)?
        .encode(prompt, false)?
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(" ");
    let new = parameters["max_new_tokens"]
        .as_u64()
        .unwrap_or(24)
        .to_string();
    let tokens = ids(Command::new(program)
        .args(["--ids", "--tokens", &new, &prompt])
        .current_dir(directory))?;
    Ok(serde_json::json!({ "generated_token_ids": tokens }))
}

// The text the transcriber's loop gives for `input`, a clip of 16 kHz samples, after Whisper's prompt for an English transcription.
fn transcribe(
    snapshot: &Path,
    directory: &Path,
    input: &Value,
    parameters: &Value,
    index: usize,
) -> Result<Value> {
    let program = build_written_loop(directory)?;
    let tokenizer = TextTokenizer::load(snapshot)?;
    let clip = Audio::open(
        super::root().join(
            input
                .as_str()
                .context("a transcription's input is a clip")?,
        ),
    )?;
    ensure!(clip.rate == 16_000, "a clip of other than 16 kHz");
    let source = directory.join(format!("input-{index}-samples.bin"));
    fs::write(
        &source,
        clip.samples
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let prompt = whisper::prompt(&tokenizer, "en", WhisperTask::Transcribe)?
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(" ");
    let new = parameters["max_new_tokens"]
        .as_u64()
        .unwrap_or(128)
        .to_string();
    let tokens = ids(Command::new(program)
        .args(["--ids", "--tokens", &new])
        .arg(&source)
        .arg(prompt)
        .current_dir(directory))?;
    let text = tokenizer.decode(&tokens, true)?.trim().to_owned();
    Ok(serde_json::json!({ "text": text }))
}

// `input`, an image, through the autoencoder's encoder and then its decoder, in [-1, 1] as the pipeline scales it.
fn reconstruct(directory: &Path, input: &Value, index: usize) -> Result<Value> {
    let image = Image::open(
        super::root().join(
            input
                .as_str()
                .context("an autoencoder's input is an image")?,
        ),
    )?;
    let (width, height) = (
        usize::try_from(image.width)?,
        usize::try_from(image.height)?,
    );
    // Pixels from height-width-channel bytes to channel-height-width values in [-1, 1].
    let mut pixels = vec![0.0_f32; 3 * height * width];
    for (at, &byte) in image.pixels.iter().enumerate() {
        let (position, channel) = (at / 3, at % 3);
        pixels[channel * height * width + position] = f32::from(byte) * (2.0 / 255.0) - 1.0;
    }
    let file = |name: &str| directory.join(format!("input-{index}-{name}.bin"));
    fs::write(
        file("pixels"),
        pixels
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    for (part, from, to) in [
        ("encoder", "pixels", "latents"),
        ("decoder", "latents", "output"),
    ] {
        let runner = build_written_runner(directory, part)?;
        let status = Command::new(runner)
            .arg(file(from))
            .arg(file(to))
            .current_dir(directory)
            .status()
            .with_context(|| format!("run the exported {part}"))?;
        ensure!(status.success(), "the exported {part} exited with {status}");
    }
    let values = floats(&fs::read(file("output"))?);
    ensure!(
        values.len() == pixels.len(),
        "the decoder gave another image size"
    );
    // Values in [-1, 1] back to bytes, as the pipeline rounds them.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to a byte"
    )]
    let bytes = (0..height * width * 3)
        .map(|at| {
            let (position, channel) = (at / 3, at % 3);
            let value = values[channel * height * width + position];
            (((value + 1.0) * 0.5).clamp(0.0, 1.0) * 255.0).round() as u8
        })
        .collect();
    let output = Image {
        width: image.width,
        height: image.height,
        pixels: bytes,
    };
    Ok(serde_json::to_value(&output)?)
}

// The token ids a generation loop printed with `--ids`, one a line.
fn ids(command: &mut Command) -> Result<Vec<i64>> {
    let output = command
        .output()
        .context("run the exported generation loop")?;
    ensure!(
        output.status.success(),
        "the exported generation loop exited with {}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)?
        .lines()
        .map(|line| Ok(line.trim().parse::<i64>()?))
        .collect()
}

// Little-endian `F32` values.
fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect()
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
    let logits = floats(&fs::read(&output)?);
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
