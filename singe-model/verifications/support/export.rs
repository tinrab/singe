// Exported programs checked against the references: a case's snapshot exported as CUDA C++ programs with nothing of Singe, built with `nvcc` as their manifests say, run on the case's inputs written to files, and what they give compared with the reference's outputs.
// A text classifier's runner reads tokens, a generation loop a prompt's tokens, a transcriber's loop a clip's samples and its prompt's tokens, and an autoencoder's two runners pixels and then latents.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context as _, Result, bail, ensure};
use serde_json::{Value, json};
use singe::{
    Device,
    backend::cuda::export::compiled::{build_written_loop, build_written_runner, program_target},
};
use singe_model::{
    Source,
    export::cuda::{ExportOptions, export},
    format::transformers::classifier_config,
    model::audio::whisper::{self, WhisperTask},
    processor::{audio::Audio, image::Image, text::TextTokenizer},
    task::Task,
};

use crate::support::{Case, DEVICE, compare, media};

// The tokens the exported program reads, which each input is padded to, more than any case's input holds.
const LENGTH: i64 = 64;

// Exports `snapshot` to `directory` for `case`'s task, builds what runs it, and runs it on each of the case's inputs, returning whether every output matched the reference's and what each comparison found.
pub fn run(case: &Case, snapshot: &Path, directory: &Path) -> Result<(bool, Vec<String>)> {
    let architecture = program_target(DEVICE).context("a GPU and nvcc to build the program for")?;
    let mut options = ExportOptions::new(architecture);
    options.data_type = case.data_type;
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
    let inputs = case
        .inputs
        .iter()
        .map(|input| media(input.clone()))
        .collect::<Vec<_>>();
    if task == Task::ImageToImage {
        let side = inputs
            .first()
            .and_then(Value::as_str)
            .map(|path| Image::open(Path::new(path)))
            .transpose()?
            .context("an autoencoder's input is an image")?
            .width;
        options.image_size = i64::from(side);
    }
    let source = Source::open(snapshot)?;
    export(&source, task, directory, &options)?;

    let mut matched = true;
    let mut details = Vec::new();
    for (index, (input, want)) in inputs.iter().zip(&case.expected).enumerate() {
        let got = match task {
            Task::TextClassification => classify(&source, directory, input, index)?,
            Task::TextGeneration => generate(&source, directory, input, &case.parameters)?,
            Task::AutomaticSpeechRecognition => {
                transcribe(&source, directory, input, &case.parameters, index)?
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
fn classify(source: &Source, directory: &Path, input: &Value, index: usize) -> Result<Value> {
    let runner = build_written_runner(directory, "model")?;
    let tokenizer = TextTokenizer::load(source)?;
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
    let config = classifier_config(source)?;
    let label = config
        .labels()
        .get(best)
        .with_context(|| format!("a score for label {best} of {}", config.labels().len()))?;
    Ok(json!({ "label": label, "score": score }))
}

// The tokens the generation loop gives after `input`, a prompt encoded without special tokens, as the reference encodes it, since an exported loop's tokenizer may only decode.
fn generate(source: &Source, directory: &Path, input: &Value, parameters: &Value) -> Result<Value> {
    let program = build_written_loop(directory)?;
    let prompt = input.as_str().context("a generation's input is a prompt")?;
    let prompt = TextTokenizer::load(source)?
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
    Ok(json!({ "generated_token_ids": tokens }))
}

// The text the transcriber's loop gives for `input`, a clip of 16 kHz samples, after Whisper's prompt for an English transcription.
fn transcribe(
    source: &Source,
    directory: &Path,
    input: &Value,
    parameters: &Value,
    index: usize,
) -> Result<Value> {
    let program = build_written_loop(directory)?;
    let tokenizer = TextTokenizer::load(source)?;
    let clip = Audio::open(Path::new(
        input
            .as_str()
            .context("a transcription's input is a clip")?,
    ))?;
    ensure!(clip.rate == 16_000, "a clip of other than 16 kHz");
    let samples = input_file(directory, index, "samples");
    write_values(&samples, &clip.samples, f32::to_le_bytes)?;
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
        .arg(&samples)
        .arg(prompt)
        .current_dir(directory))?;
    let text = tokenizer.decode(&tokens, true)?.trim().to_owned();
    Ok(json!({ "text": text }))
}

// `input`, an image, through the autoencoder's encoder and then its decoder, in [-1, 1] as the pipeline scales it.
fn reconstruct(directory: &Path, input: &Value, index: usize) -> Result<Value> {
    let image = Image::open(Path::new(
        input
            .as_str()
            .context("an autoencoder's input is an image")?,
    ))?;
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
    let file = |name: &str| input_file(directory, index, name);
    write_values(&file("pixels"), &pixels, f32::to_le_bytes)?;
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

fn input_file(directory: &Path, index: usize, name: &str) -> PathBuf {
    directory.join(format!("input-{index}-{name}.bin"))
}

fn write_values<T: Copy, const N: usize>(
    path: &Path,
    values: &[T],
    bytes: fn(T) -> [u8; N],
) -> Result<()> {
    let written = values
        .iter()
        .flat_map(|&value| bytes(value))
        .collect::<Vec<_>>();
    fs::write(path, written).with_context(|| format!("write {}", path.display()))
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
    let &[_, tokens] = batch.tokens().shape().dimensions() else {
        bail!("a token batch of other than two dimensions");
    };
    ensure!(
        tokens <= LENGTH,
        "an input of more tokens than the exported program reads"
    );
    let padded = |mut values: Vec<i64>| {
        values.resize(length, 0);
        values
    };
    let mut padding = batch
        .padding()
        .to_vec::<bool>()?
        .into_iter()
        .map(u8::from)
        .collect::<Vec<_>>();
    padding.resize(length, 0);
    let file = |name: &str| input_file(directory, index, name);
    let inputs = [file("tokens"), file("token_types"), file("padding")];
    write_values(
        &inputs[0],
        &padded(batch.tokens().to_vec::<i64>()?),
        i64::to_le_bytes,
    )?;
    write_values(
        &inputs[1],
        &padded(batch.token_types().to_vec::<i64>()?),
        i64::to_le_bytes,
    )?;
    write_values(&inputs[2], &padding, u8::to_le_bytes)?;

    let output = file("scores");
    let status = Command::new(runner)
        .args(&inputs)
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
