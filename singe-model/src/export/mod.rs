// Writes any snapshot this crate loads as CUDA C++ programs, with what runs them with nothing of Singe: a generation loop for a model that generates, and a runner for one that maps inputs to outputs.

use std::path::Path;

use singe::{
    DataType, Device, Error as SingeError, Result as SingeResult, Shape, Tensor,
    backend::cuda::export::{CudaExport, GpuArchitecture, Library},
    generation::{CachedDecoder, DecoderOptions},
    graph::Graph,
    nn::{DecoderModel, Model, TokenBatch},
};

use crate::{
    audio::automatic_speech_recognition::{TranscriberModel, load_transcriber},
    config::Snapshot,
    error::{Error, Result},
    text::{
        sentence_similarity::load_sentence_embedder,
        text_generation::{deepseek_v3, gemma2, gpt2, llama, qwen3, qwen3_5},
    },
    vision::{image_feature_extraction::load_image_encoder, image_to_image::load_autoencoder},
};

// How a snapshot exports: the target and libraries, the type of its weights, and the sizes its programs fix.
#[derive(Debug, Clone)]
pub struct ExportOptions {
    pub architecture: GpuArchitecture,
    pub libraries: Vec<Library>,
    pub data_type: DataType,
    // Where the model loads, whose captures run eagerly on it.
    pub device: Device,
    // The tokens a generating model's sequence holds, prompt included.
    pub capacity: i64,
    // The rows of a generating model's prompt step, beside its one-token step.
    pub prompt_rows: usize,
    // The seconds of audio Qwen3-ASR's encode step reads.
    pub audio_seconds: i64,
    // The tokens a sentence embedder's sequence holds.
    pub sequence_length: i64,
    // The side of a square image an autoencoder reads.
    pub image_size: i64,
}

impl ExportOptions {
    #[must_use]
    pub fn new(architecture: GpuArchitecture) -> Self {
        Self {
            architecture,
            libraries: Library::ALL.to_vec(),
            data_type: DataType::Bf16,
            device: Device::Cuda(0),
            capacity: 1024,
            prompt_rows: 64,
            audio_seconds: 30,
            sequence_length: 128,
            image_size: 256,
        }
    }

    fn export(&self) -> CudaExport {
        CudaExport::new(self.architecture).with_libraries(self.libraries.clone())
    }
}

// What an export wrote, and the files that run it, each of which `model.json` or its own manifest says how to build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Exported {
    // `model_generate`, a greedy generation loop over a program set.
    Generator,
    // `model_run`, which maps input files to output files.
    Runner,
    // `encoder_run` and `decoder_run`, an autoencoder's halves.
    Autoencoder,
}

// Writes `snapshot`'s programs to `directory`, choosing what to export by the model its config names.
pub fn export_snapshot(
    snapshot: impl AsRef<Path>,
    directory: impl AsRef<Path>,
    options: &ExportOptions,
) -> Result<Exported> {
    let (snapshot, directory) = (snapshot.as_ref(), directory.as_ref());
    let files = Snapshot::new(snapshot);
    // A `diffusers` snapshot names its class rather than a model type.
    let Ok(model_type) = files.model_type() else {
        autoencoder(snapshot, directory, options)?;
        return Ok(Exported::Autoencoder);
    };
    let (device, data_type) = (options.device, options.data_type);
    match model_type.as_str() {
        "deepseek_v3" => decoder(
            &deepseek_v3::load(snapshot, device, data_type)?,
            directory,
            options,
        )?,
        "gemma2" => decoder(
            &gemma2::load(snapshot, device, data_type)?,
            directory,
            options,
        )?,
        "gpt2" => decoder(
            &gpt2::load(snapshot, device, data_type)?,
            directory,
            options,
        )?,
        "llama" => decoder(
            &llama::load(snapshot, device, data_type)?,
            directory,
            options,
        )?,
        "qwen3" => decoder(
            &qwen3::load(snapshot, device, data_type)?,
            directory,
            options,
        )?,
        "qwen3_5" | "qwen3_5_text" => {
            decoder(
                &qwen3_5::load(snapshot, device, data_type)?,
                directory,
                options,
            )?;
        }
        "whisper" | "qwen3_asr" => transcriber(snapshot, directory, options)?,
        "dinov2" => {
            let model = load_image_encoder(snapshot, device, data_type)?;
            let size = files.config::<serde_json::Value>()?["image_size"]
                .as_i64()
                .ok_or_else(|| {
                    Error::UnsupportedConfig("an image encoder without its size".to_owned())
                })?;
            let images = Tensor::zeroed_on(device, data_type, Shape::create([1, 3, size, size])?)?;
            let (graph, _) = model.capture_graph("images", images)?;
            runner(&graph, directory, options)?;
            return Ok(Exported::Runner);
        }
        "bert" => {
            let model = load_sentence_embedder(snapshot, device, data_type)?;
            let tokens = vec![
                0;
                usize::try_from(options.sequence_length)
                    .map_err(|_| SingeError::out_of_range("sequence length"))?
            ];
            let batch = TokenBatch::from_sequences(device, &[tokens.as_slice()])?;
            let (graph, _) = model.capture_group(&batch)?;
            runner(&graph, directory, options)?;
            return Ok(Exported::Runner);
        }
        other => {
            return Err(Error::UnsupportedConfig(format!(
                "exporting model type {other}"
            )));
        }
    }
    Ok(Exported::Generator)
}

// A decoder's one-token step and a prompt step of `prompt_rows` rows as one set, with its generation loop.
fn decoder<L: DecoderModel>(
    model: &Model<L>,
    directory: &Path,
    options: &ExportOptions,
) -> Result<()> {
    let decoder = CachedDecoder::create(model, DecoderOptions::new(1, options.capacity))?;
    let step = decoder.step_graph()?;
    let prompt = decoder.chunk_step_graph(options.prompt_rows.saturating_sub(1).max(1))?;
    options
        .export()
        .with_decoding(step.layout)
        .export_steps(&[("decode", &step.graph), ("prompt", &prompt.graph)])?
        .write_to(directory, "model")?;
    Ok(())
}

// A transcriber's encode step, which reads samples, beside its decoder's steps, as one set with its generation loop.
fn transcriber(snapshot: &Path, directory: &Path, options: &ExportOptions) -> Result<()> {
    let transcriber = load_transcriber(snapshot, options.device, options.data_type)?;
    let export = options.export();
    match transcriber.model() {
        TranscriberModel::Whisper(model) => {
            let decoder = model
                .create_decoder(DecoderOptions::new(1, model.config().max_target_positions))?;
            let samples = Tensor::zeroed_on(
                options.device,
                DataType::F32,
                Shape::create([model.samples()])?,
            )?;
            let (encode, _) = model.encode_step_graph(&decoder, &samples)?;
            let step = decoder.step_graph()?;
            export
                .with_decoding(step.layout)
                .export_steps(&[("encode", &encode), ("decode", &step.graph)])?
                .write_to(directory, "model")?;
        }
        TranscriberModel::Qwen3Asr(model) => {
            let chunks = options.audio_seconds.max(1);
            let rows = chunks * model.config().audio_config.max_position_embeddings;
            // The encode step writes a row for every position of its chunks after the prompt's start, which the cache must hold beside the transcription.
            let capacity = options.capacity.max(rows + 64);
            let decoder = CachedDecoder::create(model.decoder(), DecoderOptions::new(1, capacity))?;
            let (encode, _) = model.encode_step_graph(&decoder, chunks)?;
            let step = decoder.step_graph()?;
            let prompt = decoder.chunk_step_graph(options.prompt_rows.saturating_sub(1).max(1))?;
            export
                .with_decoding(step.layout)
                .export_steps(&[
                    ("encode", &encode),
                    ("decode", &step.graph),
                    ("prompt", &prompt.graph),
                ])?
                .write_to(directory, "model")?;
        }
    }
    Ok(())
}

// An autoencoder's halves, each a program with its runner: `encoder` from images, and `decoder` from the latents it gives.
fn autoencoder(snapshot: &Path, directory: &Path, options: &ExportOptions) -> Result<()> {
    let model = load_autoencoder(snapshot, options.device, options.data_type)?;
    let size = options.image_size;
    let images = Tensor::zeroed_on(
        options.device,
        options.data_type,
        Shape::create([1, 3, size, size])?,
    )?;
    let latents = model.encode(&images)?;
    for (prefix, half, input) in [
        ("encoder", model.encoder(), images),
        ("decoder", model.decoder(), latents),
    ] {
        let (graph, _) = half.capture_graph("input", input)?;
        options
            .export()
            .export(&graph)?
            .write_to(directory, prefix)?;
    }
    Ok(())
}

// One program and its runner.
fn runner(graph: &Graph, directory: &Path, options: &ExportOptions) -> SingeResult<()> {
    options.export().export(graph)?.write_to(directory, "model")
}
