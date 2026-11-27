// `singe serve`: serves models over HTTP with `singe-server`, from the command line or a config file.

use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::Args;
use singe_model::function::DimensionRange;
use singe_server::{
    config::{ModelConfig, ServerConfig},
    server,
};
use tokio::runtime::Runtime;
use tracing_subscriber::EnvFilter;

use crate::{
    arguments::TypeChoice,
    export::{TuneArguments, named_ranges, named_shapes, parse_dim, parse_input, parse_range},
};

#[derive(Args)]
pub struct ServeArguments {
    /// Models to serve: each a directory or a cached Hugging Face repository, as `MODEL` or `NAME=MODEL`.
    models: Vec<String>,
    /// Exported programs to serve, each a `singe export --target cuda` directory, as `DIR` or `NAME=DIR`.
    #[arg(long = "program")]
    programs: Vec<String>,
    /// ONNX files to serve through Singe, as `FILE` or `NAME=FILE`.
    #[arg(long = "file")]
    files: Vec<String>,
    /// A TOML file of the server's settings and models, which the other options override and add to.
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    host: Option<String>,
    #[arg(long)]
    port: Option<u16>,
    /// Keys a request must carry one of, as `Authorization: Bearer KEY`.
    #[arg(long = "api-key", env = "SINGE_API_KEY", value_delimiter = ',')]
    api_keys: Vec<String>,
    /// Origins browsers may call from, or `*` for any.
    #[arg(long = "allowed-origin", value_delimiter = ',')]
    allowed_origins: Vec<String>,
    /// The largest request body, in bytes.
    #[arg(long)]
    max_body_size: Option<usize>,
    /// Seconds running requests may finish after a signal to stop.
    #[arg(long)]
    shutdown_grace: Option<u64>,
    #[command(flatten)]
    model: ModelArguments,
}

#[derive(Args)]
struct ModelArguments {
    /// The task to serve, or else each snapshot's default.
    #[arg(long)]
    task: Option<String>,
    /// The type the weights load in.
    #[arg(long = "type", value_enum)]
    data_type: Option<TypeChoice>,
    /// `cuda`, `cuda:N`, or `cpu`.
    #[arg(long)]
    device: Option<String>,
    /// Runs eager steps rather than compiled ones.
    #[arg(long)]
    eager: bool,
    #[arg(long)]
    max_model_len: Option<usize>,
    #[arg(long)]
    max_num_seqs: Option<usize>,
    #[arg(long)]
    cache_tokens: Option<usize>,
    #[arg(long)]
    max_queued_requests: Option<usize>,
    #[arg(long)]
    seed: Option<u64>,
    /// A Silero VAD v5 snapshot, such as `onnx-community/silero-vad`, that a speech recognizer's realtime sessions find turns with.
    #[arg(long)]
    voice_activity: Option<String>,
    /// The snapshot whose tokenizer or image processor and config a program or an ONNX file's task reads.
    #[arg(long)]
    processor: Option<String>,
    /// An ONNX file's named dimension, as `sequence_length=128,512`: a step per combination, which requests pad up to.
    #[arg(long = "dim", value_parser = parse_dim)]
    dims: Vec<(String, Vec<i64>)>,
    /// An ONNX file's named dimension that follows each request's size, as `height=224..1344` or `height=224..`; the file imports at the least size.
    #[arg(long = "range", value_parser = parse_range)]
    ranges: Vec<(String, DimensionRange)>,
    /// Fixes every axis of an ONNX file's input.
    #[arg(long = "input", value_parser = parse_input)]
    inputs: Vec<(String, Vec<i64>)>,
    #[command(flatten)]
    tune: TuneArguments,
}

#[derive(Clone, Copy)]
enum ModelKind {
    Snapshot,
    Program,
    Onnx,
}

impl ModelArguments {
    fn config(&self, argument: &str, kind: ModelKind) -> Result<ModelConfig> {
        let (name, model) = match argument.split_once('=') {
            Some((name, model)) => (Some(name.to_owned()), model),
            None => (None, argument),
        };
        let mut config = match kind {
            ModelKind::Snapshot => ModelConfig::new(model),
            ModelKind::Program => ModelConfig::program(PathBuf::from(model)),
            ModelKind::Onnx => ModelConfig::file(PathBuf::from(model)),
        };
        config.name = name;
        config.task.clone_from(&self.task);
        if let Some(device) = &self.device {
            config.device.clone_from(device);
        }

        if let Some(queued) = self.max_queued_requests {
            config.max_queued_requests = queued;
        }

        config.voice_activity.clone_from(&self.voice_activity);

        match kind {
            ModelKind::Snapshot => {
                if let Some(data_type) = self.data_type {
                    config.data_type = data_type.choice();
                }

                config.eager = self.eager;
                config.max_model_len = self.max_model_len;
                if let Some(sequences) = self.max_num_seqs {
                    config.max_num_seqs = sequences;
                }

                config.cache_tokens = self.cache_tokens;
                config.seed = self.seed;
            }
            ModelKind::Program => config.processor.clone_from(&self.processor),
            ModelKind::Onnx => {
                config.processor.clone_from(&self.processor);
                config.dims = named_shapes("--dim", &self.dims)?;
                config.ranges = named_ranges(&self.ranges)?;
                config.inputs = named_shapes("--input", &self.inputs)?;
                config.tune = self.tune.config();
            }
        }
        Ok(config)
    }

    fn has_snapshot_options(&self) -> bool {
        self.data_type.is_some()
            || self.eager
            || self.max_model_len.is_some()
            || self.max_num_seqs.is_some()
            || self.cache_tokens.is_some()
            || self.seed.is_some()
    }
}

pub fn serve(arguments: ServeArguments) -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
    let mut config = match &arguments.config {
        Some(path) => ServerConfig::read(path)?,
        None => ServerConfig::default(),
    };
    if let Some(host) = arguments.host {
        config.host = host;
    }

    if let Some(port) = arguments.port {
        config.port = port;
    }

    config.api_keys.extend(arguments.api_keys);
    config.allowed_origins.extend(arguments.allowed_origins);
    if let Some(size) = arguments.max_body_size {
        config.max_body_size = size;
    }

    if let Some(grace) = arguments.shutdown_grace {
        config.shutdown_grace_seconds = grace;
    }

    let model = &arguments.model;
    if model.tune.tune_inputs.is_some() {
        bail!("--tune-inputs applies to singe export");
    }

    if arguments.files.is_empty()
        && (model.tune.tune
            || !model.dims.is_empty()
            || !model.ranges.is_empty()
            || !model.inputs.is_empty())
    {
        bail!("--dim, --range, --input, and --tune apply to --file models");
    }

    if arguments.programs.is_empty() && arguments.files.is_empty() && model.processor.is_some() {
        bail!("--processor applies to --program and --file models");
    }

    if arguments.models.is_empty() && model.has_snapshot_options() {
        bail!("--type, --eager, --seed, and the sequence limits apply to snapshot models");
    }

    let sources = [
        (&arguments.models, ModelKind::Snapshot),
        (&arguments.programs, ModelKind::Program),
        (&arguments.files, ModelKind::Onnx),
    ];
    for (models, kind) in sources {
        for model in models {
            config.models.push(arguments.model.config(model, kind)?);
        }
    }

    Runtime::new()?.block_on(server::serve(config))?;
    Ok(())
}
