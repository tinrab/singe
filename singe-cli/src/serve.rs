// `singe serve`: serves models over HTTP with `singe-server`, from the command line or a config file.

use std::path::PathBuf;

use anyhow::Result;
use clap::Args;
use singe_server::{
    config::{ModelConfig, ServerConfig},
    server,
};
use tokio::runtime::Runtime;
use tracing_subscriber::EnvFilter;

use crate::arguments::TypeChoice;

#[derive(Args)]
pub struct ServeArguments {
    #[arg(
        help = "Models to serve: each a directory or a cached Hugging Face repository, as `MODEL` or `NAME=MODEL`."
    )]
    models: Vec<String>,
    #[arg(
        long,
        help = "A TOML file of the server's settings and models, which the other options override and add to."
    )]
    config: Option<PathBuf>,
    #[arg(long)]
    host: Option<String>,
    #[arg(long)]
    port: Option<u16>,
    #[arg(
        long = "api-key",
        env = "SINGE_API_KEY",
        value_delimiter = ',',
        help = "Keys a request must carry one of, as `Authorization: Bearer KEY`."
    )]
    api_keys: Vec<String>,
    #[arg(
        long = "allowed-origin",
        value_delimiter = ',',
        help = "Origins browsers may call from, or `*` for any."
    )]
    allowed_origins: Vec<String>,
    #[arg(long, help = "The largest request body, in bytes.")]
    max_body_size: Option<usize>,
    #[arg(
        long,
        help = "Seconds running requests may finish after a signal to stop."
    )]
    shutdown_grace: Option<u64>,
    #[command(flatten)]
    model: ModelArguments,
}

#[derive(Args)]
struct ModelArguments {
    #[arg(long, help = "The task to serve, or else each snapshot's default.")]
    task: Option<String>,
    #[arg(long = "type", value_enum, help = "The type the weights load in.")]
    data_type: Option<TypeChoice>,
    #[arg(long, help = "`cuda`, `cuda:N`, or `cpu`.")]
    device: Option<String>,
    #[arg(long, help = "Runs eager steps rather than compiled ones.")]
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
}

impl ModelArguments {
    fn config(&self, argument: &str) -> ModelConfig {
        let (name, model) = match argument.split_once('=') {
            Some((name, model)) => (Some(name.to_owned()), model),
            None => (None, argument),
        };
        let mut config = ModelConfig::new(model);
        config.name = name;
        config.task.clone_from(&self.task);
        if let Some(data_type) = self.data_type {
            config.data_type = data_type.choice();
        }

        if let Some(device) = &self.device {
            config.device.clone_from(device);
        }

        config.eager = self.eager;
        config.max_model_len = self.max_model_len;
        if let Some(sequences) = self.max_num_seqs {
            config.max_num_seqs = sequences;
        }

        config.cache_tokens = self.cache_tokens;
        if let Some(queued) = self.max_queued_requests {
            config.max_queued_requests = queued;
        }

        config.seed = self.seed;
        config
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

    config.models.extend(
        arguments
            .models
            .iter()
            .map(|model| arguments.model.config(model)),
    );
    Runtime::new()?.block_on(server::serve(config))?;
    Ok(())
}
