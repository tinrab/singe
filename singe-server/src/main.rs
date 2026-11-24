//! `singe-server`: serves `singe-model` snapshots over HTTP, describes what a snapshot would serve, and lists the tasks and their endpoints.

use std::{error::Error, path::PathBuf, process::ExitCode};

use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use singe_model::{Source, model, task::Task};
use singe_server::{
    config::{DataTypeChoice, ModelConfig, ServerConfig},
    registry::endpoints,
    server,
};
use tokio::runtime::Runtime;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "singe-server",
    version,
    about = "Serves Singe models over HTTP"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    #[command(about = "Serves models, from the command line or a config file.")]
    Serve(Box<ServeArguments>),
    #[command(about = "Describes what a model would serve, without loading its weights.")]
    Inspect {
        #[arg(help = "A model directory, or a Hugging Face repository the local cache holds.")]
        model: String,
    },
    #[command(about = "Lists every task and the endpoints that serve it.")]
    Tasks,
}

#[derive(Args)]
struct ServeArguments {
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
    #[arg(long, value_enum)]
    data_type: Option<DataTypeChoice>,
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
            config.data_type = data_type;
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

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
    let result = match Cli::parse().command {
        Command::Serve(arguments) => serve(arguments),
        Command::Inspect { model } => inspect(&model),
        Command::Tasks => {
            tasks();
            Ok(())
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn serve(arguments: Box<ServeArguments>) -> Result<()> {
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

fn inspect(name: &str) -> Result<()> {
    let source = Source::find(name)?;
    println!("directory: {}", source.directory().display());
    println!("format: {:?}", source.format());
    println!("kind: {}", source.kind());
    let classes = source.classes()?;
    if !classes.is_empty() {
        println!("classes: {}", classes.join(", "));
    }

    match model::default_head(&source) {
        Ok(head) => println!("head: {}", head.id()),
        Err(error) => println!("head: none ({})", report(&error)),
    }

    match Task::default_for(&source) {
        Ok(task) => {
            println!("task: {}", task.id());
            println!("endpoints: {}", endpoints(task.id()).join(", "));
        }
        Err(error) => println!("task: none ({})", report(&error)),
    }

    if let Some(length) = source.context_length()? {
        println!("context length: {length}");
    }

    if let Some(data_type) = source.stored_data_type()? {
        println!("stored data type: {data_type}");
    }
    Ok(())
}

fn tasks() {
    for task in Task::ALL {
        if task.heads().is_empty() {
            println!("{}: no pipeline yet", task.id());
        } else {
            println!("{}: {}", task.id(), endpoints(task.id()).join(", "));
        }
    }
}

/// Formats an error and its sources as one line.
fn report(error: &dyn Error) -> String {
    let mut message = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        message.push_str(": ");
        message.push_str(&cause.to_string());
        source = cause.source();
    }
    message
}
