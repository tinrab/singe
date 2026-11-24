//! What a server serves and how: read from a TOML file, from the command line, or built in code.

use std::{
    fs, io,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use singe::{DataType, Device};
use singe_model::Source;
use toml::de::Error as TomlError;

/// A server's settings and models, as its TOML file has them; every field has a default.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    /// The address to listen on, `127.0.0.1` by default.
    pub host: String,
    /// The port to listen on, 8000 by default.
    pub port: u16,
    /// Keys a request must carry one of; none leaves the server open.
    pub api_keys: Vec<String>,
    /// Origins browsers may call from, `*` for any; none turns CORS off.
    pub allowed_origins: Vec<String>,
    /// The largest request body, in bytes; a larger one is refused with 413.
    pub max_body_size: usize,
    /// How long running requests may finish after a signal to stop.
    pub shutdown_grace_seconds: u64,
    /// The models to serve.
    pub models: Vec<ModelConfig>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_owned(),
            port: 8000,
            api_keys: Vec::new(),
            allowed_origins: Vec::new(),
            max_body_size: 64 << 20,
            shutdown_grace_seconds: 30,
            models: Vec::new(),
        }
    }
}

impl ServerConfig {
    /// Reads a config from the TOML file at `path`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or does not parse as a config.
    pub fn read(path: &Path) -> Result<Self, ConfigError> {
        let text = fs::read_to_string(path).map_err(|source| ConfigError::Read {
            path: path.to_owned(),
            source,
        })?;
        toml::from_str(&text).map_err(|source| ConfigError::Parse {
            path: path.to_owned(),
            source: Box::new(source),
        })
    }
}

/// One model to serve and how to load it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// A model directory, or a Hugging Face repository the local cache holds, as `owner/name` or `owner/name@revision`.
    pub model: String,
    /// The name requests use, or else `model` as given.
    pub name: Option<String>,
    /// The task to serve, or else the snapshot's default.
    pub task: Option<String>,
    /// The type the weights load in.
    #[serde(default)]
    pub data_type: DataTypeChoice,
    /// `cuda`, `cuda:N`, or `cpu`, `cuda:0` by default.
    #[serde(default = "default_device")]
    pub device: String,
    /// Runs eager steps rather than compiled ones.
    #[serde(default)]
    pub eager: bool,
    /// A text generator's: the most tokens a sequence holds, prompt included, or else the model's context up to 4096.
    pub max_model_len: Option<usize>,
    /// A text generator's: the sequences a step runs at most, 4 by default.
    #[serde(default = "default_max_num_seqs")]
    pub max_num_seqs: usize,
    /// A text generator's: the tokens the shared cache holds, or else enough for every sequence at its longest.
    pub cache_tokens: Option<usize>,
    /// The requests that wait for the model at most, 64 by default; more are refused with 429.
    #[serde(default = "default_max_queued_requests")]
    pub max_queued_requests: usize,
    /// Seeds a text generator's sampling.
    pub seed: Option<u64>,
}

impl ModelConfig {
    /// Creates a config for `model` with every other field at its default.
    #[must_use]
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_owned(),
            name: None,
            task: None,
            data_type: DataTypeChoice::default(),
            device: default_device(),
            eager: false,
            max_model_len: None,
            max_num_seqs: default_max_num_seqs(),
            cache_tokens: None,
            max_queued_requests: default_max_queued_requests(),
            seed: None,
        }
    }

    /// Returns the name requests use: `name`, or else `model`.
    #[must_use]
    pub fn served_name(&self) -> &str {
        self.name.as_deref().unwrap_or(&self.model)
    }

    /// Returns the device `device` names.
    ///
    /// # Errors
    ///
    /// Returns an error if `device` is not `cpu`, `cuda`, or `cuda:N`.
    pub fn device(&self) -> Result<Device, ConfigError> {
        parse_device(&self.device)
    }
}

fn default_device() -> String {
    "cuda:0".to_owned()
}

const fn default_max_num_seqs() -> usize {
    4
}

const fn default_max_queued_requests() -> usize {
    64
}

/// Parses `cpu`, `cuda`, or `cuda:N` as a device; `cuda` is device 0.
///
/// # Errors
///
/// Returns [`ConfigError::Device`] for any other text.
pub fn parse_device(device: &str) -> Result<Device, ConfigError> {
    match device {
        "cpu" => Ok(Device::Cpu),
        "cuda" => Ok(Device::Cuda(0)),
        other => other
            .strip_prefix("cuda:")
            .and_then(|index| index.parse().ok())
            .map(Device::Cuda)
            .ok_or_else(|| ConfigError::Device(other.to_owned())),
    }
}

/// Why a config could not be read.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// The file could not be read.
    #[error("failed to read {}", path.display())]
    Read {
        /// The config file.
        path: PathBuf,
        /// The error reading it.
        #[source]
        source: io::Error,
    },
    /// The file is not valid TOML or not a valid config.
    #[error("failed to parse {}", path.display())]
    Parse {
        /// The config file.
        path: PathBuf,
        /// The error parsing it.
        #[source]
        source: Box<TomlError>,
    },
    /// A device other than `cpu`, `cuda`, or `cuda:N`.
    #[error("device {0}, rather than cpu, cuda, or cuda:N")]
    Device(String),
}

/// The type weights load in: the snapshot's own, or one named.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum DataTypeChoice {
    /// The type the config says the weights are stored in: BF16 or F16 as stored, F32 otherwise.
    #[default]
    Auto,
    /// 32-bit floats.
    F32,
    /// 16-bit IEEE floats.
    F16,
    /// 16-bit brain floats.
    Bf16,
}

impl DataTypeChoice {
    /// Returns the data type to load `source`'s weights in.
    ///
    /// # Errors
    ///
    /// Returns an error if `Auto` cannot read the snapshot's config.
    pub fn resolve(self, source: &Source) -> singe_model::Result<DataType> {
        Ok(match self {
            Self::F32 => DataType::F32,
            Self::F16 => DataType::F16,
            Self::Bf16 => DataType::Bf16,
            Self::Auto => match source.stored_data_type()?.as_deref() {
                Some("bfloat16") => DataType::Bf16,
                Some("float16") => DataType::F16,
                _ => DataType::F32,
            },
        })
    }
}
