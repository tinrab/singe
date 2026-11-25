//! What a server serves and how: read from a TOML file, from the command line, or built in code.

use std::{
    collections::BTreeMap,
    fs, io,
    path::{Path, PathBuf},
    time::Duration,
};

use bytesize::ByteSize;
use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};
use singe::{DataType, Device};
use singe_model::{
    Source,
    tune::{TuneLevel, TuneSettings, Tuning},
};
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
    pub model: Option<String>,
    pub program: Option<PathBuf>,
    pub file: Option<PathBuf>,
    pub processor: Option<String>,
    #[serde(default)]
    pub dims: BTreeMap<String, Vec<i64>>,
    #[serde(default)]
    pub inputs: BTreeMap<String, Vec<i64>>,
    #[serde(default)]
    pub tune: TuneConfig,
    /// The name requests use, or else `model` as given, the program directory's name, or the file's stem.
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
    /// Creates a config for the snapshot `model` with every other field at its default.
    #[must_use]
    pub fn new(model: &str) -> Self {
        Self {
            model: Some(model.to_owned()),
            ..Self::empty()
        }
    }

    /// Creates a config for the exported program in `directory` with every other field at its default.
    #[must_use]
    pub fn program(directory: PathBuf) -> Self {
        Self {
            program: Some(directory),
            ..Self::empty()
        }
    }

    /// Creates a config for the ONNX file at `path` with every other field at its default.
    #[must_use]
    pub fn file(path: PathBuf) -> Self {
        Self {
            file: Some(path),
            ..Self::empty()
        }
    }

    fn empty() -> Self {
        Self {
            model: None,
            program: None,
            file: None,
            processor: None,
            dims: BTreeMap::new(),
            inputs: BTreeMap::new(),
            tune: TuneConfig::default(),
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

    /// Returns the name requests use: `name`, or else `model` as given, the program directory's name, or the file's stem.
    #[must_use]
    pub fn served_name(&self) -> String {
        if let Some(name) = &self.name {
            return name.clone();
        }

        let path = match (&self.program, &self.file) {
            (Some(directory), _) => directory.file_name(),
            (None, Some(file)) => file.file_stem(),
            (None, None) => return self.model.clone().unwrap_or_else(|| "model".to_owned()),
        };
        path.map_or_else(
            || "model".to_owned(),
            |name| name.to_string_lossy().into_owned(),
        )
    }

    pub fn get_source(&self) -> Result<ModelSource<'_>, ConfigError> {
        let source = match (&self.model, &self.program, &self.file) {
            (Some(model), None, None) => ModelSource::Snapshot(model),
            (None, Some(program), None) => ModelSource::Program(program),
            (None, None, Some(file)) => ModelSource::Onnx(file),
            _ => return Err(ConfigError::Source),
        };
        let snapshot = matches!(source, ModelSource::Snapshot(_));
        let file = matches!(source, ModelSource::Onnx(_));
        let given = [
            (
                "processor",
                self.processor.is_some() && snapshot,
                "program and file",
            ),
            ("dims", !self.dims.is_empty() && !file, "file"),
            ("inputs", !self.inputs.is_empty() && !file, "file"),
            ("tune", self.tune != TuneConfig::default() && !file, "file"),
            (
                "data_type",
                self.data_type != DataTypeChoice::Auto && !snapshot,
                "snapshot",
            ),
            ("eager", self.eager && !snapshot, "snapshot"),
            (
                "max_model_len",
                self.max_model_len.is_some() && !snapshot,
                "snapshot",
            ),
            (
                "max_num_seqs",
                self.max_num_seqs != default_max_num_seqs() && !snapshot,
                "snapshot",
            ),
            (
                "cache_tokens",
                self.cache_tokens.is_some() && !snapshot,
                "snapshot",
            ),
            ("seed", self.seed.is_some() && !snapshot, "snapshot"),
        ];
        if let Some(&(option, _, models)) = given.iter().find(|(_, misplaced, _)| *misplaced) {
            return Err(ConfigError::Misplaced { option, models });
        }
        Ok(source)
    }

    /// Returns the device `device` names.
    ///
    /// # Errors
    ///
    /// Returns an error if `device` is not `cpu`, `cuda`, or `cuda:N`.
    pub fn get_device(&self) -> Result<Device, ConfigError> {
        parse_device(&self.device)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSource<'a> {
    Snapshot(&'a str),
    Program(&'a Path),
    Onnx(&'a Path),
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TuneConfig {
    pub enabled: bool,
    pub level: Option<TuneLevel>,
    #[serde(serialize_with = "write_duration", deserialize_with = "read_duration")]
    pub budget: Option<Duration>,
    pub runs: Option<usize>,
    pub schedules: Option<bool>,
    #[serde(serialize_with = "write_duration", deserialize_with = "read_duration")]
    pub plan_budget: Option<Duration>,
    pub seed: Option<u64>,
    pub memory_limit: Option<ByteSize>,
    pub jobs: Option<usize>,
    pub records: Option<PathBuf>,
    pub save_records: Option<PathBuf>,
}

impl TuneConfig {
    /// Returns whether any setting applies only to measuring, which reading `records` skips.
    #[must_use]
    pub const fn has_measuring_settings(&self) -> bool {
        self.level.is_some()
            || self.budget.is_some()
            || self.runs.is_some()
            || self.schedules.is_some()
            || self.plan_budget.is_some()
            || self.seed.is_some()
            || self.memory_limit.is_some()
            || self.jobs.is_some()
            || self.save_records.is_some()
    }

    /// Returns the tuning these settings ask for, or `None` when `enabled` is off.
    ///
    /// # Errors
    ///
    /// Returns an error if settings are given with `enabled` off, or if `records` is given with settings for measuring.
    pub fn get_tuning(&self) -> Result<Option<Tuning>, ConfigError> {
        if !self.enabled {
            if self.has_measuring_settings() || self.records.is_some() {
                return Err(ConfigError::TuneDisabled);
            }

            return Ok(None);
        }

        if self.records.is_some() && self.has_measuring_settings() {
            return Err(ConfigError::TuneRecordsMeasured);
        }

        let settings = TuneSettings {
            level: self.level.unwrap_or_default(),
            budget: self.budget,
            runs: self.runs,
            schedules: self.schedules,
            plan_budget: self.plan_budget,
            seed: self.seed,
            memory_limit: self.memory_limit.as_ref().map(ByteSize::as_u64),
            jobs: self.jobs,
        };
        Ok(Some(Tuning {
            settings,
            records: self.records.clone(),
            save: self.save_records.clone(),
            inputs: None,
        }))
    }
}

#[expect(clippy::ref_option, reason = "serde passes the field by reference")]
fn write_duration<S: Serializer>(
    duration: &Option<Duration>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    duration
        .map(|duration| humantime::format_duration(duration).to_string())
        .serialize(serializer)
}

fn read_duration<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<Duration>, D::Error> {
    Option::<String>::deserialize(deserializer)?
        .map(|text| humantime::parse_duration(&text).map_err(D::Error::custom))
        .transpose()
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
    #[error("a model of other than exactly one of model, program, and file")]
    Source,
    #[error("{option} applies to {models} models")]
    Misplaced {
        option: &'static str,
        models: &'static str,
    },
    /// Tuning settings with tuning off.
    #[error("tuning settings without tune.enabled = true")]
    TuneDisabled,
    /// Tuning records beside settings for measuring, which reading records skips.
    #[error("tuning records with settings for measuring, which reading records skips")]
    TuneRecordsMeasured,
}

/// The type weights load in: the snapshot's own, or one named.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
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
            Self::Auto => source.default_data_type()?,
        })
    }
}
