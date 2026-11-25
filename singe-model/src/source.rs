//! Where a model is read from: a directory in one of the formats [`format`](crate::format) reads, and the options a load takes.

use std::{
    env, fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use serde::{Deserialize, de::DeserializeOwned};
use serde_json::Value;
use singe::{DataType, Device};

use crate::{
    error::{Error, Result},
    format::NativeConfig,
    function::TensorFunction,
    processor::text::special_token,
};

/// The layout of a model directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// Singe's own: its configs and state names, written by `format::native::save`.
    Native,
    /// A Hugging Face snapshot `transformers` reads, whose `config.json` names a `model_type`.
    Transformers,
    /// A Hugging Face snapshot `diffusers` reads, whose `config.json` names a `_class_name`, or a pipeline of them, whose `model_index.json` names its class and its components' directories.
    Diffusers,
    /// A Hugging Face snapshot of `AutoGluon`'s `Tab2D`, which Mitra is, whose `config.json` names its `task` and its sizes but no model; its kind is `tab2d`.
    AutoGluon,
}

/// A model directory, its format, its model kind, and the external function that stands in for its weights, when one is attached.
#[derive(Debug, Clone)]
pub struct Source {
    directory: PathBuf,
    format: Format,
    /// The model kind: a native config's `model`, a `model_type`, or a `_class_name`.
    kind: String,
    function: Option<Arc<dyn TensorFunction>>,
}

impl Source {
    /// Opens `directory`, telling its format from its `config.json`, or from a `diffusers` pipeline's `model_index.json`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file it reads, `config.json` or else `model_index.json`, cannot be read or parsed, or if it names no model kind.
    pub fn open(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref().to_owned();
        let index = directory.join("model_index.json");
        let config = if directory.join("config.json").exists() || !index.exists() {
            read_json::<Value>(&directory.join("config.json"))?
        } else {
            read_json::<Value>(&index)?
        };
        let field = |name: &str| config[name].as_str().map(str::to_owned);
        let (format, kind) = if config.get("singe").is_some() {
            (Format::Native, NativeConfig::read(&directory)?.model)
        } else if let Some(kind) = field("model_type") {
            (Format::Transformers, kind)
        } else if let Some(kind) = field("_class_name") {
            (Format::Diffusers, kind)
        } else if config["task"].is_string() && config["n_layers"].is_number() {
            (Format::AutoGluon, "tab2d".to_owned())
        } else {
            return Err(Error::unsupported(format!(
                "a model directory whose config names no model: {}",
                directory.display()
            )));
        };
        Ok(Self {
            directory,
            format,
            kind,
            function: None,
        })
    }

    /// Opens `name`: a model directory, or a Hugging Face repository, as `owner/name` or `owner/name@revision`, whose snapshot the local cache holds.
    ///
    /// # Errors
    ///
    /// Returns an error if `name` is neither a directory nor a repository, if the cache holds no snapshot of it, or if [`Source::open`] fails on the directory.
    pub fn find(name: &str) -> Result<Self> {
        if Path::new(name).is_dir() {
            return Self::open(name);
        }

        let (repository, revision) = name.split_once('@').unwrap_or((name, "main"));
        if !repository.contains('/') {
            return Err(Error::invalid(format!(
                "model {name}, which is neither a directory nor a repository"
            )));
        }

        let snapshot = cached_snapshot(repository, revision).ok_or_else(|| {
            Error::invalid(format!(
                "model {name}, which the Hugging Face cache does not hold"
            ))
        })?;
        Self::open(snapshot)
    }

    /// Opens the component `name` of a `diffusers` pipeline, such as its `unet` or `vae`, as its own source.
    ///
    /// # Errors
    ///
    /// Returns an error if `model_index.json` cannot be read or does not list `name`, or if [`Source::open`] fails on the component's directory.
    pub fn component(&self, name: &str) -> Result<Self> {
        let index = self.json::<Value>("model_index.json")?;
        if !index[name].is_array() {
            return Err(Error::unsupported(format!(
                "a pipeline without the component {name}"
            )));
        }

        Self::open(self.directory.join(name))
    }

    /// Returns this source read from its subdirectory `name`, with the same format and kind, for a pipeline's directory such as `tokenizer` that holds only a processor's config.
    #[must_use]
    pub fn subdirectory(&self, name: &str) -> Self {
        Self {
            directory: self.directory.join(name),
            format: self.format,
            kind: self.kind.clone(),
            function: self.function.clone(),
        }
    }

    #[must_use]
    pub fn with_function(self, function: Arc<dyn TensorFunction>) -> Self {
        Self {
            function: Some(function),
            ..self
        }
    }

    #[must_use]
    pub fn function(&self) -> Option<&Arc<dyn TensorFunction>> {
        self.function.as_ref()
    }

    #[must_use]
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    #[must_use]
    pub const fn format(&self) -> Format {
        self.format
    }

    /// Returns the model kind the registry looks up.
    #[must_use]
    pub fn kind(&self) -> &str {
        &self.kind
    }

    /// Reads the JSON file at `path` inside the directory.
    pub fn json<T: DeserializeOwned>(&self, path: impl AsRef<Path>) -> Result<T> {
        read_json(&self.directory.join(path))
    }

    /// Reads the JSON file at `path` inside the directory, or returns `None` if there is none.
    pub fn optional_json<T: DeserializeOwned>(&self, path: impl AsRef<Path>) -> Result<Option<T>> {
        let path = self.directory.join(path);
        if !path.exists() {
            return Ok(None);
        }
        read_json(&path).map(Some)
    }

    /// Returns the `transformers` classes a config's `architectures` names, whose weights the snapshot holds, or no classes for another format.
    pub fn classes(&self) -> Result<Vec<String>> {
        if self.format != Format::Transformers {
            return Ok(Vec::new());
        }

        let config = self.json::<Value>("config.json")?;
        Ok(config["architectures"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect())
    }

    #[must_use]
    pub fn has(&self, path: impl AsRef<Path>) -> bool {
        self.directory.join(path).exists()
    }

    /// Returns the most tokens a sequence holds, prompt included: a config's `max_position_embeddings`, or its `n_positions` as GPT-2 names it, or either field of its `text_config`.
    /// Returns `None` without a `config.json` or either field.
    pub fn context_length(&self) -> Result<Option<usize>> {
        let Some(config) = self.optional_json::<Value>("config.json")? else {
            return Ok(None);
        };
        Ok([&config, &config["text_config"]]
            .into_iter()
            .flat_map(|fields| ["max_position_embeddings", "n_positions"].map(|key| &fields[key]))
            .find_map(Value::as_u64)
            .and_then(|length| usize::try_from(length).ok()))
    }

    /// Returns the type a config says its weights are stored in, such as `bfloat16`: a native config's, or `config.json`'s `torch_dtype` or `dtype`, or its `text_config`'s.
    pub fn stored_data_type(&self) -> Result<Option<String>> {
        if self.format == Format::Native {
            return Ok(Some(NativeConfig::read(&self.directory)?.data_type));
        }

        let Some(config) = self.optional_json::<Value>("config.json")? else {
            return Ok(None);
        };
        Ok([&config, &config["text_config"]]
            .into_iter()
            .flat_map(|fields| ["torch_dtype", "dtype"].map(|key| &fields[key]))
            .find_map(Value::as_str)
            .map(str::to_owned))
    }

    /// Returns the float type a load defaults to: [`DataType::Bf16`] or [`DataType::F16`] when the config says the weights are stored as `bfloat16` or `float16`, [`DataType::F32`] otherwise.
    pub fn default_data_type(&self) -> Result<DataType> {
        Ok(match self.stored_data_type()?.as_deref() {
            Some("bfloat16") => DataType::Bf16,
            Some("float16") => DataType::F16,
            _ => DataType::F32,
        })
    }

    /// Returns the tokens that end generation.
    /// A native config records them.
    /// Otherwise they are the `eos_token_id` of the first of `generation_config.json`, a `transformers` config, and its `text_config` that names one, and the tokenizer's end token, which chat models such as Qwen3.5 name only there.
    pub fn stop_tokens(&self) -> Result<Vec<i64>> {
        if self.format == Format::Native {
            return Ok(NativeConfig::read(&self.directory)?.stop_tokens);
        }

        let generation = self.optional_json::<Value>("generation_config.json")?;
        let config = match self.format {
            Format::Transformers => Some(self.json::<Value>("config.json")?),
            _ => None,
        };
        let tokens = generation
            .iter()
            .chain(
                config
                    .iter()
                    .flat_map(|config| [config, &config["text_config"]]),
            )
            .map(|fields| &fields["eos_token_id"])
            .find(|tokens| !tokens.is_null());
        let mut tokens = tokens.and_then(token_ids).unwrap_or_default();
        if let Some(token) = self.tokenizer_end_token()?
            && !tokens.contains(&token)
        {
            tokens.push(token);
        }
        Ok(tokens)
    }

    /// Returns the id of `tokenizer_config.json`'s `eos_token` among `tokenizer.json`'s added tokens, if both files name it.
    fn tokenizer_end_token(&self) -> Result<Option<i64>> {
        let Some(config) = self.optional_json::<Value>("tokenizer_config.json")? else {
            return Ok(None);
        };
        let Some(name) = special_token(&config, "eos_token") else {
            return Ok(None);
        };
        // Only the added tokens are read, so the vocabulary and merges, most of the file, are skipped rather than built into values.
        let Some(tokenizer) = self.optional_json::<AddedTokens>("tokenizer.json")? else {
            return Ok(None);
        };
        Ok(tokenizer
            .added_tokens
            .iter()
            .find(|token| token.content == name)
            .map(|token| token.id))
    }
}

/// The added tokens of a `tokenizer.json`.
#[derive(Deserialize)]
struct AddedTokens {
    #[serde(default)]
    added_tokens: Vec<AddedToken>,
}

#[derive(Deserialize)]
struct AddedToken {
    id: i64,
    content: String,
}

/// Returns the Hugging Face cache, as `huggingface_hub` finds it: `HF_HUB_CACHE`, or `hub` under `HF_HOME` or `~/.cache/huggingface`.
fn hugging_face_cache() -> Option<PathBuf> {
    env::var_os("HF_HUB_CACHE").map(PathBuf::from).or_else(|| {
        env::var_os("HF_HOME")
            .map(PathBuf::from)
            .or_else(|| env::var_os("HOME").map(|home| Path::new(&home).join(".cache/huggingface")))
            .map(|home| home.join("hub"))
    })
}

/// Returns the cached snapshot of `repository` that `refs/<revision>` names, as `huggingface_hub` resolves a revision offline, such as `refs/refs/pr/3` for `refs/pr/3`.
/// Without such a ref, `revision` names the snapshot itself, as a commit hash does.
#[must_use]
pub fn cached_snapshot(repository: &str, revision: &str) -> Option<PathBuf> {
    let directory =
        hugging_face_cache()?.join(format!("models--{}", repository.replace('/', "--")));
    let commit = fs::read_to_string(directory.join("refs").join(revision))
        .map_or_else(|_| revision.to_owned(), |commit| commit.trim().to_owned());
    let snapshot = directory.join("snapshots").join(commit);
    snapshot.is_dir().then_some(snapshot)
}

pub(crate) fn token_ids(value: &Value) -> Option<Vec<i64>> {
    match value {
        Value::Number(token) => Some(token.as_i64().into_iter().collect()),
        Value::Array(tokens) => Some(tokens.iter().filter_map(Value::as_i64).collect()),
        _ => None,
    }
}

pub(crate) fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let text = fs::read_to_string(path).map_err(Error::io(path))?;
    serde_json::from_str(&text).map_err(Error::json(path))
}

/// Where a load puts a model's weights, in what type, and whether it joins sibling projections.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadOptions {
    pub device: Device,
    pub data_type: DataType,
    /// Whether each group of sibling projections is stored joined, so eager layers run it as one matmul.
    /// An export, which traces its graphs and runs no layer eagerly, keeps each weight apart.
    pub join_projections: bool,
}

impl LoadOptions {
    #[must_use]
    pub const fn new(device: Device, data_type: DataType) -> Self {
        Self {
            device,
            data_type,
            join_projections: true,
        }
    }
}
