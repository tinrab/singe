// Where a model is read from: a directory in one of the formats `format` reads, and the options a load takes.

use std::{
    fs,
    path::{Path, PathBuf},
};

use serde::de::DeserializeOwned;
use serde_json::Value;
use singe::{DataType, Device};

use crate::{
    error::{Error, Result},
    processor::text::special_token,
};

// The layout of a model directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    // Singe's own: its configs and state names, written by `format::native::save`.
    Native,
    // A Hugging Face snapshot `transformers` reads, whose `config.json` names a `model_type`.
    Transformers,
    // A Hugging Face snapshot `diffusers` reads, whose `config.json` names a `_class_name`, or a pipeline of them, whose `model_index.json` names its class and its components' directories.
    Diffusers,
    // A Hugging Face snapshot of AutoGluon's Tab2D, which Mitra is, whose `config.json` names its `task` and its sizes but no model; its kind is `tab2d`.
    AutoGluon,
}

// A model directory and its format.
#[derive(Debug, Clone)]
pub struct Source {
    directory: PathBuf,
    format: Format,
    // The model kind: a native config's `model`, a `model_type`, or a `_class_name`.
    kind: String,
}

impl Source {
    // Opens `directory`, telling its format from its `config.json`, or from a `diffusers` pipeline's `model_index.json`.
    pub fn open(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref().to_owned();
        let index = directory.join("model_index.json");
        let config = if directory.join("config.json").exists() || !index.exists() {
            read_json::<Value>(&directory.join("config.json"))?
        } else {
            read_json::<Value>(&index)?
        };
        let field = |name: &str| config[name].as_str().map(str::to_owned);
        let (format, kind) = if let Some(kind) = config["singe"]["model"].as_str() {
            (Format::Native, kind.to_owned())
        } else if let Some(kind) = field("model_type") {
            (Format::Transformers, kind)
        } else if let Some(kind) = field("_class_name") {
            (Format::Diffusers, kind)
        } else if config["task"].is_string() && config["n_layers"].is_number() {
            (Format::AutoGluon, "tab2d".to_owned())
        } else {
            return Err(Error::UnsupportedConfig(format!(
                "a model directory whose config names no model: {}",
                directory.display()
            )));
        };
        Ok(Self {
            directory,
            format,
            kind,
        })
    }

    // Opens the component `name` of a `diffusers` pipeline, such as its `unet` or `vae`, as its own source.
    pub fn component(&self, name: &str) -> Result<Self> {
        let index = self.json::<Value>("model_index.json")?;
        if !index[name].is_array() {
            return Err(Error::UnsupportedConfig(format!(
                "a pipeline without the component {name}"
            )));
        }

        Self::open(self.directory.join(name))
    }

    // The same source read from its subdirectory `name`, as a pipeline's processor directory holds only its processor's config.
    #[must_use]
    pub fn subdirectory(&self, name: &str) -> Self {
        Self {
            directory: self.directory.join(name),
            format: self.format,
            kind: self.kind.clone(),
        }
    }

    #[must_use]
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    #[must_use]
    pub const fn format(&self) -> Format {
        self.format
    }

    // Returns the model kind the registry looks up.
    #[must_use]
    pub fn kind(&self) -> &str {
        &self.kind
    }

    // Reads a JSON file at `path` inside the directory.
    pub fn json<T: DeserializeOwned>(&self, path: impl AsRef<Path>) -> Result<T> {
        read_json(&self.directory.join(path))
    }

    // Reads a JSON file at `path` inside the directory, if it is there.
    pub fn optional_json<T: DeserializeOwned>(&self, path: impl AsRef<Path>) -> Result<Option<T>> {
        let path = self.directory.join(path);
        if !path.exists() {
            return Ok(None);
        }
        read_json(&path).map(Some)
    }

    // The `transformers` classes a config's `architectures` names, whose weights the snapshot holds; none for another format.
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

    // Returns the tokens that end generation: those a native config records, or `generation_config.json`'s `eos_token_id`, a `transformers` config's or its `text_config`'s, and the tokenizer's end token, which chat models such as Qwen3.5 name only there.
    pub fn stop_tokens(&self) -> Result<Vec<i64>> {
        if self.format == Format::Native {
            let config = self.json::<Value>("config.json")?;
            return serde_json::from_value(config["singe"]["stop_tokens"].clone())
                .map_err(Error::ConfigFormat);
        }

        let generation = self.optional_json::<Value>("generation_config.json")?;
        let config = match self.format {
            Format::Transformers => Some(self.json::<Value>("config.json")?),
            Format::Native | Format::Diffusers | Format::AutoGluon => None,
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
        let mut tokens = match tokens {
            Some(Value::Number(token)) => token.as_i64().into_iter().collect(),
            Some(Value::Array(tokens)) => tokens.iter().filter_map(Value::as_i64).collect(),
            _ => Vec::new(),
        };
        if let Some(token) = self.tokenizer_end_token()?
            && !tokens.contains(&token)
        {
            tokens.push(token);
        }
        Ok(tokens)
    }

    // The id of `tokenizer_config.json`'s `eos_token`, among `tokenizer.json`'s added tokens, if both files name it.
    fn tokenizer_end_token(&self) -> Result<Option<i64>> {
        let (Some(config), Some(tokenizer)) = (
            self.optional_json::<Value>("tokenizer_config.json")?,
            self.optional_json::<Value>("tokenizer.json")?,
        ) else {
            return Ok(None);
        };
        let Some(name) = special_token(&config, "eos_token") else {
            return Ok(None);
        };
        Ok(tokenizer["added_tokens"]
            .as_array()
            .into_iter()
            .flatten()
            .find(|token| token["content"].as_str() == Some(name))
            .and_then(|token| token["id"].as_i64()))
    }
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let text = fs::read_to_string(path).map_err(Error::ConfigIo)?;
    serde_json::from_str(&text).map_err(Error::ConfigFormat)
}

// Where a model's weights go and in what type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadOptions {
    pub device: Device,
    pub data_type: DataType,
}

impl LoadOptions {
    #[must_use]
    pub const fn new(device: Device, data_type: DataType) -> Self {
        Self { device, data_type }
    }
}
