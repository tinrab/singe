// What Hugging Face configs share, and how a snapshot directory is read.

use std::{
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, de::DeserializeOwned};
use serde_json::Value;
use singe::{
    DataType,
    nn::{Activation, RotaryFrequencies, YarnFrequencies},
};

use crate::{
    assembly::DecoderSpec,
    error::{Error, Result},
};

// The rotary fields of a `config.json`: `rope_parameters`, as `transformers` 5 saves them, or the older `rope_theta` and `rope_scaling`.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[expect(
    clippy::struct_field_names,
    reason = "the fields are named as the config's keys"
)]
pub struct RotaryConfig {
    #[serde(default)]
    rope_parameters: Option<RopeFields>,
    #[serde(default)]
    rope_theta: Option<f64>,
    #[serde(default)]
    rope_scaling: Option<RopeFields>,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
struct RopeFields {
    #[serde(rename = "type", default)]
    kind: Option<String>,
    #[serde(default)]
    rope_type: Option<String>,
    #[serde(default)]
    rope_theta: Option<f64>,
    #[serde(default)]
    partial_rotary_factor: Option<f64>,
    #[serde(default)]
    factor: Option<f64>,
    #[serde(default)]
    original_max_position_embeddings: Option<i64>,
    #[serde(default)]
    beta_fast: Option<f64>,
    #[serde(default)]
    beta_slow: Option<f64>,
    #[serde(default)]
    attention_factor: Option<f64>,
    #[serde(default)]
    mscale: Option<f64>,
    #[serde(default)]
    mscale_all_dim: Option<f64>,
    #[serde(default)]
    truncate: Option<bool>,
}

// The theta `transformers` falls back on.
const DEFAULT_ROTARY_THETA: f64 = 10_000.0;

impl RotaryConfig {
    // The fraction of each head the rotation turns, its leading features; the rest pass through.
    #[must_use]
    pub fn rotated_fraction(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .or(self.rope_scaling.as_ref())
            .and_then(|fields| fields.partial_rotary_factor)
            .unwrap_or(1.0)
    }

    pub fn frequencies(&self) -> Result<RotaryFrequencies> {
        let fields = self.rope_parameters.as_ref().or(self.rope_scaling.as_ref());
        let theta = fields
            .and_then(|fields| fields.rope_theta)
            .or(self.rope_theta)
            .unwrap_or(DEFAULT_ROTARY_THETA);
        let frequencies = RotaryFrequencies::new(theta);
        let Some(fields) = fields else {
            return Ok(frequencies);
        };
        match fields.kind.as_deref().or(fields.rope_type.as_deref()) {
            None | Some("default") => Ok(frequencies),
            Some("yarn") => Ok(frequencies.with_yarn(YarnFrequencies {
                factor: fields
                    .factor
                    .ok_or_else(|| unsupported("yarn without a factor"))?,
                original_max_positions: fields
                    .original_max_position_embeddings
                    .ok_or_else(|| unsupported("yarn without its original length"))?,
                beta_fast: fields.beta_fast.unwrap_or(32.0),
                beta_slow: fields.beta_slow.unwrap_or(1.0),
                attention_factor: fields.attention_factor,
                mscale: fields.mscale,
                mscale_all_dim: fields.mscale_all_dim,
                truncate: fields.truncate.unwrap_or(true),
            })),
            Some(other) => Err(Error::UnsupportedConfig(format!("rotary type {other}"))),
        }
    }
}

// The fields Llama-family configs share.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DecoderConfig {
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub intermediate_size: i64,
    pub num_hidden_layers: i64,
    pub num_attention_heads: i64,
    // As many as there are query heads when absent.
    #[serde(default)]
    pub num_key_value_heads: Option<i64>,
    // The hidden size over the heads when absent.
    #[serde(default)]
    pub head_dim: Option<i64>,
    pub rms_norm_eps: f32,
    #[serde(alias = "hidden_activation")]
    pub hidden_act: Activation,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    // Whether the output head shares the embedding's weight, whose default differs by model.
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(flatten)]
    pub rotary: RotaryConfig,
}

impl DecoderConfig {
    // Refuses a layer type other than full attention, which a definition would otherwise compute as full attention.
    pub fn refuse_sliding_layers(&self) -> Result<()> {
        if self
            .layer_types
            .iter()
            .flatten()
            .any(|layer| layer != "full_attention")
        {
            return Err(unsupported("a layer other than full attention"));
        }
        Ok(())
    }

    // Returns the sizes of a decoder in `data_type`, whose head is tied unless the config says otherwise when `tied` is set.
    pub fn spec(&self, data_type: DataType, tied: bool) -> Result<DecoderSpec> {
        let heads = self.num_attention_heads;
        let head_dim = self.head_dim.unwrap_or(self.hidden_size / heads);
        Ok(DecoderSpec {
            data_type,
            vocab_size: self.vocab_size,
            width: self.hidden_size,
            inner: self.intermediate_size,
            layers: self.num_hidden_layers,
            heads,
            kv_heads: self.num_key_value_heads.unwrap_or(heads),
            head_dim,
            rotary_dim: rotated_features(head_dim, self.rotary.rotated_fraction())?,
            eps: self.rms_norm_eps,
            activation: self.hidden_act,
            attention_bias: self.attention_bias,
            mlp_bias: self.mlp_bias,
            tied: self.tie_word_embeddings.unwrap_or(tied),
            head_norms: false,
            frequencies: self.rotary.frequencies()?,
        })
    }
}

#[must_use]
pub fn unsupported(what: &str) -> Error {
    Error::UnsupportedConfig(what.to_owned())
}

// A Hugging Face snapshot directory.
#[derive(Debug, Clone)]
pub struct Snapshot {
    directory: PathBuf,
}

impl Snapshot {
    #[must_use]
    pub fn new(directory: impl AsRef<Path>) -> Self {
        Self {
            directory: directory.as_ref().to_owned(),
        }
    }

    #[must_use]
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    // Reads `config.json`.
    pub fn config<T: DeserializeOwned>(&self) -> Result<T> {
        self.json("config.json")
    }

    // Reads a JSON file at `path` inside the snapshot, such as `modules.json`.
    pub fn json<T: DeserializeOwned>(&self, path: impl AsRef<Path>) -> Result<T> {
        let text = fs::read_to_string(self.directory.join(path)).map_err(Error::ConfigIo)?;
        serde_json::from_str(&text).map_err(Error::ConfigFormat)
    }

    // Returns the config's `model_type`.
    pub fn model_type(&self) -> Result<String> {
        #[derive(Deserialize)]
        struct Fields {
            model_type: String,
        }
        Ok(self.config::<Fields>()?.model_type)
    }

    // Returns a `diffusers` config's `_class_name`, which it has in place of a `model_type`.
    pub fn class_name(&self) -> Result<String> {
        #[derive(Deserialize)]
        struct Fields {
            #[serde(rename = "_class_name")]
            class_name: String,
        }
        Ok(self.config::<Fields>()?.class_name)
    }

    // Returns the tokens that end generation: `generation_config.json`'s `eos_token_id`, or `config.json`'s or its `text_config`'s, and the tokenizer's end token, which chat models such as Qwen3.5 name only there.
    pub fn stop_tokens(&self) -> Result<Vec<i64>> {
        let generation = fs::read_to_string(self.directory.join("generation_config.json"))
            .ok()
            .map(|text| serde_json::from_str::<Value>(&text).map_err(Error::ConfigFormat))
            .transpose()?;
        let config = self.config::<Value>()?;
        let tokens = generation
            .iter()
            .chain([&config, &config["text_config"]])
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
        let read = |path: &str| {
            fs::read_to_string(self.directory.join(path))
                .ok()
                .map(|text| serde_json::from_str::<Value>(&text).map_err(Error::ConfigFormat))
                .transpose()
        };
        let (Some(config), Some(tokenizer)) =
            (read("tokenizer_config.json")?, read("tokenizer.json")?)
        else {
            return Ok(None);
        };
        let end = &config["eos_token"];
        let Some(name) = end.as_str().or_else(|| end["content"].as_str()) else {
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

// The features of a head of `head_dim` that a rotation of `fraction` of it turns, as `transformers` rounds them down.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "head dimensions are far below where `F64` rounds them, and `transformers` truncates"
)]
pub fn rotated_features(head_dim: i64, fraction: f64) -> Result<i64> {
    let features = (head_dim as f64 * fraction) as i64;
    if features <= 0 || features > head_dim || features % 2 != 0 {
        return Err(Error::UnsupportedConfig(format!(
            "a rotation of {features} of {head_dim} features"
        )));
    }
    Ok(features)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> RotaryConfig {
        serde_json::from_str(text).unwrap()
    }

    #[test]
    fn test_rotary_configs_read_both_formats() {
        let saved = parse(r#"{"rope_parameters": {"rope_type": "default", "rope_theta": 100.0}}"#);
        assert_eq!(saved.frequencies().unwrap(), RotaryFrequencies::new(100.0));
        let older = parse(r#"{"rope_theta": 1000000, "rope_scaling": null}"#);
        assert_eq!(
            older.frequencies().unwrap(),
            RotaryFrequencies::new(1_000_000.0)
        );
        assert_eq!(
            parse("{}").frequencies().unwrap(),
            RotaryFrequencies::new(DEFAULT_ROTARY_THETA)
        );
    }

    #[test]
    fn test_rotary_configs_read_yarn_in_both_formats() {
        let saved = parse(
            r#"{"rope_parameters": {"rope_type": "yarn", "rope_theta": 100.0, "factor": 4.0, "original_max_position_embeddings": 32}}"#,
        );
        let older = parse(
            r#"{"rope_theta": 100.0, "rope_scaling": {"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32}}"#,
        );
        assert_eq!(saved.frequencies().unwrap(), older.frequencies().unwrap());
        assert!(saved.frequencies().unwrap().yarn.is_some());
    }

    #[test]
    fn test_rotary_configs_refuse_what_they_would_compute_wrongly() {
        for text in [
            r#"{"rope_parameters": {"rope_type": "linear", "factor": 2.0}}"#,
            r#"{"rope_scaling": {"type": "llama3"}}"#,
            r#"{"rope_parameters": {"rope_type": "yarn"}}"#,
        ] {
            assert!(
                matches!(parse(text).frequencies(), Err(Error::UnsupportedConfig(_))),
                "{text}"
            );
        }
    }
}
