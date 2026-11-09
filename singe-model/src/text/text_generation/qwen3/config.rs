use serde::Deserialize;

use crate::{
    config::{DecoderConfig, unsupported},
    error::Result,
};

// The fields of a Hugging Face `config.json` this definition reads.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Qwen3Config {
    #[serde(flatten)]
    pub decoder: DecoderConfig,
    #[serde(default)]
    pub use_sliding_window: bool,
}

impl Qwen3Config {
    // A checkpoint using one of these would load and then compute something else.
    pub fn validate(&self) -> Result<()> {
        self.decoder.rotary.frequencies()?;
        self.decoder.refuse_sliding_layers()?;
        if self.use_sliding_window {
            return Err(unsupported("a sliding window"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;

    fn config(overrides: &str) -> Result<Qwen3Config> {
        let base = r#"{"vocab_size": 8, "hidden_size": 16, "intermediate_size": 32,
            "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 16, "rms_norm_eps": 1e-6"#;
        // A later key of the same name would be a duplicate, so overrides carry these if they change them.
        let hidden_act = if overrides.contains("hidden_act") {
            ""
        } else {
            r#", "hidden_act": "silu""#
        };
        let config: Qwen3Config = serde_json::from_str(&format!("{base}{hidden_act}{overrides}}}"))
            .map_err(Error::ConfigFormat)?;
        config.validate()?;
        Ok(config)
    }

    #[test]
    fn test_configs_refuse_what_this_definition_would_compute_wrongly() {
        assert!(config("").is_ok());
        for overrides in [
            r#", "use_sliding_window": true"#,
            r#", "layer_types": ["full_attention", "sliding_attention"]"#,
            r#", "rope_scaling": {"rope_type": "linear"}"#,
        ] {
            assert!(
                matches!(config(overrides), Err(Error::UnsupportedConfig(_))),
                "{overrides}"
            );
        }
    }

    #[test]
    fn test_configs_refuse_an_activation_they_do_not_know() {
        assert!(matches!(
            config(r#", "hidden_act": "cube""#),
            Err(Error::ConfigFormat(_))
        ));
    }

    #[test]
    fn test_configs_take_what_this_definition_computes() {
        for overrides in [
            r#", "attention_bias": true"#,
            r#", "tie_word_embeddings": false"#,
            r#", "layer_types": ["full_attention"]"#,
            r#", "rope_parameters": {"rope_type": "default", "rope_theta": 100.0}"#,
        ] {
            assert!(config(overrides).is_ok(), "{overrides}");
        }
    }
}
