use serde::Deserialize;
use singe::nn::{Activation, RotaryLayout, RotarySpec};

use crate::{
    config::{DecoderConfig, unsupported},
    error::Result,
};

const fn default_interleave() -> bool {
    true
}

const fn default_true() -> bool {
    true
}

const fn default_one() -> i64 {
    1
}

// The defaults `transformers` gives these fields, which a real checkpoint's config states anyway.
const fn default_moe_intermediate() -> i64 {
    2048
}

const fn default_routed_experts() -> i64 {
    256
}

const fn default_scaling() -> f64 {
    2.5
}

const fn default_groups() -> i64 {
    8
}

const fn default_kept_groups() -> i64 {
    4
}

fn default_topk_method() -> String {
    "noaux_tc".to_owned()
}

fn default_scoring() -> String {
    "sigmoid".to_owned()
}

// The fields of a Hugging Face `config.json` this definition reads.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DeepseekV3Config {
    #[serde(flatten)]
    pub decoder: DecoderConfig,
    pub kv_lora_rank: i64,
    #[serde(default)]
    pub q_lora_rank: Option<i64>,
    pub qk_rope_head_dim: i64,
    pub qk_nope_head_dim: i64,
    pub v_head_dim: i64,
    pub first_k_dense_replace: i64,
    #[serde(default = "default_interleave")]
    pub rope_interleave: bool,
    #[serde(default = "default_moe_intermediate")]
    pub moe_intermediate_size: i64,
    #[serde(default = "default_one")]
    pub n_shared_experts: i64,
    #[serde(default = "default_routed_experts")]
    pub n_routed_experts: i64,
    #[serde(default = "default_scaling")]
    pub routed_scaling_factor: f64,
    #[serde(default = "default_groups")]
    pub n_group: i64,
    #[serde(default = "default_kept_groups")]
    pub topk_group: i64,
    #[serde(default = "default_groups")]
    pub num_experts_per_tok: i64,
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,
    #[serde(default = "default_topk_method")]
    pub topk_method: String,
    #[serde(default = "default_scoring")]
    pub scoring_func: String,
    #[serde(default = "default_one")]
    pub moe_layer_freq: i64,
}

impl DeepseekV3Config {
    // A checkpoint using one of these would load and then compute something else.
    pub fn validate(&self) -> Result<()> {
        self.rotary()?;
        if self.decoder.attention_bias {
            return Err(unsupported("attention bias"));
        }

        if self.first_k_dense_replace < self.decoder.num_hidden_layers {
            self.validate_moe()?;
        }
        Ok(())
    }

    // What `MixtureOfExperts` computes differently, or cannot take.
    fn validate_moe(&self) -> Result<()> {
        if !self.norm_topk_prob {
            return Err(unsupported("unnormalized expert weights"));
        }

        if self.topk_method != "noaux_tc" {
            return Err(unsupported("a routing method other than `noaux_tc`"));
        }

        if self.scoring_func != "sigmoid" {
            return Err(unsupported("expert scores other than sigmoid"));
        }

        if self.moe_layer_freq != 1 {
            return Err(unsupported(
                "a mixture-of-experts layer frequency other than one",
            ));
        }

        if self.decoder.hidden_act != Activation::Silu {
            return Err(unsupported("experts whose gate is not SiLU"));
        }

        if self.decoder.mlp_bias {
            return Err(unsupported("experts with a bias"));
        }

        if self.n_group <= 0 || self.n_routed_experts % self.n_group != 0 {
            return Err(unsupported("expert groups that do not divide the experts"));
        }

        if self.topk_group > self.n_group
            || self.num_experts_per_tok * self.n_group > self.n_routed_experts * self.topk_group
        {
            return Err(unsupported("more experts chosen than kept groups hold"));
        }
        Ok(())
    }

    pub fn rotary(&self) -> Result<RotarySpec> {
        Ok(RotarySpec {
            head_dim: self.qk_rope_head_dim,
            frequencies: self.decoder.rotary.frequencies()?,
            layout: if self.rope_interleave {
                RotaryLayout::Interleaved
            } else {
                RotaryLayout::HalfSplit
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;

    fn config(overrides: &str) -> Result<DeepseekV3Config> {
        let base = r#"{"vocab_size": 8, "hidden_size": 16, "intermediate_size": 32,
            "num_hidden_layers": 2, "num_attention_heads": 2, "kv_lora_rank": 32,
            "qk_rope_head_dim": 16, "qk_nope_head_dim": 8, "v_head_dim": 8,
            "rms_norm_eps": 1e-6, "rope_theta": 100.0, "hidden_act": "silu""#;
        let dense = if overrides.contains("first_k_dense_replace") {
            ""
        } else {
            r#", "first_k_dense_replace": 2"#
        };
        let config: DeepseekV3Config = serde_json::from_str(&format!("{base}{dense}{overrides}}}"))
            .map_err(Error::ConfigFormat)?;
        config.validate()?;
        Ok(config)
    }

    #[test]
    fn test_configs_refuse_what_this_definition_would_compute_wrongly() {
        assert!(config("").is_ok());
        assert!(
            config(r#", "first_k_dense_replace": 1, "n_routed_experts": 8, "n_group": 4, "topk_group": 2, "num_experts_per_tok": 3"#).is_ok(),
            "moe layers"
        );
        for refused in [
            r#""norm_topk_prob": false"#,
            r#""topk_method": "greedy""#,
            r#""scoring_func": "softmax""#,
            r#""moe_layer_freq": 2"#,
            r#""n_group": 3"#,
            r#""num_experts_per_tok": 200"#,
        ] {
            assert!(
                config(&format!(r#", "first_k_dense_replace": 1, {refused}"#)).is_err(),
                "{refused}"
            );
        }
        assert!(matches!(
            config(r#", "attention_bias": true"#),
            Err(Error::UnsupportedConfig(_))
        ));
    }

    #[test]
    fn test_rotary_layout_follows_the_interleave_flag_which_defaults_on() {
        assert_eq!(
            config("").unwrap().rotary().unwrap().layout,
            RotaryLayout::Interleaved
        );
        assert_eq!(
            config(r#", "rope_interleave": false"#)
                .unwrap()
                .rotary()
                .unwrap()
                .layout,
            RotaryLayout::HalfSplit
        );
    }
}
