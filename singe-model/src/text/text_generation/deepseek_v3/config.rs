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
