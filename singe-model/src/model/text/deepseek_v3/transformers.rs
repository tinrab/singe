// Reads a `transformers` DeepSeek-V3 config, and imports its checkpoint.

use serde::Deserialize;
use singe::nn::Activation;

use super::{DeepseekV3Config, ExpertsConfig, LatentAttentionConfig};
use crate::{
    assembly::{DecoderConfig, expert_rules, llama_rules},
    error::{Error, Result},
    format::{
        Rules,
        transformers::{DecoderDefaults, DecoderFields},
    },
    source::Source,
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

const fn default_experts_per_token() -> i64 {
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

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    #[serde(flatten)]
    decoder: DecoderFields,
    kv_lora_rank: i64,
    #[serde(default)]
    q_lora_rank: Option<i64>,
    qk_rope_head_dim: i64,
    qk_nope_head_dim: i64,
    v_head_dim: i64,
    first_k_dense_replace: i64,
    #[serde(default = "default_interleave")]
    rope_interleave: bool,
    #[serde(default = "default_moe_intermediate")]
    moe_intermediate_size: i64,
    #[serde(default = "default_one")]
    n_shared_experts: i64,
    #[serde(default = "default_routed_experts")]
    n_routed_experts: i64,
    #[serde(default = "default_scaling")]
    routed_scaling_factor: f64,
    #[serde(default = "default_groups")]
    n_group: i64,
    #[serde(default = "default_kept_groups")]
    topk_group: i64,
    #[serde(default = "default_experts_per_token")]
    num_experts_per_tok: i64,
    #[serde(default = "default_true")]
    norm_topk_prob: bool,
    #[serde(default = "default_topk_method")]
    topk_method: String,
    #[serde(default = "default_scoring")]
    scoring_func: String,
    #[serde(default = "default_one")]
    moe_layer_freq: i64,
}

impl Fields {
    // A checkpoint using one of these would load and then compute something else.
    fn validate(&self) -> Result<()> {
        if self.decoder.attention_bias {
            return Err(Error::unsupported("attention bias"));
        }

        if self.first_k_dense_replace < self.decoder.num_hidden_layers {
            self.validate_moe()?;
        }
        Ok(())
    }

    // What `MixtureOfExperts` computes differently, or cannot take.
    fn validate_moe(&self) -> Result<()> {
        if !self.norm_topk_prob {
            return Err(Error::unsupported("unnormalized expert weights"));
        }

        if self.topk_method != "noaux_tc" {
            return Err(Error::unsupported("a routing method other than `noaux_tc`"));
        }

        if self.scoring_func != "sigmoid" {
            return Err(Error::unsupported("expert scores other than sigmoid"));
        }

        if self.moe_layer_freq != 1 {
            return Err(Error::unsupported(
                "a mixture-of-experts layer frequency other than one",
            ));
        }

        if self.decoder.hidden_act != Activation::Silu {
            return Err(Error::unsupported("experts whose gate is not SiLU"));
        }

        if self.decoder.mlp_bias {
            return Err(Error::unsupported("experts with a bias"));
        }

        if self.n_group <= 0 || self.n_routed_experts % self.n_group != 0 {
            return Err(Error::unsupported(
                "expert groups that do not divide the experts",
            ));
        }

        if self.topk_group > self.n_group
            || self.num_experts_per_tok * self.n_group > self.n_routed_experts * self.topk_group
        {
            return Err(Error::unsupported(
                "more experts chosen than kept groups hold",
            ));
        }
        Ok(())
    }
}

pub(crate) fn config(source: &Source) -> Result<DeepseekV3Config> {
    let fields: Fields = source.json("config.json")?;
    fields.validate()?;
    // Each query and key head is `qk_nope_head_dim` features no rotation turns, then `qk_rope_head_dim` it turns.
    let decoder = DecoderConfig {
        head_dim: fields.qk_nope_head_dim + fields.qk_rope_head_dim,
        rotary_dim: fields.qk_rope_head_dim,
        ..fields.decoder.config(DecoderDefaults {
            tied: false,
            head_dim: None,
        })?
    };
    Ok(DeepseekV3Config {
        decoder,
        attention: LatentAttentionConfig {
            query_rank: fields.q_lora_rank,
            latent_dim: fields.kv_lora_rank,
            key_dim: fields.qk_nope_head_dim,
            value_dim: fields.v_head_dim,
            interleaved: fields.rope_interleave,
        },
        dense_blocks: fields.first_k_dense_replace,
        experts: ExpertsConfig {
            count: fields.n_routed_experts,
            inner: fields.moe_intermediate_size,
            shared_inner: fields.moe_intermediate_size * fields.n_shared_experts,
            top_k: fields.num_experts_per_tok,
            groups: fields.n_group,
            kept_groups: fields.topk_group,
            scale: fields.routed_scaling_factor,
        },
    })
}

// Rules for latent attention, routers, and shared experts go before the experts' and Llama's, which import the rest.
pub(crate) fn rules(config: &DeepseekV3Config) -> Rules {
    let layer = |name: &str| format!("model.layers.*.{name}");
    // The checkpoints store a multi-token-prediction layer after the last block, which `transformers` leaves unread.
    let draft = Rules::new().skip(&format!("model.layers.{}", config.decoder.layers));
    [
        ("self_attn.q_a_proj", "attention.query_compress"),
        ("self_attn.q_a_layernorm", "attention.query_latent_norm"),
        ("self_attn.q_b_proj", "attention.query_expand"),
        ("self_attn.kv_a_proj_with_mqa", "attention.latent_compress"),
        ("self_attn.kv_a_layernorm", "attention.latent_norm"),
        ("self_attn.kv_b_proj", "attention.latent_expand"),
        ("mlp.gate.weight", "feed_forward.router.weight"),
        (
            "mlp.gate.e_score_correction_bias",
            "feed_forward.router.bias",
        ),
        ("mlp.shared_experts.gate_proj", "feed_forward.shared.gate"),
        ("mlp.shared_experts.up_proj", "feed_forward.shared.up"),
        ("mlp.shared_experts.down_proj", "feed_forward.shared.down"),
    ]
    .into_iter()
    .fold(draft, |rules, (name, target)| {
        rules.rename(&layer(name), &format!("blocks.*.{target}"))
    })
    .then(expert_rules("model", config.experts.inner))
    .then(llama_rules("model"))
}
