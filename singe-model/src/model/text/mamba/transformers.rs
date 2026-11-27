//! Reads a `transformers` Mamba config and imports its checkpoint, whose names start with `backbone`.

use serde::Deserialize;
use singe::nn::MambaSpec;

use crate::{
    error::{Error, Result},
    format::{RuleSet, transformers::default_true},
    model::text::mamba::MambaConfig,
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
enum Rank {
    Given(i64),
    Derived(String),
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    vocab_size: i64,
    hidden_size: i64,
    num_hidden_layers: i64,
    state_size: i64,
    expand: i64,
    conv_kernel: i64,
    time_step_rank: Rank,
    #[serde(default)]
    use_bias: bool,
    #[serde(default = "default_true")]
    use_conv_bias: bool,
    layer_norm_epsilon: f32,
    #[serde(default = "default_true")]
    tie_word_embeddings: bool,
    #[serde(default = "default_silu")]
    hidden_act: String,
}

fn default_silu() -> String {
    "silu".to_owned()
}

pub(crate) fn config(source: &Source) -> Result<MambaConfig> {
    let fields: Fields = source.json("config.json")?;
    if fields.hidden_act != "silu" {
        return Err(Error::unsupported("a mamba activation other than silu"));
    }

    let rank = match fields.time_step_rank {
        Rank::Given(rank) => rank,
        Rank::Derived(rank) if rank == "auto" => (fields.hidden_size + 15) / 16,
        Rank::Derived(_) => return Err(Error::unsupported("a mamba time step rank")),
    };
    Ok(MambaConfig {
        vocab_size: fields.vocab_size,
        width: fields.hidden_size,
        layers: fields.num_hidden_layers,
        mixer: MambaSpec {
            inner: fields.expand * fields.hidden_size,
            state: fields.state_size,
            rank,
            kernel: fields.conv_kernel,
        },
        bias: fields.use_bias,
        convolution_bias: fields.use_conv_bias,
        eps: fields.layer_norm_epsilon,
        tied: fields.tie_word_embeddings,
    })
}

pub(crate) fn rules() -> RuleSet {
    RuleSet::new()
        .rename("backbone.embeddings", "embeddings.tokens")
        .rename_each(
            "backbone.layers.*.mixer",
            "blocks.*.mixer",
            &[
                ("in_proj", "input"),
                ("conv1d", "convolution"),
                ("x_proj", "projection"),
                ("dt_proj", "step"),
                ("A_log", "decay_log"),
                ("D", "skip"),
                ("out_proj", "output"),
            ],
        )
        .rename("backbone.layers.*.norm", "blocks.*.norm")
        .rename("backbone.norm_f", "norm")
        .rename("lm_head", "head")
}
