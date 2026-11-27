//! Reads a `transformers` `BitNet` config and imports its checkpoint, whose projection weights are stored four ternary rows to a byte.

use serde::Deserialize;

use crate::{
    assembly::{DecoderConfig, llama_rules},
    error::{Error, Result},
    format::{
        RuleSet,
        transformers::{DecoderDefaults, DecoderFields},
    },
    source::Source,
};

/// Each projection, as `transformers` and Singe name it.
const PROJECTIONS: &[(&str, &str)] = &[
    ("self_attn.q_proj", "attention.query"),
    ("self_attn.k_proj", "attention.key"),
    ("self_attn.v_proj", "attention.value"),
    ("self_attn.o_proj", "attention.output"),
    ("mlp.gate_proj", "feed_forward.gate"),
    ("mlp.up_proj", "feed_forward.up"),
    ("mlp.down_proj", "feed_forward.down"),
];

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct QuantizationFields {
    linear_class: String,
    quantization_mode: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    #[serde(flatten)]
    decoder: DecoderFields,
    quantization_config: QuantizationFields,
}

/// Returns the decoder's config, refusing checkpoints other than `AutoBitLinear`'s offline ones.
pub(crate) fn config(source: &Source) -> Result<DecoderConfig> {
    let fields: Fields = source.json("config.json")?;
    let quantization = &fields.quantization_config;
    if quantization.linear_class != "autobitlinear" || quantization.quantization_mode != "offline" {
        return Err(Error::unsupported(format!(
            "a BitNet checkpoint of {} weights quantized {}, rather than `autobitlinear` ones quantized offline",
            quantization.linear_class, quantization.quantization_mode
        )));
    }

    fields.decoder.refuse_sliding_layers()?;
    fields.decoder.config(DecoderDefaults {
        tied: true,
        head_dim: None,
    })
}

pub(crate) fn rules() -> RuleSet {
    let layers = "model.layers.*";
    let mut rules = RuleSet::new().rename_each(
        layers,
        "blocks.*",
        &[
            ("self_attn.attn_sub_norm", "attention.output_norm"),
            ("mlp.ffn_sub_norm", "feed_forward.down_norm"),
        ],
    );
    for &(from, to) in PROJECTIONS {
        rules = rules
            .rename(
                &format!("{layers}.{from}.weight"),
                &format!("blocks.*.{to}.weight"),
            )
            .rename(
                &format!("{layers}.{from}.weight_scale"),
                &format!("blocks.*.{to}.scale"),
            );
    }
    rules.then(llama_rules("model"))
}
