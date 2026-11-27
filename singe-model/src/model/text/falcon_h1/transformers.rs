//! Reads a `transformers` Falcon-H1 config and imports its checkpoint, folding the μP multipliers inside each block into the weights they follow.

use std::iter;

use serde::Deserialize;
use singe::{DataType, Error as SingeError, Shape, Tensor, nn::Mamba2Spec};

use crate::{
    assembly::{llama_rules, rename_scaled},
    error::{Error, Result},
    format::{
        Rules,
        transformers::{DecoderDefaults, DecoderFields, default_true},
    },
    model::text::falcon_h1::{FalconH1Config, FalconH1Multipliers},
    source::Source,
};

/// Whether the mixer's projections and convolution have biases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
struct MambaBiasFields {
    #[serde(default)]
    mamba_proj_bias: bool,
    #[serde(default = "default_true")]
    mamba_conv_bias: bool,
}

/// Whether the mixer norms its output, and before or after the gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
struct MambaNormFields {
    #[serde(default)]
    mamba_rms_norm: bool,
    #[serde(default = "default_true")]
    mamba_norm_before_gate: bool,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    #[serde(flatten)]
    decoder: DecoderFields,
    mamba_n_heads: i64,
    mamba_d_head: i64,
    #[serde(default)]
    mamba_d_ssm: Option<i64>,
    mamba_n_groups: i64,
    mamba_d_state: i64,
    mamba_d_conv: i64,
    #[serde(flatten)]
    mamba_biases: MambaBiasFields,
    #[serde(flatten)]
    mamba_norm: MambaNormFields,
    embedding_multiplier: f32,
    lm_head_multiplier: f32,
    attention_in_multiplier: f32,
    attention_out_multiplier: f32,
    key_multiplier: f32,
    ssm_in_multiplier: f32,
    ssm_out_multiplier: f32,
    ssm_multipliers: [f32; 5],
    mlp_multipliers: [f32; 2],
}

pub(crate) fn config(source: &Source) -> Result<FalconH1Config> {
    let fields: Fields = source.json("config.json")?;
    let mixer = Mamba2Spec {
        heads: fields.mamba_n_heads,
        head_dim: fields.mamba_d_head,
        groups: fields.mamba_n_groups,
        state: fields.mamba_d_state,
        kernel: fields.mamba_d_conv,
    };
    if fields
        .mamba_d_ssm
        .is_some_and(|inner| inner != mixer.inner())
    {
        return Err(Error::unsupported(
            "a falcon-h1 mixer whose width is not its heads' features",
        ));
    }

    if fields.mamba_norm.mamba_rms_norm && fields.mamba_norm.mamba_norm_before_gate {
        return Err(Error::unsupported("a falcon-h1 mixer norm before its gate"));
    }

    let decoder = fields.decoder.config(DecoderDefaults {
        tied: false,
        head_dim: None,
    })?;
    let [gate, down] = fields.mlp_multipliers;
    Ok(FalconH1Config {
        decoder,
        mixer,
        mixer_norm: fields.mamba_norm.mamba_rms_norm,
        mixer_bias: fields.mamba_biases.mamba_proj_bias,
        convolution_bias: fields.mamba_biases.mamba_conv_bias,
        embedding_scale: fields.embedding_multiplier,
        logit_scale: fields.lm_head_multiplier,
        multipliers: FalconH1Multipliers {
            attention_in: fields.attention_in_multiplier,
            attention_out: fields.attention_out_multiplier,
            key: fields.key_multiplier,
            mixer_in: fields.ssm_in_multiplier,
            mixer_out: fields.ssm_out_multiplier,
            sections: fields.ssm_multipliers,
            gate,
            down,
        },
    })
}

/// Each row's multiplier of the mixer's input projection, `[P, 1]`, its section's times the input multiplier, as `compute_mup_vector` lays the sections out.
fn input_scales(spec: Mamba2Spec, multipliers: FalconH1Multipliers) -> Vec<f32> {
    let (inner, width) = (spec.inner(), spec.groups * spec.state);
    let sections = [inner, inner, width, width, spec.heads];
    sections
        .iter()
        .zip(multipliers.sections)
        .flat_map(|(&rows, scale)| {
            iter::repeat_n(
                scale * multipliers.mixer_in,
                usize::try_from(rows).unwrap_or_default(),
            )
        })
        .collect()
}

pub(crate) fn rules(config: &FalconH1Config) -> Rules {
    let layers = "model.layers.*";
    let multipliers = config.multipliers;
    let scales = input_scales(config.mixer, multipliers);
    let attention = multipliers.attention_in;
    let projection = |name: &str, to: &str, scale: f32| {
        rename_scaled(
            &format!("{layers}.{name}.weight"),
            &format!("blocks.*.{to}.weight"),
            scale,
        )
    };
    projection("self_attn.q_proj", "attention.query", attention)
        .then(projection(
            "self_attn.k_proj",
            "attention.key",
            attention * multipliers.key,
        ))
        .then(projection("self_attn.v_proj", "attention.value", attention))
        .then(projection(
            "self_attn.o_proj",
            "attention.output",
            multipliers.attention_out,
        ))
        .then(projection(
            "mamba.out_proj",
            "mixer.output",
            multipliers.mixer_out,
        ))
        .then(projection(
            "feed_forward.gate_proj",
            "feed_forward.gate",
            multipliers.gate,
        ))
        .then(projection(
            "feed_forward.down_proj",
            "feed_forward.down",
            multipliers.down,
        ))
        .derive(
            &format!("{layers}.mamba.in_proj.weight"),
            "blocks.*.mixer.input.weight",
            move |weight| {
                let rows = i64::try_from(scales.len())
                    .map_err(|_| SingeError::out_of_range("falcon-h1 input scales"))?;
                let scales =
                    Tensor::from_vec(weight.device(), Shape::create([rows, 1])?, scales.clone())?;
                weight.cast(DataType::F32)?.mul(&scales)
            },
        )
        .rename_each(
            &format!("{layers}.mamba"),
            "blocks.*.mixer",
            &[
                ("conv1d", "convolution"),
                ("dt_bias", "step_bias"),
                ("A_log", "decay_log"),
                ("D", "skip"),
                ("norm", "norm"),
            ],
        )
        .rename(
            &format!("{layers}.pre_ff_layernorm"),
            "blocks.*.feed_forward_norm",
        )
        .rename(
            &format!("{layers}.feed_forward.up_proj"),
            "blocks.*.feed_forward.up",
        )
        .rename("model.final_layernorm", "norm")
        .then(llama_rules("model"))
}
