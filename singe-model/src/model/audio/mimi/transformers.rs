//! Reads a `transformers` Mimi config and imports its checkpoint, whose codebooks are stored as sums of their assigned vectors and their usage.

use serde::Deserialize;
use singe::{
    DataType,
    nn::{Derivation, TensorSource},
};

use crate::{
    assembly::{PadMode, SeanetConfig, seanet_rules},
    error::{Error, Result},
    format::{RuleSet, join, transformers::activation},
    model::audio::mimi::{CodecTransformerConfig, MimiConfig},
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub(crate) struct Fields {
    sampling_rate: i64,
    frame_rate: f64,
    audio_channels: i64,
    num_filters: i64,
    upsampling_ratios: Vec<i64>,
    num_residual_layers: i64,
    residual_kernel_size: i64,
    dilation_growth_rate: i64,
    compress: i64,
    kernel_size: i64,
    last_kernel_size: i64,
    hidden_size: i64,
    use_conv_shortcut: bool,
    use_causal_conv: bool,
    pad_mode: String,
    trim_right_ratio: f64,
    num_hidden_layers: i64,
    num_attention_heads: i64,
    num_key_value_heads: i64,
    head_dim: i64,
    intermediate_size: i64,
    hidden_act: String,
    norm_eps: f32,
    rope_theta: f64,
    sliding_window: i64,
    upsample_groups: i64,
    codebook_size: i64,
    codebook_dim: i64,
    vector_quantization_hidden_dimension: i64,
    num_semantic_quantizers: i64,
    #[serde(default = "codebooks")]
    num_quantizers: i64,
    normalize: bool,
}

/// `MimiConfig`'s default `num_quantizers`.
const fn codebooks() -> i64 {
    32
}

/// The usage below which a codebook entry's sum is divided by this instead, as `MimiEuclideanCodebook` clamps it.
const USAGE_FLOOR: f32 = 1e-5;

pub(crate) fn config(source: &Source) -> Result<MimiConfig> {
    fields_config(source.json("config.json")?)
}

/// Returns the config of Mimi's `fields`, which another model's config may hold as its codec's.
pub(crate) fn fields_config(fields: Fields) -> Result<MimiConfig> {
    if fields.normalize || fields.audio_channels != 1 {
        return Err(Error::unsupported(
            "Mimi with normalized or multichannel clips",
        ));
    }

    if fields.num_key_value_heads != fields.num_attention_heads
        || fields.vector_quantization_hidden_dimension != fields.codebook_dim
    {
        return Err(Error::unsupported(
            "Mimi with grouped attention or a quantizer width other than its codebooks'",
        ));
    }

    let pad = match fields.pad_mode.as_str() {
        "constant" => PadMode::Constant,
        "reflect" => PadMode::Reflect,
        "replicate" => PadMode::Replicate,
        other => return Err(Error::unsupported(format!("Mimi's `{other}` padding"))),
    };
    // The SEANet's frame rate over the quantizer's.
    let hop = fields.upsampling_ratios.iter().product::<i64>();
    let latent_stride =
        (fields.sampling_rate as f64 / hop as f64 / fields.frame_rate).round() as i64;
    Ok(MimiConfig {
        sample_rate: fields.sampling_rate,
        seanet: SeanetConfig {
            channels: fields.audio_channels,
            filters: fields.num_filters,
            ratios: fields.upsampling_ratios,
            residual_layers: fields.num_residual_layers,
            residual_kernel: fields.residual_kernel_size,
            dilation_growth: fields.dilation_growth_rate,
            compress: fields.compress,
            kernel: fields.kernel_size,
            last_kernel: fields.last_kernel_size,
            hidden: fields.hidden_size,
            lstm_layers: 0,
            conv_shortcut: fields.use_conv_shortcut,
            causal: fields.use_causal_conv,
            pad,
            trim_right: fields.trim_right_ratio,
        },
        transformer: CodecTransformerConfig {
            width: fields.hidden_size,
            layers: fields.num_hidden_layers,
            heads: fields.num_attention_heads,
            head_dim: fields.head_dim,
            inner: fields.intermediate_size,
            activation: activation(&fields.hidden_act)?,
            eps: fields.norm_eps,
            theta: fields.rope_theta,
            window: fields.sliding_window,
        },
        latent_stride,
        upsample_groups: fields.upsample_groups,
        codebook_size: fields.codebook_size,
        codebook_dim: fields.codebook_dim,
        semantic_codebooks: fields.num_semantic_quantizers,
        codebooks: fields.num_quantizers,
    })
}

fn transformer_rules(prefix: &str, part: &str) -> RuleSet {
    let layers = join(prefix, &format!("{part}.layers.*"));
    RuleSet::new().rename_each(
        &layers,
        &layers,
        &[
            ("input_layernorm", "attention_norm"),
            ("self_attn.q_proj", "attention.query"),
            ("self_attn.k_proj", "attention.key"),
            ("self_attn.v_proj", "attention.value"),
            ("self_attn.o_proj", "attention.output"),
            ("self_attn_layer_scale.scale", "attention_scale"),
            ("post_attention_layernorm", "feed_forward_norm"),
            ("mlp.fc1", "feed_forward.up"),
            ("mlp.fc2", "feed_forward.down"),
            ("mlp_layer_scale.scale", "feed_forward_scale"),
        ],
    )
}

fn quantizer_rules(prefix: &str, from: &str, to: &str) -> RuleSet {
    let from = join(prefix, &format!("quantizer.{from}"));
    let to = join(prefix, to);
    let entries = format!("{to}.codebooks.*");
    RuleSet::new()
        .rename(&format!("{from}.input_proj"), &format!("{to}.project_in"))
        .rename(&format!("{from}.output_proj"), &format!("{to}.project_out"))
        .custom(
            &format!("{from}.layers.*.codebook.embed_sum"),
            move |found| {
                let usage = found.name.replace("embed_sum", "cluster_usage");
                Ok(vec![(
                    found.target(&entries),
                    TensorSource::Derived(Derivation::new(
                        vec![found.name.to_owned(), usage],
                        |inputs| {
                            // In `F32`, as `transformers` divides after loading, whatever type the checkpoint stores, such as CSM's `F16`.
                            let sums = inputs[0].cast(DataType::F32)?;
                            let usage = inputs[1].cast(DataType::F32)?;
                            sums.div(&usage.clamp(USAGE_FLOOR, f32::INFINITY)?.unsqueeze(1)?)
                        },
                    )),
                )])
            },
        )
        // Read through the sums' rule.
        .skip(&format!("{from}.layers.*.codebook.cluster_usage"))
        .skip(&format!("{from}.layers.*.codebook.initialized"))
}

/// Imports Mimi stored under `prefix`, empty for Mimi's own checkpoint, to the same names under `prefix`.
pub(crate) fn rules(config: &MimiConfig, prefix: &str) -> RuleSet {
    seanet_rules(
        &config.seanet,
        [&join(prefix, "encoder"), &join(prefix, "decoder")],
        false,
    )
    .then(transformer_rules(prefix, "encoder_transformer"))
    .then(transformer_rules(prefix, "decoder_transformer"))
    .then(quantizer_rules(
        prefix,
        "semantic_residual_vector_quantizer",
        "semantic",
    ))
    .then(quantizer_rules(
        prefix,
        "acoustic_residual_vector_quantizer",
        "acoustic",
    ))
    .rename(&join(prefix, "downsample"), &join(prefix, "downsample"))
    .rename(&join(prefix, "upsample"), &join(prefix, "upsample"))
}
