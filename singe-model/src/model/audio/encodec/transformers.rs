//! Reads a `transformers` `EnCodec` config and imports its checkpoint, folding each convolution's weight norm.

use serde::Deserialize;

use crate::{
    assembly::{PadMode, SeanetConfig, seanet_rules},
    error::{Error, Result},
    format::{RuleSet, join},
    model::audio::encodec::EncodecConfig,
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub(crate) struct Fields {
    sampling_rate: i64,
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
    num_lstm_layers: i64,
    #[serde(default = "conv_shortcut")]
    use_conv_shortcut: bool,
    use_causal_conv: bool,
    pad_mode: String,
    trim_right_ratio: f64,
    norm_type: String,
    codebook_size: i64,
    codebook_dim: Option<i64>,
    target_bandwidths: Vec<f64>,
    normalize: bool,
    chunk_length_s: Option<f64>,
}

const fn conv_shortcut() -> bool {
    true
}

pub(crate) fn config(source: &Source) -> Result<EncodecConfig> {
    fields_config(source.json("config.json")?)
}

/// Reads `EnCodec`'s config from its fields, as a composite model such as `MusicGen` nests them.
pub(crate) fn fields_config(fields: Fields) -> Result<EncodecConfig> {
    if fields.norm_type != "weight_norm" {
        return Err(Error::unsupported(format!(
            "EnCodec's `{}` norm",
            fields.norm_type
        )));
    }

    if fields.normalize || fields.chunk_length_s.is_some() {
        return Err(Error::unsupported(
            "EnCodec with normalized or chunked clips",
        ));
    }

    let pad = match fields.pad_mode.as_str() {
        "reflect" => PadMode::Reflect,
        "constant" => PadMode::Constant,
        other => return Err(Error::unsupported(format!("EnCodec's `{other}` padding"))),
    };
    // The frame rate and bits per codebook give each bandwidth's codebooks, the default, smallest, bandwidth's among them, as `get_num_quantizers_for_bandwidth` counts them.
    let hop = fields.upsampling_ratios.iter().product::<i64>();
    let frame_rate = (fields.sampling_rate as f64 / hop as f64).ceil();
    let bandwidth = fields
        .target_bandwidths
        .first()
        .copied()
        .ok_or_else(|| Error::unsupported("EnCodec without target bandwidths"))?;
    let bits = (fields.codebook_size as f64).log2() * frame_rate;
    let codebooks = ((bandwidth * 1000.0 / bits).floor() as i64).max(1);
    Ok(EncodecConfig {
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
            lstm_layers: fields.num_lstm_layers,
            conv_shortcut: fields.use_conv_shortcut,
            causal: fields.use_causal_conv,
            pad,
            trim_right: fields.trim_right_ratio,
        },
        codebook_size: fields.codebook_size,
        codebook_dim: fields.codebook_dim.unwrap_or(fields.hidden_size),
        codebooks,
        bandwidths: fields.target_bandwidths,
        codebook_bits: bits,
    })
}

/// Imports an `EnCodec` stored under `prefix` to [`EncodecLayers`](crate::model::audio::encodec::EncodecLayers)' names under the same prefix.
pub(crate) fn rules(config: &EncodecConfig, prefix: &str) -> RuleSet {
    let mut rules = seanet_rules(
        &config.seanet,
        [&join(prefix, "encoder"), &join(prefix, "decoder")],
        true,
    );
    for index in 0..config.held_codebooks() {
        rules = rules.rename(
            &join(prefix, &format!("quantizer.layers.{index}.codebook.embed")),
            &join(prefix, &format!("codebooks.{index}")),
        );
    }
    // The codebooks past the largest bandwidth's, and the training statistics.
    rules.skip(&join(prefix, "quantizer"))
}
