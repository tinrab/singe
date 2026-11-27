//! Reads a `transformers` `UnivNet` config and its feature extractor's, and imports its checkpoint, whose names Singe keeps.

use serde::Deserialize;

use crate::{
    error::{Error, Result},
    format::Rules,
    model::audio::univnet::UnivNetConfig,
    processor::fbank::AmplitudeMelOptions,
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    num_mel_bins: i64,
    model_in_channels: i64,
    model_hidden_channels: i64,
    kernel_predictor_hidden_channels: i64,
    kernel_predictor_num_blocks: i64,
    kernel_predictor_conv_size: i64,
    resblock_kernel_sizes: Vec<i64>,
    resblock_stride_sizes: Vec<i64>,
    resblock_dilation_sizes: Vec<Vec<i64>>,
    leaky_relu_slope: f64,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct ExtractorFields {
    sampling_rate: i64,
    filter_length: usize,
    win_length: usize,
    hop_length: usize,
    num_mel_bins: usize,
    fmin: f64,
    fmax: Option<f64>,
    mel_floor: f64,
    compression_clip_val: f64,
    compression_factor: f64,
    center: bool,
    do_normalize: bool,
    win_function: String,
}

pub(crate) fn config(source: &Source) -> Result<UnivNetConfig> {
    let fields: Fields = source.json("config.json")?;
    let extractor: ExtractorFields = source.json("preprocessor_config.json")?;
    if extractor.center
        || extractor.do_normalize
        || extractor.win_length != extractor.filter_length
        || extractor.win_function != "hann_window"
        || (extractor.compression_factor - 1.0).abs() > f64::EPSILON
        || !extractor.filter_length.is_power_of_two()
    {
        return Err(Error::unsupported(
            "UnivNet features other than uncentered, unnormalized Hann frames of a power-of-two transform",
        ));
    }
    Ok(UnivNetConfig {
        mels: fields.num_mel_bins,
        noise_channels: fields.model_in_channels,
        hidden: fields.model_hidden_channels,
        predictor_hidden: fields.kernel_predictor_hidden_channels,
        predictor_blocks: fields.kernel_predictor_num_blocks,
        predictor_kernel: fields.kernel_predictor_conv_size,
        kernels: fields.resblock_kernel_sizes,
        strides: fields.resblock_stride_sizes,
        dilations: fields.resblock_dilation_sizes,
        leaky_slope: fields.leaky_relu_slope,
        features: AmplitudeMelOptions {
            sample_rate: extractor.sampling_rate,
            fft: extractor.filter_length,
            hop: extractor.hop_length,
            mels: extractor.num_mel_bins,
            low: extractor.fmin,
            high: extractor
                .fmax
                .unwrap_or(extractor.sampling_rate as f64 / 2.0),
            floor: extractor.mel_floor,
            clip: extractor.compression_clip_val,
        },
    })
}

pub(crate) fn rules() -> Rules {
    Rules::new()
        .rename("conv_pre", "conv_pre")
        .rename("resblocks", "resblocks")
        .rename("conv_post", "conv_post")
}
