//! Reads a `transformers` FastSpeech2Conformer-with-HiFi-GAN config, which nests the acoustic model's and the vocoder's, and imports its checkpoint.

use serde::Deserialize;
use singe::nn::Activation;

use crate::{
    assembly::{
        ConformerConfig, ConformerFeedForward, HifiGanConfig, PYTORCH_NORM_EPS, hifigan_rules,
    },
    error::{Error, Result},
    format::{RuleSet, join},
    model::audio::fastspeech2_conformer::{ConvStack, FastSpeech2Config},
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct StackFields {
    kernel_size: i64,
    layers: i64,
    linear_units: i64,
    normalize_before: bool,
    num_attention_heads: i64,
    concat_after: bool,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct ModelFields {
    vocab_size: i64,
    hidden_size: i64,
    num_mel_bins: i64,
    encoder_config: StackFields,
    decoder_config: StackFields,
    positionwise_conv_kernel_size: i64,
    use_macaron_style_in_conformer: bool,
    use_cnn_in_conformer: bool,
    duration_predictor_layers: i64,
    duration_predictor_channels: i64,
    duration_predictor_kernel_size: i64,
    pitch_predictor_layers: i64,
    pitch_predictor_channels: i64,
    pitch_predictor_kernel_size: i64,
    pitch_embed_kernel_size: i64,
    energy_predictor_layers: i64,
    energy_predictor_channels: i64,
    energy_predictor_kernel_size: i64,
    energy_embed_kernel_size: i64,
    speech_decoder_postnet_layers: i64,
    speech_decoder_postnet_units: i64,
    speech_decoder_postnet_kernel: i64,
    reduction_factor: i64,
    speaking_speed: f32,
    num_speakers: Option<i64>,
    num_languages: Option<i64>,
    speaker_embed_dim: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct VocoderFields {
    model_in_dim: i64,
    upsample_initial_channel: i64,
    upsample_rates: Vec<i64>,
    upsample_kernel_sizes: Vec<i64>,
    resblock_kernel_sizes: Vec<i64>,
    resblock_dilation_sizes: Vec<Vec<i64>>,
    leaky_relu_slope: f64,
    normalize_before: bool,
    sampling_rate: i64,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    model_config: ModelFields,
    vocoder_config: VocoderFields,
}

pub(crate) fn config(source: &Source) -> Result<FastSpeech2Config> {
    let Fields {
        model_config: model,
        vocoder_config: vocoder,
    } = source.json("config.json")?;
    let single = |count: Option<i64>| count.is_none_or(|count| count <= 1);
    if !model.use_macaron_style_in_conformer
        || !model.use_cnn_in_conformer
        || model.encoder_config.concat_after
        || model.decoder_config.concat_after
        || model.reduction_factor != 1
        || (model.speaking_speed - 1.0).abs() > f32::EPSILON
        || !single(model.num_speakers)
        || !single(model.num_languages)
        || model.speaker_embed_dim.is_some()
        || vocoder.normalize_before
    {
        return Err(Error::unsupported(
            "a FastSpeech2Conformer other than one speaker's macaron Conformer at its own speed",
        ));
    }

    let stack = |fields: &StackFields| ConformerConfig {
        width: model.hidden_size,
        heads: fields.num_attention_heads,
        inner: fields.linear_units,
        kernel: fields.kernel_size,
        activation: Activation::Silu,
        projection_biases: true,
        convolution_biases: true,
        eps: PYTORCH_NORM_EPS,
        local: None,
        pre_norm: fields.normalize_before,
        feed_forward: ConformerFeedForward::Convolution {
            kernel: model.positionwise_conv_kernel_size,
            activation: Activation::Relu,
        },
    };
    Ok(FastSpeech2Config {
        vocab_size: model.vocab_size,
        width: model.hidden_size,
        mels: model.num_mel_bins,
        encoder: stack(&model.encoder_config),
        encoder_layers: model.encoder_config.layers,
        decoder: stack(&model.decoder_config),
        decoder_layers: model.decoder_config.layers,
        duration: ConvStack {
            layers: model.duration_predictor_layers,
            channels: model.duration_predictor_channels,
            kernel: model.duration_predictor_kernel_size,
        },
        pitch: ConvStack {
            layers: model.pitch_predictor_layers,
            channels: model.pitch_predictor_channels,
            kernel: model.pitch_predictor_kernel_size,
        },
        pitch_embed_kernel: model.pitch_embed_kernel_size,
        energy: ConvStack {
            layers: model.energy_predictor_layers,
            channels: model.energy_predictor_channels,
            kernel: model.energy_predictor_kernel_size,
        },
        energy_embed_kernel: model.energy_embed_kernel_size,
        postnet: ConvStack {
            layers: model.speech_decoder_postnet_layers,
            channels: model.speech_decoder_postnet_units,
            kernel: model.speech_decoder_postnet_kernel,
        },
        vocoder: HifiGanConfig {
            input: vocoder.model_in_dim,
            initial: vocoder.upsample_initial_channel,
            upsample_rates: vocoder.upsample_rates,
            upsample_kernels: vocoder.upsample_kernel_sizes,
            resblock_kernels: vocoder.resblock_kernel_sizes,
            resblock_dilations: vocoder.resblock_dilation_sizes,
            leaky_slope: vocoder.leaky_relu_slope,
            post_bias: true,
        },
        sample_rate: vocoder.sampling_rate,
    })
}

fn stack_rules(from: &str, to: &str) -> RuleSet {
    let (from, to) = (join(from, "conformer_layers.*"), join(to, "blocks.*"));
    RuleSet::new()
        // A batch norm's count of the batches it saw in training, which evaluation never reads.
        .skip(&join(&from, "conv_module.norm.num_batches_tracked"))
        .rename_each(
            &from,
            &to,
            &[
                ("ff_macaron_layer_norm", "feed_forward_in_norm"),
                ("feed_forward_macaron.conv1", "feed_forward_in.up"),
                ("feed_forward_macaron.conv2", "feed_forward_in.down"),
                ("self_attn_layer_norm", "attention_norm"),
                ("self_attn.linear_q", "attention.query"),
                ("self_attn.linear_k", "attention.key"),
                ("self_attn.linear_v", "attention.value"),
                ("self_attn.linear_out", "attention.output"),
                ("self_attn.linear_pos", "attention.positions"),
                ("self_attn.pos_bias_u", "attention.content_bias"),
                ("self_attn.pos_bias_v", "attention.position_bias"),
                ("conv_layer_norm", "convolution_norm"),
                ("conv_module.pointwise_conv1", "convolution.pointwise_in"),
                ("conv_module.depthwise_conv", "convolution.depthwise"),
                ("conv_module.norm", "convolution.norm"),
                ("conv_module.pointwise_conv2", "convolution.pointwise_out"),
                ("ff_layer_norm", "feed_forward_out_norm"),
                ("feed_forward.conv1", "feed_forward_out.up"),
                ("feed_forward.conv2", "feed_forward_out.down"),
                ("final_layer_norm", "norm"),
            ],
        )
}

fn predictor_rules(from: &str, to: &str) -> RuleSet {
    RuleSet::new()
        .rename_each(
            &format!("model.{from}.conv_layers.*"),
            &format!("{to}.layers.*"),
            &[("conv", "conv"), ("layer_norm", "norm")],
        )
        .rename(&format!("model.{from}.linear"), &format!("{to}.output"))
}

pub(crate) fn rules() -> RuleSet {
    RuleSet::new()
        .rename("model.encoder.embed", "tokens")
        .then(stack_rules("model.encoder", "encoder"))
        .then(stack_rules("model.decoder", "decoder"))
        .then(predictor_rules("duration_predictor", "duration"))
        .then(predictor_rules("pitch_predictor", "pitch"))
        .then(predictor_rules("energy_predictor", "energy"))
        .rename("model.pitch_embed.conv", "pitch_embed")
        .rename("model.energy_embed.conv", "energy_embed")
        .rename("model.speech_decoder_postnet.feat_out", "mels")
        .skip("model.speech_decoder_postnet.layers.*.batch_norm.num_batches_tracked")
        .rename_each(
            "model.speech_decoder_postnet.layers.*",
            "postnet.layers.*",
            &[("conv", "conv"), ("batch_norm", "norm")],
        )
        .then(hifigan_rules("vocoder", "vocoder"))
        // The vocoder's input statistics, which it reads only with `normalize_before`.
        .skip("vocoder.mean")
        .skip("vocoder.scale")
}
