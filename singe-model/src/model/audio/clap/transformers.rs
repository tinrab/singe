//! Reads a `transformers` CLAP config, its HTSAT audio tower's, its `RoBERTa` text tower's, and its feature extractor's, and imports its checkpoint.

use serde::Deserialize;
use singe::{Error as SingeError, nn::Activation};

use crate::{
    assembly::{SwinConfig, post_norm_encoder_rules, swin_rules},
    error::{Error, Result},
    format::{
        Rules, join,
        transformers::{EncoderFields, activation_field, default_true},
    },
    model::audio::clap::ClapConfig,
    processor::fbank::SpectrogramOptions,
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Audio {
    depths: Vec<i64>,
    num_attention_heads: Vec<i64>,
    window_size: i64,
    patch_embeds_hidden_size: i64,
    spec_size: i64,
    num_mel_bins: i64,
    patch_size: i64,
    patch_stride: Vec<i64>,
    enable_fusion: bool,
    #[serde(default = "default_true")]
    enable_patch_layer_norm: bool,
    mlp_ratio: f64,
    #[serde(deserialize_with = "activation_field")]
    hidden_act: Activation,
    layer_norm_eps: f32,
}

const fn relu() -> Activation {
    Activation::Relu
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Text {
    #[serde(flatten)]
    encoder: EncoderFields,
    pad_token_id: i64,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    audio_config: Audio,
    text_config: Text,
    projection_dim: i64,
    /// The activation of both towers' projections, which `ClapConfig` gives them whatever their own configs say.
    #[serde(default = "relu", deserialize_with = "activation_field")]
    projection_hidden_act: Activation,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Extractor {
    sampling_rate: i64,
    max_length_s: usize,
    fft_window_size: usize,
    hop_length: usize,
    feature_size: usize,
    frequency_min: f64,
    frequency_max: f64,
    truncation: String,
    padding: String,
}

pub(crate) fn config(source: &Source) -> Result<ClapConfig> {
    let fields: Fields = source.json("config.json")?;
    let extractor: Extractor = source.json("preprocessor_config.json")?;
    let audio = fields.audio_config;
    if audio.enable_fusion {
        return Err(Error::unsupported("fused CLAP audio (`enable_fusion`)"));
    }

    if !audio.enable_patch_layer_norm {
        return Err(Error::unsupported(
            "CLAP patches without a norm (`enable_patch_layer_norm`)",
        ));
    }

    if audio
        .patch_stride
        .iter()
        .any(|&stride| stride != audio.patch_size)
    {
        return Err(Error::unsupported(
            "a CLAP `patch_stride` other than its `patch_size`",
        ));
    }

    if extractor.truncation != "rand_trunc" {
        return Err(Error::unsupported(format!(
            "CLAP truncation `{}`",
            extractor.truncation
        )));
    }

    if extractor.padding != "repeatpad" {
        return Err(Error::unsupported(format!(
            "CLAP padding `{}`",
            extractor.padding
        )));
    }

    if audio.num_mel_bins <= 0 {
        return Err(Error::unsupported("a CLAP `num_mel_bins` below one"));
    }

    let rate = usize::try_from(extractor.sampling_rate)
        .map_err(|_| SingeError::out_of_range("sampling_rate"))?;
    Ok(ClapConfig {
        audio: SwinConfig {
            width: audio.patch_embeds_hidden_size,
            stages: audio
                .depths
                .into_iter()
                .zip(audio.num_attention_heads)
                .collect(),
            window: audio.window_size,
            patch_size: audio.patch_size,
            channels: 1,
            mlp_ratio: audio.mlp_ratio,
            activation: audio.hidden_act,
            eps: audio.layer_norm_eps,
        },
        spec_size: audio.spec_size,
        text: fields
            .text_config
            .encoder
            .config(fields.text_config.pad_token_id + 1)?,
        projection: fields.projection_dim,
        projection_activation: fields.projection_hidden_act,
        features: SpectrogramOptions {
            sample_rate: extractor.sampling_rate,
            samples: extractor.max_length_s * rate,
            fft: extractor.fft_window_size,
            hop: extractor.hop_length,
            mels: usize::try_from(audio.num_mel_bins)
                .map_err(|_| SingeError::out_of_range("num_mel_bins"))?,
            low: extractor.frequency_min,
            high: extractor.frequency_max,
        },
    })
}

pub(crate) fn rules() -> Rules {
    let encoder = "audio_model.audio_encoder";
    let projection = [("linear1", "up"), ("linear2", "down")];
    Rules::new()
        // Counts training updates, which evaluation never reads.
        .skip(&join(encoder, "batch_norm.num_batches_tracked"))
        .rename(&join(encoder, "batch_norm"), "audio.norm")
        .rename(&join(encoder, "norm"), "audio.final_norm")
        .then(swin_rules(
            encoder,
            &join(encoder, "patch_embed.proj"),
            &join(encoder, "patch_embed.norm"),
            "audio",
        ))
        .rename_each("audio_projection", "audio_projection", &projection)
        .rename("text_model.pooler.dense", "text.pooler")
        .then(post_norm_encoder_rules("text_model", "text"))
        .rename_each("text_projection", "text_projection", &projection)
        .rename("logit_scale_a", "logit_scale")
        // The text side's own scale, which `logits_per_audio` leaves unread.
        .skip("logit_scale_t")
}
