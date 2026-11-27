//! Reads a `transformers` Qwen3-ASR snapshot: its config, and the rules that import its audio encoder's and Qwen3 decoder's tensors.

use serde::Deserialize;
use singe::nn::Activation;

use crate::{
    assembly::{DecoderConfig, llama_rules, pre_norm_encoder_rules},
    error::{Error, Result},
    format::{
        RuleSet, join,
        transformers::{DecoderDefaults, DecoderFields, activation_field},
    },
    model::audio::qwen3_asr::{Qwen3AsrConfig, Qwen3AsrEncoderConfig},
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    audio_config: AudioFields,
    text_config: TextFields,
    audio_token_id: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct AudioFields {
    num_mel_bins: i64,
    d_model: i64,
    encoder_layers: i64,
    encoder_attention_heads: i64,
    encoder_ffn_dim: i64,
    downsample_hidden_size: i64,
    /// The positions of one chunk after the convolutions.
    max_position_embeddings: i64,
    /// Half the mel frames of a chunk.
    n_window: i64,
    /// The mel frames of an attention window.
    n_window_infer: i64,
    output_dim: i64,
    #[serde(deserialize_with = "activation_field")]
    activation_function: Activation,
    #[serde(default)]
    scale_embedding: bool,
}

/// Qwen3's fields, which the decoder's config holds.
#[derive(Debug, Clone, PartialEq, Deserialize)]
struct TextFields {
    #[serde(flatten)]
    decoder: DecoderFields,
    #[serde(default)]
    use_sliding_window: bool,
}

/// Reads a Qwen3-ASR config.
/// It refuses settings that a checkpoint would load with and then compute something else from: scaled audio embeddings, attention windows that are not whole chunks of mel frames, a `max_position_embeddings` other than a chunk's positions, and sliding-window attention.
pub(crate) fn config(source: &Source) -> Result<Qwen3AsrConfig> {
    let fields: Fields = source.json("config.json")?;
    let audio = &fields.audio_config;
    if audio.scale_embedding {
        return Err(Error::unsupported("scaled audio embeddings"));
    }

    let chunk = 2 * audio.n_window;
    if audio.n_window <= 0 || audio.n_window_infer % chunk != 0 {
        return Err(Error::unsupported(
            "attention windows that are not whole chunks of mel frames",
        ));
    }

    let encoder = Qwen3AsrEncoderConfig {
        mels: audio.num_mel_bins,
        width: audio.d_model,
        layers: audio.encoder_layers,
        heads: audio.encoder_attention_heads,
        inner: audio.encoder_ffn_dim,
        channels: audio.downsample_hidden_size,
        chunk_frames: chunk,
        window_frames: audio.n_window_infer,
        output_width: audio.output_dim,
        activation: audio.activation_function,
    };
    if encoder.chunk_positions() != audio.max_position_embeddings {
        return Err(Error::unsupported(
            "a Qwen3-ASR `max_position_embeddings` other than a chunk's positions",
        ));
    }

    let text = &fields.text_config;
    text.decoder.refuse_sliding_layers()?;
    if text.use_sliding_window {
        return Err(Error::unsupported("a sliding window"));
    }

    // Qwen3-ASR ties its head to the token embeddings unless the config says otherwise.
    let decoder = DecoderConfig {
        head_norms: true,
        ..text.decoder.config(DecoderDefaults {
            tied: true,
            head_dim: Some(128),
        })?
    };
    Ok(Qwen3AsrConfig {
        encoder,
        decoder,
        audio_token: fields.audio_token_id,
    })
}

pub(crate) fn encoder_rules() -> RuleSet {
    let tower = "model.audio_tower";
    RuleSet::new()
        .rename_each(
            tower,
            "encoder",
            &[
                ("conv2d1", "convolutions.0"),
                ("conv2d2", "convolutions.1"),
                ("conv2d3", "convolutions.2"),
                ("conv_out", "convolutions.output"),
                ("ln_post", "norm"),
            ],
        )
        .then(pre_norm_encoder_rules(
            &join(tower, "layers"),
            "encoder.blocks",
        ))
        .rename_each(
            "model.multi_modal_projector",
            "encoder.projector",
            &[("linear_1", "up"), ("linear_2", "down")],
        )
}

pub(crate) fn decoder_rules() -> RuleSet {
    llama_rules("model.language_model")
}
