//! Reads a `transformers` Parakeet snapshot, CTC, RNN-T, or TDT: its config, its feature extractor's settings, and the rules that import its encoder, prediction and joint networks, and CTC head.

use serde::Deserialize;
use serde_json::Value;
use singe::nn::{Activation, LocalAttention, RecurrentCell, RecurrentConfig};

use crate::{
    assembly::{
        ConformerConfig, ConformerEncoderConfig, ConformerFeedForward, PYTORCH_NORM_EPS,
        SubsamplingConfig, SubsamplingKind, TransducerConfig, TransducerDecoding, conformer_rules,
        transducer_rules,
    },
    error::{Error, Result},
    format::{
        RuleSet,
        transformers::{activation, activation_field},
    },
    model::audio::parakeet::{ParakeetConfig, ParakeetFeatures, ParakeetHead},
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    model_type: String,
    encoder_config: EncoderFields,
    vocab_size: i64,
    #[serde(default = "decoder_hidden_size")]
    decoder_hidden_size: i64,
    #[serde(default = "decoder_layers")]
    num_decoder_layers: i64,
    #[serde(default = "joint_activation")]
    hidden_act: String,
    #[serde(default = "max_symbols_per_step")]
    max_symbols_per_step: i64,
    #[serde(default)]
    blank_token_id: Option<i64>,
    #[serde(default)]
    durations: Option<Vec<i64>>,
}

const fn decoder_hidden_size() -> i64 {
    640
}

const fn decoder_layers() -> i64 {
    2
}

fn joint_activation() -> String {
    "relu".to_owned()
}

const fn max_symbols_per_step() -> i64 {
    10
}

/// The encoder's fields, with `ParakeetEncoderConfig`'s defaults.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
struct EncoderFields {
    hidden_size: i64,
    num_hidden_layers: i64,
    num_attention_heads: i64,
    num_key_value_heads: Option<i64>,
    intermediate_size: i64,
    #[serde(deserialize_with = "activation_field")]
    hidden_act: Activation,
    attention_bias: bool,
    convolution_bias: bool,
    conv_kernel_size: i64,
    subsampling_factor: i64,
    subsampling_conv_channels: i64,
    num_mel_bins: i64,
    subsampling_conv_kernel_size: i64,
    subsampling_conv_stride: i64,
    scale_input: bool,
    /// `NeMo`'s rows on each side that attention reads, `[-1, -1]` for all of them; `transformers` ignores it.
    att_context_size: Option<[i64; 2]>,
}

impl Default for EncoderFields {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 8,
            num_key_value_heads: None,
            intermediate_size: 4096,
            hidden_act: Activation::Silu,
            attention_bias: true,
            convolution_bias: true,
            conv_kernel_size: 9,
            subsampling_factor: 8,
            subsampling_conv_channels: 256,
            num_mel_bins: 80,
            subsampling_conv_kernel_size: 3,
            subsampling_conv_stride: 2,
            scale_input: true,
            att_context_size: None,
        }
    }
}

/// The feature extractor's fields in `processor_config.json`, or `preprocessor_config.json` in older snapshots.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
struct FeatureFields {
    feature_size: i64,
    sampling_rate: i64,
    hop_length: i64,
    n_fft: i64,
    win_length: i64,
    preemphasis: Option<f64>,
}

impl Default for FeatureFields {
    fn default() -> Self {
        Self {
            feature_size: 80,
            sampling_rate: 16_000,
            hop_length: 160,
            n_fft: 512,
            win_length: 400,
            preemphasis: Some(0.97),
        }
    }
}

fn features(source: &Source) -> Result<FeatureFields> {
    if let Some(processor) = source.optional_json::<Value>("processor_config.json")?
        && let Some(extractor) = processor.get("feature_extractor")
    {
        return serde_json::from_value(extractor.clone())
            .map_err(Error::json_value("read Parakeet's feature extractor"));
    }

    Ok(source
        .optional_json::<FeatureFields>("preprocessor_config.json")?
        .unwrap_or_default())
}

/// Returns the token `generate` stops at, the first `eos_token_id` in `generation_config.json`, or `None` without one.
fn end_token(source: &Source) -> Result<Option<i64>> {
    let Some(generation) = source.optional_json::<Value>("generation_config.json")? else {
        return Ok(None);
    };
    Ok(match &generation["eos_token_id"] {
        Value::Number(number) => number.as_i64(),
        Value::Array(tokens) => tokens.first().and_then(Value::as_i64),
        _ => None,
    })
}

/// Reads a Parakeet config and its feature extractor's settings.
/// It refuses settings that a checkpoint would load with and then compute something else from: fewer key heads than query heads, a subsampling stride other than two, a feature extractor of other mels than the encoder reads, and an attention context other than the same rows on each side.
pub(crate) fn config(source: &Source) -> Result<ParakeetConfig> {
    let fields: Fields = source.json("config.json")?;
    let encoder = &fields.encoder_config;
    if encoder
        .num_key_value_heads
        .is_some_and(|heads| heads != encoder.num_attention_heads)
    {
        return Err(Error::unsupported(
            "Parakeet attention with fewer key heads than query heads",
        ));
    }

    if encoder.subsampling_conv_stride != 2 {
        return Err(Error::unsupported(
            "a Parakeet subsampling of a stride other than two",
        ));
    }

    let extractor = features(source)?;
    if extractor.feature_size != encoder.num_mel_bins {
        return Err(Error::unsupported(
            "a Parakeet feature extractor of other mels than its encoder reads",
        ));
    }

    // Above 8,192 rows, about 11 minutes, each row attends to the 128 rows on each side, as llama.cpp's Parakeet does after NeMo's `change_attention_model("rel_pos_local_attn", [128, 128])`.
    // `transformers` attends to every row at any length, so a clip that long differs from it.
    // A config's `att_context_size` sets the window at every length instead, or full attention at every length with `[-1, -1]`.
    let local = match encoder.att_context_size {
        None => Some(LocalAttention {
            window: 128,
            above: 8192,
        }),
        Some([-1, -1]) => None,
        Some([left, right]) if left == right && left > 0 => Some(LocalAttention {
            window: left,
            above: 0,
        }),
        Some(_) => {
            return Err(Error::unsupported(
                "a Parakeet attention context other than the same rows on each side",
            ));
        }
    };
    let block = ConformerConfig {
        width: encoder.hidden_size,
        heads: encoder.num_attention_heads,
        inner: encoder.intermediate_size,
        kernel: encoder.conv_kernel_size,
        activation: encoder.hidden_act,
        projection_biases: encoder.attention_bias,
        convolution_biases: encoder.convolution_bias,
        eps: PYTORCH_NORM_EPS,
        local,
        pre_norm: true,
        feed_forward: ConformerFeedForward::Linear,
    };
    let encoder_config = ConformerEncoderConfig {
        subsampling: SubsamplingConfig {
            kind: SubsamplingKind::DepthwiseStriding,
            mels: encoder.num_mel_bins,
            factor: encoder.subsampling_factor,
            channels: encoder.subsampling_conv_channels,
            kernel: encoder.subsampling_conv_kernel_size,
            width: encoder.hidden_size,
        },
        block,
        layers: encoder.num_hidden_layers,
        scale_input: encoder.scale_input,
    };
    let features = ParakeetFeatures {
        sample_rate: extractor.sampling_rate,
        mels: extractor.feature_size,
        window: extractor.win_length,
        fft: extractor.n_fft,
        hop: extractor.hop_length,
        preemphasis: extractor.preemphasis,
    };
    let head = match fields.model_type.as_str() {
        "parakeet_ctc" => ParakeetHead::Ctc {
            vocabulary: fields.vocab_size,
        },
        kind @ ("parakeet_rnnt" | "parakeet_tdt") => {
            let decoding = if kind == "parakeet_tdt" {
                TransducerDecoding::Tdt {
                    durations: fields
                        .durations
                        .clone()
                        .unwrap_or_else(|| vec![0, 1, 2, 3, 4]),
                    max_symbols: None,
                }
            } else {
                TransducerDecoding::Rnnt {
                    max_symbols: fields.max_symbols_per_step,
                }
            };
            let width = fields.decoder_hidden_size;
            ParakeetHead::Transducer(TransducerConfig {
                encoder_width: encoder.hidden_size,
                vocabulary: fields.vocab_size,
                // `transformers`' default, whatever the vocabulary's size.
                blank: fields.blank_token_id.unwrap_or(8192),
                end: end_token(source)?,
                prediction: RecurrentConfig {
                    cell: RecurrentCell::Lstm,
                    input: width,
                    hidden: width,
                    layers: fields.num_decoder_layers,
                    bidirectional: false,
                    biases: true,
                    projection: None,
                },
                prediction_projection: true,
                joint_width: width,
                joint_activation: activation(&fields.hidden_act)?,
                joint_prediction: false,
                decoding,
                steps_per_row: fields.max_symbols_per_step,
            })
        }
        other => return Err(Error::unsupported(format!("Parakeet model type {other}"))),
    };
    Ok(ParakeetConfig {
        encoder: encoder_config,
        features,
        head,
    })
}

pub(crate) fn encoder_rules() -> RuleSet {
    conformer_rules("encoder", "encoder")
}

pub(crate) fn decoder_rules() -> RuleSet {
    transducer_rules("decoder")
}

pub(crate) fn ctc_rules() -> RuleSet {
    conformer_rules("encoder", "encoder").rename("ctc_head", "head")
}
