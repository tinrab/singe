//! Reads a `transformers` vision encoder-decoder config, which nests its encoder's and decoder's, and imports its checkpoint, which stores them under `encoder.` and `decoder.`.

use serde::Deserialize;
use serde_json::Value;
use singe::nn::Activation;

use crate::{
    assembly::{
        BartStackConfig, PYTORCH_NORM_EPS, SwinConfig, bart_stack_rules, embedding_scale,
        swin_rules,
    },
    error::{Error, Result},
    format::{
        Rules,
        transformers::{activation_field, default_true, parse},
    },
    model::{
        multimodal::vision_encoder_decoder::{ImageEncoderConfig, VisionEncoderDecoderConfig},
        vision::vit::transformers::{VitFields, encoder_rules},
    },
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Deit {
    #[serde(flatten)]
    vit: VitFields,
    image_size: i64,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "the fields are `config.json`'s"
)]
struct TrOcr {
    vocab_size: i64,
    d_model: i64,
    decoder_layers: i64,
    decoder_attention_heads: i64,
    decoder_ffn_dim: i64,
    #[serde(deserialize_with = "activation_field")]
    activation_function: Activation,
    max_position_embeddings: i64,
    scale_embedding: bool,
    layernorm_embedding: bool,
    use_learned_position_embeddings: bool,
    cross_attention_hidden_size: Option<i64>,
    #[serde(default = "default_true")]
    tie_word_embeddings: bool,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct DonutSwin {
    embed_dim: i64,
    depths: Vec<i64>,
    num_heads: Vec<i64>,
    window_size: i64,
    patch_size: i64,
    num_channels: i64,
    mlp_ratio: f64,
    #[serde(deserialize_with = "activation_field")]
    hidden_act: Activation,
    layer_norm_eps: f32,
    #[serde(default)]
    use_absolute_embeddings: bool,
    #[serde(default = "default_true")]
    qkv_bias: bool,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct MBart {
    vocab_size: i64,
    d_model: i64,
    decoder_layers: i64,
    decoder_attention_heads: i64,
    decoder_ffn_dim: i64,
    #[serde(deserialize_with = "activation_field")]
    activation_function: Activation,
    max_position_embeddings: i64,
    scale_embedding: bool,
    #[serde(default = "default_true")]
    tie_word_embeddings: bool,
}

fn encoder(value: &Value) -> Result<ImageEncoderConfig> {
    match value["model_type"].as_str() {
        Some("donut-swin") => {
            let fields: DonutSwin = parse(value.clone())?;
            if fields.use_absolute_embeddings {
                return Err(Error::unsupported(
                    "a Donut Swin encoder with absolute position embeddings",
                ));
            }

            if !fields.qkv_bias {
                return Err(Error::unsupported(
                    "a Donut Swin encoder without query, key, and value biases",
                ));
            }

            if fields.depths.len() != fields.num_heads.len() {
                return Err(Error::unsupported(
                    "a Donut Swin encoder with other than one head count per stage",
                ));
            }
            Ok(ImageEncoderConfig::Swin(SwinConfig {
                width: fields.embed_dim,
                stages: fields.depths.into_iter().zip(fields.num_heads).collect(),
                window: fields.window_size,
                patch_size: fields.patch_size,
                channels: fields.num_channels,
                mlp_ratio: fields.mlp_ratio,
                activation: fields.hidden_act,
                eps: fields.layer_norm_eps,
            }))
        }
        Some("deit") => {
            let fields: Deit = parse(value.clone())?;
            Ok(ImageEncoderConfig::Deit(
                fields.vit.tower(fields.image_size)?,
            ))
        }
        other => Err(Error::unsupported(format!(
            "an image encoder of type {other:?}"
        ))),
    }
}

fn decoder(value: &Value, encoder_width: i64) -> Result<(BartStackConfig, bool)> {
    match value["model_type"].as_str() {
        // MBart's decoder norms before each part and after its last block.
        Some("mbart") => {
            let fields: MBart = parse(value.clone())?;
            Ok((
                BartStackConfig {
                    vocab_size: fields.vocab_size,
                    width: fields.d_model,
                    layers: fields.decoder_layers,
                    heads: fields.decoder_attention_heads,
                    inner: fields.decoder_ffn_dim,
                    activation: fields.activation_function,
                    max_positions: fields.max_position_embeddings,
                    embedding_scale: embedding_scale(fields.scale_embedding, fields.d_model),
                    embedding_norm: true,
                    pre_norm: true,
                    cross_width: encoder_width,
                    eps: PYTORCH_NORM_EPS,
                },
                fields.tie_word_embeddings,
            ))
        }
        // TrOCR's decoder norms after each residual and has no norm after its last block.
        Some("trocr") => {
            let fields: TrOcr = parse(value.clone())?;
            if !fields.use_learned_position_embeddings {
                return Err(Error::unsupported("TrOCR with sinusoidal positions"));
            }
            Ok((
                BartStackConfig {
                    vocab_size: fields.vocab_size,
                    width: fields.d_model,
                    layers: fields.decoder_layers,
                    heads: fields.decoder_attention_heads,
                    inner: fields.decoder_ffn_dim,
                    activation: fields.activation_function,
                    max_positions: fields.max_position_embeddings,
                    embedding_scale: embedding_scale(fields.scale_embedding, fields.d_model),
                    embedding_norm: fields.layernorm_embedding,
                    pre_norm: false,
                    cross_width: fields.cross_attention_hidden_size.unwrap_or(encoder_width),
                    eps: PYTORCH_NORM_EPS,
                },
                fields.tie_word_embeddings,
            ))
        }
        other => Err(Error::unsupported(format!(
            "a text decoder of type {other:?}"
        ))),
    }
}

pub(crate) fn config(source: &Source) -> Result<VisionEncoderDecoderConfig> {
    let fields: Value = source.json("config.json")?;
    let encoder = encoder(&fields["encoder"])?;
    let width = match &encoder {
        ImageEncoderConfig::Deit(tower) => tower.encoder.width,
        ImageEncoderConfig::Swin(swin) => swin.out_width(),
    };
    let (decoder, tied) = decoder(&fields["decoder"], width)?;
    Ok(VisionEncoderDecoderConfig {
        encoder,
        decoder,
        tied,
    })
}

pub(crate) fn rules(config: &VisionEncoderDecoderConfig) -> Rules {
    let encoder = match config.encoder {
        ImageEncoderConfig::Deit(_) => encoder_rules("encoder", "encoder").rename(
            "encoder.embeddings.distillation_token",
            "encoder.embeddings.distillation",
        ),
        ImageEncoderConfig::Swin(_) => swin_rules(
            "encoder.encoder",
            "encoder.embeddings.patch_embeddings.projection",
            "encoder.embeddings.norm",
            "encoder",
        ),
    };
    encoder
        .then(bart_stack_rules("decoder.model.decoder", "decoder"))
        .rename("decoder.output_projection", "head")
        .rename("decoder.lm_head", "head")
}
