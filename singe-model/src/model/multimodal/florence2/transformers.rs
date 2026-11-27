//! Reads a `transformers` Florence-2 config, which nests the vision tower's and the language model's, and imports its checkpoint.

use serde::Deserialize;
use singe::nn::Activation;

use crate::{
    assembly::{BartStackConfig, PYTORCH_NORM_EPS, bart_stack_rules, embedding_scale},
    error::{Error, Result},
    format::{RuleSet, transformers::activation_field},
    model::multimodal::florence2::{DavitStage, Florence2Config},
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct VisionFields {
    in_channels: i64,
    depths: Vec<i64>,
    embed_dim: Vec<i64>,
    num_heads: Vec<i64>,
    num_groups: Vec<i64>,
    patch_size: Vec<i64>,
    patch_stride: Vec<i64>,
    patch_padding: Vec<i64>,
    patch_prenorm: Vec<bool>,
    window_size: i64,
    mlp_ratio: f64,
    #[serde(deserialize_with = "activation_field")]
    activation_function: Activation,
    max_position_embeddings: i64,
    projection_dim: i64,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct TextFields {
    vocab_size: i64,
    d_model: i64,
    encoder_layers: i64,
    decoder_layers: i64,
    encoder_attention_heads: i64,
    decoder_attention_heads: i64,
    encoder_ffn_dim: i64,
    decoder_ffn_dim: i64,
    #[serde(deserialize_with = "activation_field")]
    activation_function: Activation,
    max_position_embeddings: i64,
    #[serde(default)]
    scale_embedding: bool,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    vision_config: VisionFields,
    text_config: TextFields,
    image_token_id: i64,
}

pub(crate) fn config(source: &Source) -> Result<Florence2Config> {
    let Fields {
        vision_config: vision,
        text_config: text,
        image_token_id,
    } = source.json("config.json")?;
    let count = vision.depths.len();
    if [
        vision.embed_dim.len(),
        vision.num_heads.len(),
        vision.num_groups.len(),
        vision.patch_size.len(),
        vision.patch_stride.len(),
        vision.patch_padding.len(),
        vision.patch_prenorm.len(),
    ]
    .iter()
    .any(|&length| length != count)
    {
        return Err(Error::unsupported(
            "a DaViT whose stages' settings differ in count",
        ));
    }

    let stages = (0..count)
        .map(|index| DavitStage {
            width: vision.embed_dim[index],
            depth: vision.depths[index],
            heads: vision.num_heads[index],
            groups: vision.num_groups[index],
            kernel: vision.patch_size[index],
            stride: vision.patch_stride[index],
            padding: vision.patch_padding[index],
            norm_first: vision.patch_prenorm[index],
        })
        .collect();
    let stack = |layers, heads, inner| BartStackConfig {
        vocab_size: text.vocab_size,
        width: text.d_model,
        layers,
        heads,
        inner,
        activation: text.activation_function,
        max_positions: text.max_position_embeddings,
        embedding_scale: embedding_scale(text.scale_embedding, text.d_model),
        embedding_norm: true,
        pre_norm: false,
        cross_width: text.d_model,
        eps: PYTORCH_NORM_EPS,
    };
    Ok(Florence2Config {
        channels: vision.in_channels,
        stages,
        window: vision.window_size,
        mlp_ratio: vision.mlp_ratio,
        activation: vision.activation_function,
        max_positions: vision.max_position_embeddings,
        projection: vision.projection_dim,
        encoder: stack(
            text.encoder_layers,
            text.encoder_attention_heads,
            text.encoder_ffn_dim,
        ),
        decoder: stack(
            text.decoder_layers,
            text.decoder_attention_heads,
            text.decoder_ffn_dim,
        ),
        image_token: image_token_id,
    })
}

fn half_block_rules(from: &str, to: &str, attention: &str) -> RuleSet {
    let attention_from = format!("{from}.{attention}");
    RuleSet::new()
        .split(
            &format!("{attention_from}.qkv"),
            0,
            &[
                &format!("{to}.attention.query"),
                &format!("{to}.attention.key"),
                &format!("{to}.attention.value"),
            ],
        )
        .rename(
            &format!("{attention_from}.proj"),
            &format!("{to}.attention.output"),
        )
        .rename_each(
            from,
            to,
            &[
                ("conv1", "first_conv"),
                ("norm1", "attention_norm"),
                ("conv2", "second_conv"),
                ("norm2", "feed_forward_norm"),
                ("ffn.fc1", "feed_forward.up"),
                ("ffn.fc2", "feed_forward.down"),
            ],
        )
}

pub(crate) fn rules() -> RuleSet {
    let (vision, projector) = ("model.vision_tower", "model.multi_modal_projector");
    RuleSet::new()
        .rename(&format!("{vision}.convs.*.conv"), "vision.stages.*.conv")
        .rename(&format!("{vision}.convs.*.norm"), "vision.stages.*.norm")
        .then(half_block_rules(
            &format!("{vision}.blocks.*.*.spatial_block"),
            "vision.stages.*.blocks.*.spatial",
            "window_attn",
        ))
        .then(half_block_rules(
            &format!("{vision}.blocks.*.*.channel_block"),
            "vision.stages.*.blocks.*.channel",
            "channel_attn",
        ))
        .rename(
            &format!("{projector}.image_position_embed.row_embeddings.weight"),
            "projector.rows",
        )
        .rename(
            &format!("{projector}.image_position_embed.column_embeddings.weight"),
            "projector.columns",
        )
        // The first frame's row of the cosine temporal table, which one image reads.
        .derive(
            &format!("{projector}.visual_temporal_embed.pos_idx_to_embed"),
            "projector.temporal",
            |table| table.narrow(0, 0, 1)?.contiguous(),
        )
        .rename(
            &format!("{projector}.image_projection"),
            "projector.projection",
        )
        .rename(&format!("{projector}.image_proj_norm"), "projector.norm")
        .rename("model.language_model.shared", "tokens")
        // Copies of the shared embedding, which the output reads too.
        .skip("model.language_model.encoder.embed_tokens")
        .skip("model.language_model.decoder.embed_tokens")
        .skip("lm_head")
        .then(bart_stack_rules("model.language_model.encoder", "encoder"))
        .then(bart_stack_rules("model.language_model.decoder", "decoder"))
}
