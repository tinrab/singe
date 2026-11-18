// Reads a `transformers` SAM config, of a vision encoder, a prompt encoder, and a mask decoder, and imports its checkpoint.

use serde::Deserialize;
use singe::nn::Activation;

use super::SamConfig;
use crate::{
    error::{Error, Result},
    format::{Rules, transformers::activation_field},
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Vision {
    hidden_size: i64,
    num_hidden_layers: i64,
    num_attention_heads: i64,
    mlp_dim: i64,
    layer_norm_eps: f32,
    patch_size: i64,
    image_size: i64,
    window_size: i64,
    global_attn_indexes: Vec<i64>,
    output_channels: i64,
    #[serde(deserialize_with = "activation_field")]
    hidden_act: Activation,
    use_abs_pos: bool,
    use_rel_pos: bool,
    qkv_bias: bool,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Decoder {
    hidden_size: i64,
    num_hidden_layers: i64,
    num_attention_heads: i64,
    mlp_dim: i64,
    #[serde(deserialize_with = "activation_field")]
    hidden_act: Activation,
    attention_downsample_rate: i64,
    num_multimask_outputs: i64,
    iou_head_depth: i64,
    iou_head_hidden_dim: i64,
    layer_norm_eps: f32,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    vision_config: Vision,
    mask_decoder_config: Decoder,
}

pub(crate) fn config(source: &Source) -> Result<SamConfig> {
    let fields: Fields = source.json("config.json")?;
    let (vision, decoder) = (fields.vision_config, fields.mask_decoder_config);
    // SAM's own encoder: absolute and relative positions, biased queries, keys, and values, and exact GELU.
    if !vision.use_abs_pos
        || !vision.use_rel_pos
        || !vision.qkv_bias
        || vision.hidden_act != Activation::GeluExact
        || decoder.hidden_size != vision.output_channels
        || (vision.layer_norm_eps - decoder.layer_norm_eps).abs() > f32::EPSILON
        || decoder.iou_head_depth < 2
    {
        return Err(Error::unsupported("this SAM config"));
    }

    Ok(SamConfig {
        width: vision.hidden_size,
        layers: vision.num_hidden_layers,
        heads: vision.num_attention_heads,
        inner: vision.mlp_dim,
        eps: vision.layer_norm_eps,
        patch_size: vision.patch_size,
        image_size: vision.image_size,
        window: vision.window_size,
        global: vision.global_attn_indexes,
        embedding: vision.output_channels,
        decoder_layers: decoder.num_hidden_layers,
        decoder_heads: decoder.num_attention_heads,
        decoder_inner: decoder.mlp_dim,
        decoder_activation: decoder.hidden_act,
        downsample: decoder.attention_downsample_rate,
        masks: decoder.num_multimask_outputs,
        iou_depth: decoder.iou_head_depth,
        iou_hidden: decoder.iou_head_hidden_dim,
    })
}

// A `SamFeedForward`'s layers stored as `proj_in`, `layers.N`, and `proj_out`, numbered in order.
fn feed_forward(rules: Rules, from: &str, to: &str, depth: i64) -> Rules {
    let mut rules = rules
        .rename(&format!("{from}.proj_in"), &format!("{to}.0"))
        .rename(&format!("{from}.proj_out"), &format!("{to}.{}", depth - 1));
    for index in 0..depth - 2 {
        rules = rules.rename(
            &format!("{from}.layers.{index}"),
            &format!("{to}.{}", index + 1),
        );
    }
    rules
}

fn attention(rules: Rules, from: &str, to: &str) -> Rules {
    rules
        .rename(&format!("{from}.q_proj"), &format!("{to}.query"))
        .rename(&format!("{from}.k_proj"), &format!("{to}.key"))
        .rename(&format!("{from}.v_proj"), &format!("{to}.value"))
        .rename(&format!("{from}.out_proj"), &format!("{to}.output"))
}

pub(crate) fn rules(iou_depth: i64) -> Rules {
    let vision = |name: &str| format!("vision_encoder.layers.*.{name}");
    let mut rules = Rules::new()
        .rename("vision_encoder.patch_embed.projection", "patches")
        .rename("vision_encoder.pos_embed", "positions")
        .rename(&vision("layer_norm1"), "blocks.*.attention_norm")
        .rename(&vision("attn.qkv"), "blocks.*.qkv")
        .rename(&vision("attn.proj"), "blocks.*.output")
        .rename(&vision("attn.rel_pos_h"), "blocks.*.relative_height")
        .rename(&vision("attn.rel_pos_w"), "blocks.*.relative_width")
        .rename(&vision("layer_norm2"), "blocks.*.feed_forward_norm")
        .rename(&vision("mlp.lin1"), "blocks.*.up")
        .rename(&vision("mlp.lin2"), "blocks.*.down")
        .rename("vision_encoder.neck.conv1", "neck.0")
        .rename("vision_encoder.neck.conv2", "neck.1")
        .rename("vision_encoder.neck.layer_norm1", "neck_norms.0")
        .rename("vision_encoder.neck.layer_norm2", "neck_norms.1")
        .rename("shared_image_embedding.positional_embedding", "gaussian")
        .rename("prompt_encoder.point_embed.*.weight", "point_embeddings.*")
        .rename("prompt_encoder.not_a_point_embed.weight", "not_a_point")
        .rename("prompt_encoder.no_mask_embed.weight", "no_mask")
        // Embeds input masks, which the pipeline never gives.
        .skip("prompt_encoder.mask_embed")
        .rename("mask_decoder.iou_token.weight", "iou_token")
        .rename("mask_decoder.mask_tokens.weight", "mask_tokens")
        .rename(
            "mask_decoder.transformer.layers.*.layer_norm1",
            "two_way.*.norm1",
        )
        .rename(
            "mask_decoder.transformer.layers.*.layer_norm2",
            "two_way.*.norm2",
        )
        .rename(
            "mask_decoder.transformer.layers.*.layer_norm3",
            "two_way.*.norm3",
        )
        .rename(
            "mask_decoder.transformer.layers.*.layer_norm4",
            "two_way.*.norm4",
        )
        .rename("mask_decoder.transformer.layers.*.mlp.lin1", "two_way.*.up")
        .rename(
            "mask_decoder.transformer.layers.*.mlp.lin2",
            "two_way.*.down",
        )
        .rename(
            "mask_decoder.transformer.layer_norm_final_attn",
            "final_norm",
        )
        .rename("mask_decoder.upscale_conv1", "upscale.0")
        .rename("mask_decoder.upscale_conv2", "upscale.1")
        .rename("mask_decoder.upscale_layer_norm", "upscale_norm");
    for (from, to) in [
        ("self_attn", "self_attention"),
        ("cross_attn_token_to_image", "token_to_image"),
        ("cross_attn_image_to_token", "image_to_token"),
    ] {
        rules = attention(
            rules,
            &format!("mask_decoder.transformer.layers.*.{from}"),
            &format!("two_way.*.{to}"),
        );
    }

    rules = attention(
        rules,
        "mask_decoder.transformer.final_attn_token_to_image",
        "final_attention",
    );
    rules = feed_forward(
        rules,
        "mask_decoder.output_hypernetworks_mlps.*",
        "hypernetworks.*",
        3,
    );
    feed_forward(
        rules,
        "mask_decoder.iou_prediction_head",
        "iou_head",
        iou_depth,
    )
}
