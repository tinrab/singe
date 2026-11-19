// Reads a `transformers` SAM config, of a vision encoder, a prompt encoder, and a mask decoder, and imports its checkpoint.

use serde::Deserialize;
use singe::nn::Activation;

use crate::{
    assembly::attention_rules,
    error::{Error, Result},
    format::{Rules, join, transformers::activation_field},
    model::vision::sam::SamConfig,
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
    if !vision.use_abs_pos {
        return Err(Error::unsupported(
            "a SAM vision encoder without absolute positions",
        ));
    }

    if !vision.use_rel_pos {
        return Err(Error::unsupported(
            "a SAM vision encoder without relative positions",
        ));
    }

    if !vision.qkv_bias {
        return Err(Error::unsupported(
            "a SAM vision encoder without query, key, and value biases",
        ));
    }

    if vision.hidden_act != Activation::GeluExact {
        return Err(Error::unsupported(format!(
            "SAM vision activation {:?}",
            vision.hidden_act
        )));
    }

    if decoder.hidden_size != vision.output_channels {
        return Err(Error::unsupported(format!(
            "a SAM mask decoder of width {} over vision embeddings of width {}",
            decoder.hidden_size, vision.output_channels
        )));
    }

    if (vision.layer_norm_eps - decoder.layer_norm_eps).abs() > f32::EPSILON {
        return Err(Error::unsupported(format!(
            "SAM vision norm epsilon {} beside mask decoder norm epsilon {}",
            vision.layer_norm_eps, decoder.layer_norm_eps
        )));
    }

    if decoder.iou_head_depth < 2 {
        return Err(Error::unsupported(format!(
            "SAM IoU head depth {}",
            decoder.iou_head_depth
        )));
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
fn linear_stack_rules(from: &str, to: &str, depth: i64) -> Rules {
    let mut rules = Rules::new()
        .rename(&join(from, "proj_in"), &join(to, "0"))
        .rename(&join(from, "proj_out"), &join(to, &(depth - 1).to_string()));
    for index in 0..depth - 2 {
        rules = rules.rename(
            &join(from, &format!("layers.{index}")),
            &join(to, &(index + 1).to_string()),
        );
    }
    rules
}

pub(crate) fn rules(iou_depth: i64) -> Rules {
    let two_way = "mask_decoder.transformer.layers.*";
    Rules::new()
        .rename("vision_encoder.patch_embed.projection", "patches")
        .rename("vision_encoder.pos_embed", "positions")
        .rename_each(
            "vision_encoder.layers.*",
            "blocks.*",
            &[
                ("layer_norm1", "attention_norm"),
                ("attn.qkv", "attention.qkv"),
                ("attn.proj", "attention.output"),
                ("attn.rel_pos_h", "attention.relative_height"),
                ("attn.rel_pos_w", "attention.relative_width"),
                ("layer_norm2", "feed_forward_norm"),
                ("mlp.lin1", "feed_forward.up"),
                ("mlp.lin2", "feed_forward.down"),
            ],
        )
        .rename_each(
            "vision_encoder.neck",
            "",
            &[
                ("conv1", "neck.0"),
                ("conv2", "neck.1"),
                ("layer_norm1", "neck_norms.0"),
                ("layer_norm2", "neck_norms.1"),
            ],
        )
        .rename("shared_image_embedding.positional_embedding", "gaussian")
        .rename_each(
            "prompt_encoder",
            "",
            &[
                ("point_embed.*.weight", "point_embeddings.*"),
                ("not_a_point_embed.weight", "not_a_point"),
                ("no_mask_embed.weight", "no_mask"),
            ],
        )
        // Embeds input masks, which the pipeline never gives.
        .skip("prompt_encoder.mask_embed")
        .rename_each(
            "mask_decoder",
            "",
            &[
                ("iou_token.weight", "iou_token"),
                ("mask_tokens.weight", "mask_tokens"),
                ("transformer.layer_norm_final_attn", "final_norm"),
                ("upscale_conv1", "upscale.0"),
                ("upscale_conv2", "upscale.1"),
                ("upscale_layer_norm", "upscale_norm"),
            ],
        )
        .rename_each(
            two_way,
            "two_way.*",
            &[
                ("layer_norm1", "attention_norm"),
                ("layer_norm2", "token_to_image_norm"),
                ("mlp.lin1", "feed_forward.up"),
                ("mlp.lin2", "feed_forward.down"),
                ("layer_norm3", "feed_forward_norm"),
                ("layer_norm4", "image_to_token_norm"),
            ],
        )
        .then(attention_rules(
            &join(two_way, "self_attn"),
            "two_way.*.attention",
        ))
        .then(attention_rules(
            &join(two_way, "cross_attn_token_to_image"),
            "two_way.*.token_to_image",
        ))
        .then(attention_rules(
            &join(two_way, "cross_attn_image_to_token"),
            "two_way.*.image_to_token",
        ))
        .then(attention_rules(
            "mask_decoder.transformer.final_attn_token_to_image",
            "final_attention",
        ))
        .then(linear_stack_rules(
            "mask_decoder.output_hypernetworks_mlps.*",
            "hypernetworks.*",
            3,
        ))
        .then(linear_stack_rules(
            "mask_decoder.iou_prediction_head",
            "iou_head",
            iou_depth,
        ))
}
