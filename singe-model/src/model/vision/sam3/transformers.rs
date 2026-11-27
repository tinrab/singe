//! Reads a `transformers` SAM 3 config and imports its checkpoint under the same names, its attention projections, feed-forwards, and small stacks of linear layers renamed to Singe's.

use serde::Deserialize;
use singe::nn::Activation;

use crate::{
    assembly::attention_rules_with_output,
    error::{Error, Result},
    format::{RuleSet, transformers::activation_field},
    model::{
        multimodal::clip::transformers::{TextFields, text_config, text_rules},
        vision::sam3::{Sam3Config, Sam3VitConfig},
    },
    source::Source,
};

#[derive(Debug, Deserialize)]
struct BackboneFields {
    hidden_size: i64,
    intermediate_size: i64,
    num_attention_heads: i64,
    num_hidden_layers: i64,
    #[serde(deserialize_with = "activation_field")]
    hidden_act: Activation,
    layer_norm_eps: f32,
    image_size: i64,
    patch_size: i64,
    pretrain_image_size: i64,
    window_size: i64,
    global_attn_indexes: Vec<usize>,
    rope_theta: f32,
    num_channels: i64,
}

#[derive(Debug, Deserialize)]
struct VisionFields {
    backbone_config: BackboneFields,
    fpn_hidden_size: i64,
    scale_factors: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct StackFields {
    hidden_size: i64,
    num_attention_heads: i64,
    intermediate_size: i64,
    num_layers: i64,
    #[serde(deserialize_with = "activation_field")]
    hidden_act: Activation,
}

#[derive(Debug, Deserialize)]
struct DecoderFields {
    #[serde(flatten)]
    stack: StackFields,
    num_queries: i64,
    use_presence_token: bool,
    box_rpb_mode: String,
}

#[derive(Debug, Deserialize)]
struct MaskFields {
    hidden_size: i64,
    num_attention_heads: i64,
    num_upsampling_stages: i64,
}

#[derive(Debug, Deserialize)]
struct Fields {
    #[serde(rename = "vision_config")]
    vision: VisionFields,
    #[serde(rename = "text_config")]
    text: TextFields,
    #[serde(rename = "detr_encoder_config")]
    encoder: StackFields,
    #[serde(rename = "detr_decoder_config")]
    decoder: DecoderFields,
    #[serde(rename = "mask_decoder_config")]
    mask: MaskFields,
}

/// The pyramid levels the detector reads, of the neck's four: four, two, and one times the patches' grid.
const SCALES: &[f64] = &[4.0, 2.0, 1.0];

pub(crate) fn config(source: &Source) -> Result<Sam3Config> {
    let fields: Fields = source.json("config.json")?;
    let backbone = fields.vision.backbone_config;
    let (encoder, decoder, mask) = (fields.encoder, fields.decoder, fields.mask);
    let width = encoder.hidden_size;
    let fits = fields.vision.scale_factors.starts_with(SCALES)
        && decoder.use_presence_token
        && decoder.box_rpb_mode == "log"
        && decoder.stack.hidden_size == width
        && decoder.stack.num_attention_heads == encoder.num_attention_heads
        && mask.hidden_size == width
        && mask.num_upsampling_stages == 3
        && fields.vision.fpn_hidden_size == width;
    if !fits {
        return Err(Error::unsupported(
            "a SAM 3 other than `facebook/sam3`'s layout: its pyramid, presence token, box biases, and widths",
        ));
    }

    Ok(Sam3Config {
        vit: Sam3VitConfig {
            width: backbone.hidden_size,
            layers: backbone.num_hidden_layers,
            heads: backbone.num_attention_heads,
            inner: backbone.intermediate_size,
            activation: backbone.hidden_act,
            eps: backbone.layer_norm_eps,
            patch: backbone.patch_size,
            image_size: backbone.image_size,
            pretrain_side: backbone.pretrain_image_size / backbone.patch_size,
            window: backbone.window_size,
            global: backbone.global_attn_indexes,
            theta: backbone.rope_theta,
            channels: backbone.num_channels,
        },
        text: text_config(&fields.text),
        width,
        heads: encoder.num_attention_heads,
        inner: encoder.intermediate_size,
        encoder_layers: encoder.num_layers,
        encoder_activation: encoder.hidden_act,
        decoder_layers: decoder.stack.num_layers,
        decoder_inner: decoder.stack.intermediate_size,
        decoder_activation: decoder.stack.hidden_act,
        queries: decoder.num_queries,
        mask_heads: mask.num_attention_heads,
    })
}

pub(crate) fn rules() -> RuleSet {
    let mut rules = RuleSet::new()
        // Text prompts read no geometry, the detector no coarsest level, and the instances no semantic map.
        .skip("geometry_encoder")
        .skip("vision_encoder.neck.fpn_layers.3")
        .skip("mask_decoder.semantic_projection")
        .skip("text_encoder.text_projection")
        .then(text_rules("text_encoder.text_model", "text"))
        .rename(
            "vision_encoder.backbone.embeddings.patch_embeddings.projection",
            "vision_encoder.backbone.patches",
        )
        .rename(
            "vision_encoder.backbone.embeddings.position_embeddings",
            "vision_encoder.backbone.positions",
        )
        .rename(
            "mask_decoder.mask_embedder.layers",
            "mask_decoder.mask_embedder",
        );
    for attention in [
        "vision_encoder.backbone.layers.*.attention",
        "detr_encoder.layers.*.self_attn",
        "detr_encoder.layers.*.cross_attn",
        "detr_decoder.layers.*.self_attn",
        "detr_decoder.layers.*.text_cross_attn",
        "detr_decoder.layers.*.vision_cross_attn",
        "mask_decoder.prompt_cross_attn",
    ] {
        rules = rules.then(attention_rules_with_output(attention, attention, "o_proj"));
    }

    for mlp in [
        "vision_encoder.backbone.layers.*.mlp",
        "detr_encoder.layers.*.mlp",
        "detr_decoder.layers.*.mlp",
    ] {
        rules = rules.rename_each(mlp, mlp, &[("fc1", "up"), ("fc2", "down")]);
    }

    for stack in [
        "detr_decoder.box_head",
        "detr_decoder.presence_head",
        "detr_decoder.ref_point_head",
        "detr_decoder.box_rpb_embed_x",
        "detr_decoder.box_rpb_embed_y",
        "dot_product_scoring.text_mlp",
    ] {
        rules = rules.rename_each(
            stack,
            stack,
            &[("layer1", "0"), ("layer2", "1"), ("layer3", "2")],
        );
    }

    for part in [
        "vision_encoder",
        "detr_encoder",
        "detr_decoder",
        "dot_product_scoring",
        "mask_decoder",
        "text_projection",
    ] {
        rules = rules.rename(part, part);
    }
    rules
}
