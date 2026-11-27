//! Reads a `transformers` SAM 2 config, whose video checkpoints `transformers` loads as `Sam2Model` for images, and imports the image path's checkpoint, with the video memory's when the model tracks.

use serde::Deserialize;
use singe::nn::Activation;

use crate::{
    assembly::attention_rules_with_output,
    error::{Error, Result},
    format::{RuleSet, join, transformers::activation_field},
    model::vision::{
        sam::{TwoWaySizes, transformers::linear_stack_rules},
        sam2::{HieraBlock, Sam2Config, Sam2MemoryConfig},
    },
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct BackboneFields {
    blocks_per_stage: Vec<usize>,
    embed_dim_per_stage: Vec<i64>,
    num_attention_heads_per_stage: Vec<i64>,
    window_size_per_stage: Vec<i64>,
    global_attention_blocks: Vec<usize>,
    query_stride: [i64; 2],
    num_query_pool_stages: usize,
    mlp_ratio: f64,
    #[serde(deserialize_with = "activation_field")]
    hidden_act: Activation,
    layer_norm_eps: f32,
    hidden_size: i64,
    patch_kernel_size: [i64; 2],
    patch_stride: [i64; 2],
    patch_padding: [i64; 2],
    window_positional_embedding_background_size: [i64; 2],
    image_size: [i64; 2],
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct VisionFields {
    backbone_config: BackboneFields,
    fpn_hidden_size: i64,
    fpn_top_down_levels: Vec<usize>,
    num_feature_levels: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct DecoderFields {
    hidden_size: i64,
    num_hidden_layers: i64,
    num_attention_heads: i64,
    mlp_dim: i64,
    attention_downsample_rate: i64,
    iou_head_depth: i64,
    iou_head_hidden_dim: i64,
    num_multimask_outputs: i64,
    #[serde(default = "stability_delta")]
    dynamic_multimask_stability_delta: f32,
    #[serde(default = "stability_threshold")]
    dynamic_multimask_stability_thresh: f32,
}

const fn stability_delta() -> f32 {
    0.05
}

const fn stability_threshold() -> f32 {
    0.98
}

/// The memory's embeddings a config turns on, both of which Singe's memory has.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
struct MemoryEmbeddingFields {
    enable_occlusion_spatial_embedding: bool,
    enable_temporal_pos_encoding_for_object_pointers: bool,
}

/// Where a config has the decoder give several masks, which Singe's tracker does for prompted and tracked frames alike.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
struct MultimaskFields {
    multimask_output_in_sam: bool,
    multimask_output_for_tracking: bool,
}

/// The video memory's fields, which a `sam2_video` config has.
#[derive(Debug, Clone, PartialEq, Deserialize)]
struct MemoryFields {
    memory_attention_num_layers: i64,
    memory_attention_hidden_size: i64,
    memory_attention_num_attention_heads: i64,
    memory_attention_downsample_rate: i64,
    memory_attention_feed_forward_hidden_size: i64,
    #[serde(deserialize_with = "activation_field")]
    memory_attention_feed_forward_hidden_act: Activation,
    memory_attention_rope_theta: f32,
    memory_attention_rope_feat_sizes: [i64; 2],
    memory_encoder_hidden_size: i64,
    memory_encoder_output_channels: i64,
    num_maskmem: usize,
    max_object_pointers_in_encoder: usize,
    mask_downsampler_embed_dim: i64,
    mask_downsampler_kernel_size: i64,
    mask_downsampler_stride: i64,
    mask_downsampler_padding: i64,
    mask_downsampler_total_stride: i64,
    #[serde(deserialize_with = "activation_field")]
    mask_downsampler_hidden_act: Activation,
    memory_fuser_embed_dim: i64,
    memory_fuser_intermediate_dim: i64,
    memory_fuser_kernel_size: i64,
    memory_fuser_padding: i64,
    memory_fuser_num_layers: i64,
    #[serde(deserialize_with = "activation_field")]
    memory_fuser_hidden_act: Activation,
    sigmoid_scale_for_mem_enc: f32,
    sigmoid_bias_for_mem_enc: f32,
    #[serde(flatten)]
    embeddings: MemoryEmbeddingFields,
    #[serde(flatten)]
    several_masks: MultimaskFields,
    multimask_min_pt_num: usize,
    multimask_max_pt_num: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    vision_config: VisionFields,
    mask_decoder_config: DecoderFields,
    #[serde(flatten)]
    memory: Option<MemoryFields>,
}

/// Returns the memory's config, refusing what Singe's memory does not build: a memory attention that narrows its width, a downsampler or fuser of other shapes, and the configs that leave out occlusion, pointer positions, or the several masks on one point.
fn memory_config(
    memory: &MemoryFields,
    fpn: i64,
    decoder: &DecoderFields,
    coarsest: i64,
) -> Result<Sam2MemoryConfig> {
    let downsampler_layers = memory.mask_downsampler_total_stride.ilog2()
        / memory.mask_downsampler_stride.max(2).ilog2();
    let fits = memory.memory_attention_hidden_size == fpn
        && memory.memory_attention_downsample_rate == 1
        && memory.memory_attention_rope_feat_sizes == [coarsest, coarsest]
        && memory.memory_encoder_hidden_size == fpn
        && memory.mask_downsampler_embed_dim == fpn
        && memory.mask_downsampler_stride == 2
        && memory.mask_downsampler_kernel_size == 3
        && memory.mask_downsampler_padding == 1
        && 4_i64.pow(downsampler_layers) == fpn
        && memory.memory_fuser_embed_dim == fpn
        && memory.memory_fuser_intermediate_dim == 4 * fpn
        && memory.memory_fuser_kernel_size == 7
        && memory.memory_fuser_padding == 3
        && memory.embeddings.enable_occlusion_spatial_embedding
        && memory
            .embeddings
            .enable_temporal_pos_encoding_for_object_pointers
        && memory.several_masks.multimask_output_in_sam
        && memory.several_masks.multimask_output_for_tracking
        && memory.multimask_min_pt_num == 0
        && memory.multimask_max_pt_num == 1
        && fpn % memory.memory_encoder_output_channels == 0;
    if !fits {
        return Err(Error::unsupported(
            "a SAM 2 video memory other than `facebook/sam2.1`'s layout",
        ));
    }

    Ok(Sam2MemoryConfig {
        layers: memory.memory_attention_num_layers,
        heads: memory.memory_attention_num_attention_heads,
        inner: memory.memory_attention_feed_forward_hidden_size,
        activation: memory.memory_attention_feed_forward_hidden_act,
        rope_theta: memory.memory_attention_rope_theta,
        channels: memory.memory_encoder_output_channels,
        memories: memory.num_maskmem,
        pointers: memory.max_object_pointers_in_encoder,
        downsampler_layers: downsampler_layers as usize,
        downsampler_activation: memory.mask_downsampler_hidden_act,
        fuser_layers: memory.memory_fuser_num_layers,
        fuser_activation: memory.memory_fuser_hidden_act,
        mask_scale: memory.sigmoid_scale_for_mem_enc,
        mask_bias: memory.sigmoid_bias_for_mem_enc,
        stability_delta: decoder.dynamic_multimask_stability_delta,
        stability_threshold: decoder.dynamic_multimask_stability_thresh,
    })
}

pub(crate) fn config(source: &Source) -> Result<Sam2Config> {
    let Fields {
        vision_config: vision,
        mask_decoder_config: decoder,
        memory,
    } = source.json("config.json")?;
    let backbone = vision.backbone_config;
    let square = |pair: [i64; 2], what: &str| {
        if pair[0] == pair[1] {
            Ok(pair[0])
        } else {
            Err(Error::unsupported(format!(
                "a SAM 2 {what} that is not square"
            )))
        }
    };
    let mut blocks = Vec::new();
    let mut stage_ends = Vec::new();
    for (stage, &count) in backbone.blocks_per_stage.iter().enumerate() {
        for block in 0..count {
            let starts = stage > 0 && block == 0;
            let from = if starts { stage - 1 } else { stage };
            let window = if backbone.global_attention_blocks.contains(&blocks.len()) {
                0
            } else {
                backbone.window_size_per_stage[from]
            };
            blocks.push(HieraBlock {
                input: backbone.embed_dim_per_stage[from],
                output: backbone.embed_dim_per_stage[stage],
                heads: backbone.num_attention_heads_per_stage[stage],
                window,
                pools: starts && stage <= backbone.num_query_pool_stages,
            });
        }

        stage_ends.push(blocks.len() - 1);
    }

    let image_size = square(backbone.image_size, "image")?;
    let patch_stride = square(backbone.patch_stride, "patch stride")?;
    let query_stride = square(backbone.query_stride, "query stride")?;
    let poolings = u32::try_from(vision.num_feature_levels.saturating_sub(1))
        .map_err(|_| Error::unsupported("a SAM 2 neck with that many levels"))?;
    // The side of the coarsest map the decoder reads: the patches' side, pooled once per level after the finest.
    let coarsest = image_size / patch_stride / query_stride.pow(poolings);
    let memory = memory
        .map(|memory| memory_config(&memory, vision.fpn_hidden_size, &decoder, coarsest))
        .transpose()?;
    Ok(Sam2Config {
        blocks,
        stage_ends,
        hidden: backbone.hidden_size,
        patch_kernel: square(backbone.patch_kernel_size, "patch kernel")?,
        patch_stride,
        patch_padding: square(backbone.patch_padding, "patch padding")?,
        background: square(
            backbone.window_positional_embedding_background_size,
            "background",
        )?,
        window_positions: backbone.window_size_per_stage[0],
        query_stride,
        mlp_ratio: backbone.mlp_ratio,
        activation: backbone.hidden_act,
        eps: backbone.layer_norm_eps,
        fpn: vision.fpn_hidden_size,
        top_down: vision.fpn_top_down_levels,
        levels: vision.num_feature_levels,
        image_size,
        decoder: TwoWaySizes {
            embedding: decoder.hidden_size,
            heads: decoder.num_attention_heads,
            inner: decoder.mlp_dim,
            // `Sam2FeedForward`'s default, which the two-way blocks keep.
            activation: Activation::Relu,
            downsample: decoder.attention_downsample_rate,
            eps: 1e-5,
        },
        decoder_layers: decoder.num_hidden_layers,
        masks: decoder.num_multimask_outputs,
        iou_depth: decoder.iou_head_depth,
        iou_hidden: decoder.iou_head_hidden_dim,
        memory,
    })
}

/// Imports the image path's weights, and the video memory's when `tracking`.
pub(crate) fn rules(config: &Sam2Config, tracking: bool) -> RuleSet {
    let backbone = "vision_encoder.backbone";
    let two_way = "mask_decoder.transformer.layers.*";
    let mut rules = RuleSet::new()
        .rename(&format!("{backbone}.patch_embed.projection"), "patches")
        .rename(&format!("{backbone}.pos_embed_window"), "window_positions")
        .rename(&format!("{backbone}.pos_embed"), "background")
        .rename_each(
            &format!("{backbone}.blocks.*"),
            "blocks.*",
            &[
                ("layer_norm1", "attention_norm"),
                ("attn.qkv", "qkv"),
                ("attn.proj", "output"),
                ("layer_norm2", "feed_forward_norm"),
                ("mlp.proj_in", "feed_forward.up"),
                ("mlp.proj_out", "feed_forward.down"),
                ("proj", "projection"),
            ],
        )
        .rename("vision_encoder.neck.convs", "lateral")
        .rename("no_memory_embedding", "no_memory")
        .rename("shared_image_embedding.positional_embedding", "gaussian")
        .rename_each(
            "prompt_encoder",
            "",
            &[
                ("shared_embedding.positional_embedding", "point_gaussian"),
                ("point_embed.weight", "point_embeddings"),
                ("not_a_point_embed.weight", "not_a_point"),
                ("no_mask_embed.weight", "no_mask"),
            ],
        )
        .rename_each(
            "mask_decoder",
            "",
            &[
                ("obj_score_token.weight", "object_token"),
                ("iou_token.weight", "iou_token"),
                ("mask_tokens.weight", "mask_tokens"),
                ("transformer.layer_norm_final_attn", "final_norm"),
                ("conv_s0", "finest"),
                ("conv_s1", "finer"),
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
                ("mlp.proj_in", "feed_forward.up"),
                ("mlp.proj_out", "feed_forward.down"),
                ("layer_norm3", "feed_forward_norm"),
                ("layer_norm4", "image_to_token_norm"),
            ],
        )
        .then(attention_rules_with_output(
            &join(two_way, "self_attn"),
            "two_way.*.attention",
            "o_proj",
        ))
        .then(attention_rules_with_output(
            &join(two_way, "cross_attn_token_to_image"),
            "two_way.*.token_to_image",
            "o_proj",
        ))
        .then(attention_rules_with_output(
            &join(two_way, "cross_attn_image_to_token"),
            "two_way.*.image_to_token",
            "o_proj",
        ))
        .then(attention_rules_with_output(
            "mask_decoder.transformer.final_attn_token_to_image",
            "final_attention",
            "o_proj",
        ))
        .then(linear_stack_rules(
            "mask_decoder.output_hypernetworks_mlps.*",
            "hypernetworks.*",
            3,
        ))
        .then(linear_stack_rules(
            "mask_decoder.iou_prediction_head",
            "iou_head",
            config.iou_depth,
        ))
        .then(linear_stack_rules(
            "mask_decoder.pred_obj_score_head",
            "object_score",
            3,
        ));
    // Neither path reads a mask prompt, and the memory's `no_memory_positional_encoding` goes unread in `transformers` too.
    for unread in [
        "prompt_encoder.mask_embed",
        "mask_downsample",
        "no_memory_positional_encoding",
    ] {
        rules = rules.skip(unread);
    }

    if tracking {
        return rules.then(memory_rules());
    }

    for unread in [
        "memory_attention",
        "memory_encoder",
        "memory_temporal_positional_encoding",
        "no_object_pointer",
        "object_pointer_proj",
        "occlusion_spatial_embedding_parameter",
        "temporal_positional_encoding_projection_layer",
    ] {
        rules = rules.skip(unread);
    }
    rules
}

/// Imports the video memory's weights under `memory`.
fn memory_rules() -> RuleSet {
    let layers = "memory_attention.layers.*";
    RuleSet::new()
        .rename_each(
            layers,
            "memory.layers.*",
            &[
                ("layer_norm1", "self_norm"),
                ("layer_norm2", "cross_norm"),
                ("layer_norm3", "feed_forward_norm"),
                ("linear1", "feed_forward.up"),
                ("linear2", "feed_forward.down"),
            ],
        )
        .then(attention_rules_with_output(
            &join(layers, "self_attn"),
            "memory.layers.*.self_attention",
            "o_proj",
        ))
        .then(attention_rules_with_output(
            &join(layers, "cross_attn_image"),
            "memory.layers.*.cross_attention",
            "o_proj",
        ))
        .rename("memory_attention.layer_norm", "memory.norm")
        .rename_each(
            "memory_encoder.mask_downsampler",
            "memory",
            &[
                ("layers", "downsampler"),
                ("final_conv", "downsampler_output"),
            ],
        )
        .rename_each(
            "memory_encoder.memory_fuser.layers.*",
            "memory.fuser.*",
            &[
                ("depthwise_conv", "depthwise"),
                ("layer_norm", "norm"),
                ("pointwise_conv1", "up"),
                ("pointwise_conv2", "down"),
                ("scale", "scale"),
            ],
        )
        .rename_each(
            "memory_encoder",
            "memory",
            &[
                ("feature_projection", "feature_projection"),
                ("projection", "projection"),
            ],
        )
        .rename("memory_temporal_positional_encoding", "memory.temporal")
        .rename("no_object_pointer", "memory.no_object_pointer")
        .rename("occlusion_spatial_embedding_parameter", "memory.occlusion")
        .rename(
            "temporal_positional_encoding_projection_layer",
            "memory.pointer_positions",
        )
        .then(linear_stack_rules(
            "object_pointer_proj",
            "memory.pointer_projection",
            3,
        ))
}
