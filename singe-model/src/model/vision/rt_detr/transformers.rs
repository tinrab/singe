//! Reads a `transformers` RT-DETR or RT-DETRv2 config, whose backbone is its own `ResNet`-vd, and imports its checkpoint.

use serde::Deserialize;
use singe::nn::Activation;

use crate::{
    assembly::{ResNetConfig, attention_rules, deformable_decoder_rules},
    error::{Error, Result},
    format::{
        Rules,
        transformers::{activation_field, classifier_config, detection_scoring},
    },
    model::vision::rt_detr::RtDetrConfig,
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct BackboneFields {
    model_type: String,
    #[serde(default = "default_depths")]
    depths: Vec<i64>,
    #[serde(default = "default_embedding_size")]
    embedding_size: i64,
    #[serde(default = "default_layer_type")]
    layer_type: String,
    #[serde(default)]
    downsample_in_bottleneck: bool,
    #[serde(default)]
    downsample_in_first_stage: bool,
    #[serde(default)]
    out_indices: Vec<usize>,
}

fn default_depths() -> Vec<i64> {
    vec![3, 4, 6, 3]
}

const fn default_embedding_size() -> i64 {
    64
}

fn default_layer_type() -> String {
    "bottleneck".to_owned()
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    backbone_config: BackboneFields,
    #[serde(default)]
    use_timm_backbone: bool,
    d_model: i64,
    encoder_hidden_dim: i64,
    hidden_expansion: f32,
    encode_proj_layers: Vec<usize>,
    encoder_layers: i64,
    encoder_attention_heads: i64,
    encoder_ffn_dim: i64,
    #[serde(deserialize_with = "activation_field")]
    encoder_activation_function: Activation,
    #[serde(deserialize_with = "activation_field")]
    activation_function: Activation,
    positional_encoding_temperature: f32,
    num_feature_levels: usize,
    decoder_layers: i64,
    decoder_attention_heads: i64,
    decoder_ffn_dim: i64,
    decoder_n_points: i64,
    #[serde(deserialize_with = "activation_field")]
    decoder_activation_function: Activation,
    layer_norm_eps: f32,
    num_queries: i64,
    #[serde(default)]
    learn_initial_query: bool,
    #[serde(default)]
    normalize_before: bool,
    decoder_n_levels: Option<usize>,
    decoder_offset_scale: Option<f32>,
    decoder_method: Option<String>,
}

pub(crate) fn config(source: &Source) -> Result<RtDetrConfig> {
    let fields: Fields = source.json("config.json")?;
    let backbone = &fields.backbone_config;
    if fields.use_timm_backbone
        || backbone.model_type != "rt_detr_resnet"
        || backbone.layer_type != "bottleneck"
        || backbone.downsample_in_bottleneck
        || backbone.downsample_in_first_stage
    {
        return Err(Error::unsupported(
            "an RT-DETR backbone other than its bottleneck ResNet-vd",
        ));
    }

    // The levels are the backbone's last three stages, each with its own projection, as `out_indices` (2, 3, 4) has them.
    let stages = backbone.depths.len();
    if fields.num_feature_levels != 3
        || (!backbone.out_indices.is_empty()
            && backbone.out_indices != [stages - 2, stages - 1, stages])
    {
        return Err(Error::unsupported(
            "RT-DETR levels other than the backbone's last three stages",
        ));
    }

    if fields.encode_proj_layers.len() != 1
        || fields
            .encode_proj_layers
            .iter()
            .any(|&level| level >= fields.num_feature_levels)
        || fields.d_model != fields.encoder_hidden_dim
        || fields.learn_initial_query
        || fields.normalize_before
    {
        return Err(Error::unsupported(
            "an RT-DETR encoder other than one post-norm attention stack on one of the levels, or learned initial queries",
        ));
    }

    // RT-DETRv2 samples as RT-DETR does when every level has the same power of two of points, which `1 / points` scales exactly, the offsets take half the box, and sampling is bilinear.
    let points_scale_exactly =
        fields.decoder_n_points > 0 && fields.decoder_n_points & (fields.decoder_n_points - 1) == 0;
    if fields
        .decoder_n_levels
        .is_some_and(|levels| levels != fields.num_feature_levels)
        || fields
            .decoder_offset_scale
            .is_some_and(|scale| scale != 0.5)
        || fields
            .decoder_method
            .as_deref()
            .is_some_and(|method| method != "default")
        || !points_scale_exactly
    {
        return Err(Error::unsupported(
            "RT-DETRv2 sampling other than RT-DETR's: other levels, an offset scale other than 0.5, discrete sampling, or points not a power of two",
        ));
    }

    Ok(RtDetrConfig {
        backbone: ResNetConfig {
            depths: backbone.depths.clone(),
            stem: backbone.embedding_size,
        },
        width: fields.d_model,
        hidden: (fields.encoder_hidden_dim as f32 * fields.hidden_expansion) as i64,
        attended_levels: fields.encode_proj_layers,
        encoder_layers: fields.encoder_layers,
        encoder_heads: fields.encoder_attention_heads,
        encoder_inner: fields.encoder_ffn_dim,
        encoder_activation: fields.encoder_activation_function,
        activation: fields.activation_function,
        temperature: fields.positional_encoding_temperature,
        decoder_layers: fields.decoder_layers,
        decoder_heads: fields.decoder_attention_heads,
        decoder_inner: fields.decoder_ffn_dim,
        decoder_points: fields.decoder_n_points,
        decoder_activation: fields.decoder_activation_function,
        norm_eps: fields.layer_norm_eps,
        queries: fields.num_queries,
        classifier: classifier_config(source)?,
        scoring: detection_scoring(source)?,
    })
}

/// Renames a convolution and batch norm stored as `conv` and `norm` under `from` to a [`ConvNorm`](crate::assembly::ConvNorm)'s names under `to`.
fn conv_norm_rules(from: &str, to: &str) -> Rules {
    Rules::new().rename_each(from, to, &[("conv", "convolution"), ("norm", "norm")])
}

/// Returns the rules of an RT-DETR of `decoder_layers` decoder layers.
pub(crate) fn rules(decoder_layers: i64) -> Rules {
    let backbone = "model.backbone.model";
    let mut rules = Rules::new()
        // Counts training updates, which evaluation never reads.
        .skip("model.encoder_input_proj.*.1.num_batches_tracked")
        .skip("model.decoder_input_proj.*.1.num_batches_tracked")
        .skip("model.encoder.lateral_convs.*.norm.num_batches_tracked")
        .skip("model.encoder.downsample_convs.*.norm.num_batches_tracked")
        // Embeds the denoising queries of training.
        .skip("model.denoising_class_embed")
        // RT-DETRv2's `1 / points` for each point, which the offsets' division by the points equals.
        .skip("model.decoder.layers.*.encoder_attn.n_points_scale")
        .rename_each(
            &format!("{backbone}.embedder.embedder.*"),
            "backbone.stem.*",
            &[("convolution", "convolution"), ("normalization", "norm")],
        )
        .rename_each(
            &format!("{backbone}.encoder.stages.*.layers.*"),
            "backbone.stages.*.*",
            &[
                ("layer.0.convolution", "reduce.convolution"),
                ("layer.0.normalization", "reduce.norm"),
                ("layer.1.convolution", "spatial.convolution"),
                ("layer.1.normalization", "spatial.norm"),
                ("layer.2.convolution", "expand.convolution"),
                ("layer.2.normalization", "expand.norm"),
                // A strided block's shortcut pools first, its projection second.
                ("shortcut.1.convolution", "shortcut.convolution"),
                ("shortcut.1.normalization", "shortcut.norm"),
                ("shortcut.convolution", "shortcut.convolution"),
                ("shortcut.normalization", "shortcut.norm"),
            ],
        );
    for (from, to) in [
        ("model.encoder_input_proj.*", "input_projections.*"),
        ("model.decoder_input_proj.*", "decoder_projections.*"),
    ] {
        rules = rules.rename_each(from, to, &[("0", "convolution"), ("1", "norm")]);
    }

    let encoder = "model.encoder.encoder.0.layers.*";
    rules = rules
        .then(attention_rules(
            &format!("{encoder}.self_attn"),
            "encoder.*.attention",
        ))
        .rename_each(
            encoder,
            "encoder.*",
            &[
                ("self_attn_layer_norm", "attention_norm"),
                ("fc1", "feed_forward.up"),
                ("fc2", "feed_forward.down"),
                ("final_layer_norm", "feed_forward_norm"),
            ],
        )
        .then(conv_norm_rules(
            "model.encoder.lateral_convs.*",
            "lateral.*",
        ))
        .then(conv_norm_rules(
            "model.encoder.downsample_convs.*",
            "downsample.*",
        ));
    for (from, to) in [("fpn_blocks", "top_down"), ("pan_blocks", "bottom_up")] {
        let from = format!("model.encoder.{from}.*");
        let to = format!("{to}.*");
        for (part, role) in [("conv1", "first"), ("conv2", "second"), ("conv3", "output")] {
            rules = rules
                .skip(&format!("{from}.{part}.norm.num_batches_tracked"))
                .then(conv_norm_rules(
                    &format!("{from}.{part}"),
                    &format!("{to}.{role}"),
                ));
        }
        for (part, role) in [("conv1", "wide"), ("conv2", "narrow")] {
            rules = rules
                .skip(&format!(
                    "{from}.bottlenecks.*.{part}.norm.num_batches_tracked"
                ))
                .then(conv_norm_rules(
                    &format!("{from}.bottlenecks.*.{part}"),
                    &format!("{to}.blocks.*.{role}"),
                ));
        }
    }

    rules
        .rename("model.enc_output.0", "proposal")
        .rename("model.enc_output.1", "proposal_norm")
        .rename("model.enc_score_head", "proposal_scores")
        .rename("model.enc_bbox_head.layers", "proposal_boxes")
        .rename("model.decoder.query_pos_head.layers", "query_positions")
        .rename(
            &format!("model.decoder.class_embed.{}", decoder_layers - 1),
            "classes",
        )
        // The other layers' class heads score only the intermediate predictions of training.
        .skip("model.decoder.class_embed")
        .rename("model.decoder.bbox_embed.*.layers", "boxes.*")
        .then(deformable_decoder_rules(
            "model.decoder.layers.*",
            "decoder.*",
        ))
}
