//! Reads a `transformers` `PatchTST` config and imports its checkpoint.

use serde::Deserialize;
use singe::nn::Activation;

use crate::{
    error::{Error, Result},
    format::{Rules, transformers::activation_field},
    model::time_series::patchtst::{PatchNorm, PatchPooling, PatchTstConfig},
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    context_length: i64,
    num_input_channels: i64,
    patch_length: i64,
    patch_stride: i64,
    d_model: i64,
    num_attention_heads: i64,
    num_hidden_layers: i64,
    ffn_dim: i64,
    #[serde(deserialize_with = "activation_field")]
    activation_function: Activation,
    norm_eps: f32,
    norm_type: String,
    scaling: Option<String>,
    #[serde(flatten)]
    layout: LayoutFields,
    #[serde(flatten)]
    sharing: SharingFields,
    loss: String,
    prediction_length: i64,
    #[serde(default)]
    pooling_type: Option<String>,
    #[serde(default)]
    do_mask_input: Option<bool>,
}

/// How the blocks are laid out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
struct LayoutFields {
    pre_norm: bool,
    use_cls_token: bool,
    channel_attention: bool,
}

/// Whether the channels share the patch embedding and the head's projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
struct SharingFields {
    share_embedding: bool,
    share_projection: bool,
}

pub(crate) fn config(source: &Source) -> Result<PatchTstConfig> {
    let fields: Fields = source.json("config.json")?;
    let norm = match fields.norm_type.as_str() {
        "batchnorm" => PatchNorm::Batch,
        "layernorm" => PatchNorm::Layer,
        other => {
            return Err(Error::unsupported(format!("PatchTST norm type `{other}`")));
        }
    };
    let pooling = match (fields.layout.use_cls_token, fields.pooling_type.as_deref()) {
        (true, _) => PatchPooling::Class,
        (false, Some("mean")) => PatchPooling::Mean,
        (false, None) => PatchPooling::Flatten,
        (false, Some(other)) => {
            return Err(Error::unsupported(format!("PatchTST pooling `{other}`")));
        }
    };
    if !fields.layout.pre_norm
        || fields.layout.channel_attention
        || fields.scaling.as_deref() != Some("std")
        || !fields.sharing.share_embedding
        || !fields.sharing.share_projection
        || fields.loss != "mse"
        || fields.do_mask_input == Some(true)
    {
        return Err(Error::unsupported(
            "a PatchTST other than a pre-norm, std-scaled point forecaster without channel attention, with shared embedding and projection",
        ));
    }
    Ok(PatchTstConfig {
        context: fields.context_length,
        channels: fields.num_input_channels,
        patch: fields.patch_length,
        stride: fields.patch_stride,
        width: fields.d_model,
        heads: fields.num_attention_heads,
        layers: fields.num_hidden_layers,
        inner: fields.ffn_dim,
        activation: fields.activation_function,
        eps: fields.norm_eps,
        horizon: fields.prediction_length,
        minimum_scale: 1e-5,
        norm,
        pooling,
    })
}

pub(crate) fn rules() -> Rules {
    Rules::new()
        .rename("model.encoder.embedder.input_embedding", "embedding")
        .rename("model.encoder.positional_encoder.position_enc", "positions")
        .derive(
            "model.encoder.positional_encoder.cls_token",
            "class",
            |class| class.squeeze(0),
        )
        // A batch norm's count of the batches it saw in training, which evaluation never reads.
        .skip("model.encoder.layers.*.norm_sublayer1.batchnorm.num_batches_tracked")
        .skip("model.encoder.layers.*.norm_sublayer3.batchnorm.num_batches_tracked")
        .rename_each(
            "model.encoder.layers.*",
            "blocks.*",
            &[
                ("norm_sublayer1.batchnorm", "attention_norm"),
                ("norm_sublayer1", "attention_norm"),
                ("self_attn.q_proj", "attention.query"),
                ("self_attn.k_proj", "attention.key"),
                ("self_attn.v_proj", "attention.value"),
                ("self_attn.out_proj", "attention.output"),
                ("norm_sublayer3.batchnorm", "feed_forward_norm"),
                ("norm_sublayer3", "feed_forward_norm"),
                ("ff.0", "feed_forward.up"),
                ("ff.3", "feed_forward.down"),
            ],
        )
        .rename("head.projection", "head")
}
