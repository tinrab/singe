//! Reads Shap-E's `diffusers` components: the prior's and the renderer's configs and their tensors' names.

use std::iter;

use serde::Deserialize;
use singe::nn::Activation;

use crate::{
    error::{Error, Result},
    format::{
        Rules,
        diffusers::{activation_field, silu},
    },
    model::diffusion::shap_e::{PriorConfig, RendererConfig},
    source::Source,
};

#[expect(
    clippy::unnecessary_wraps,
    reason = "an explicit null in the config means no projection"
)]
fn linear() -> Option<String> {
    Some("linear".to_owned())
}

#[expect(
    clippy::unnecessary_wraps,
    reason = "an explicit null in the config means no added embedding"
)]
fn prd() -> Option<String> {
    Some("prd".to_owned())
}

const fn four() -> i64 {
    4
}

#[derive(Debug, Deserialize)]
struct PriorFields {
    num_attention_heads: i64,
    attention_head_dim: i64,
    num_layers: i64,
    num_embeddings: i64,
    embedding_dim: i64,
    #[serde(default)]
    embedding_proj_dim: Option<i64>,
    #[serde(default)]
    embedding_proj_norm_type: Option<String>,
    #[serde(default)]
    clip_embed_dim: Option<i64>,
    #[serde(default)]
    time_embed_dim: Option<i64>,
    #[serde(default = "silu", deserialize_with = "activation_field")]
    time_embed_act_fn: Activation,
    #[serde(default)]
    norm_in_type: Option<String>,
    #[serde(default = "prd")]
    added_emb_type: Option<String>,
    #[serde(default = "four")]
    additional_embeddings: i64,
    #[serde(default = "linear")]
    encoder_hid_proj_type: Option<String>,
}

pub(crate) fn prior_config(source: &Source) -> Result<PriorConfig> {
    let fields: PriorFields = source.json("config.json")?;
    if fields.norm_in_type.as_deref() != Some("layer")
        || fields.added_emb_type.is_some()
        || fields.additional_embeddings != 0
        || fields.encoder_hid_proj_type.is_some()
    {
        return Err(Error::unsupported(
            "PriorTransformer norm_in_type, added_emb_type, additional_embeddings, or encoder_hid_proj_type other than Shap-E's",
        ));
    }

    if fields.time_embed_act_fn != Activation::GeluExact {
        return Err(Error::unsupported(
            "PriorTransformer time_embed_act_fn other than gelu",
        ));
    }

    let condition_norm = match fields.embedding_proj_norm_type.as_deref() {
        None => false,
        Some("layer") => true,
        Some(other) => {
            return Err(Error::unsupported(format!(
                "PriorTransformer embedding_proj_norm_type {other}"
            )));
        }
    };
    let width = fields.num_attention_heads * fields.attention_head_dim;
    Ok(PriorConfig {
        width,
        heads: fields.num_attention_heads,
        layers: fields.num_layers,
        embeddings: fields.num_embeddings,
        embedding_width: fields.embedding_dim,
        condition_width: fields.embedding_proj_dim.unwrap_or(fields.embedding_dim),
        condition_norm,
        clip_width: fields.clip_embed_dim.unwrap_or(fields.embedding_dim),
        time_width: fields.time_embed_dim.unwrap_or(width),
    })
}

pub(crate) fn prior_rules() -> Rules {
    Rules::new()
        .rename("time_embedding.linear_1", "time.up")
        .rename("time_embedding.linear_2", "time.down")
        .rename("embedding_proj_norm", "embedding_norm")
        .rename("positional_embedding", "positions")
        .rename_each(
            "transformer_blocks.*",
            "blocks.*",
            &[
                ("norm1", "attention_norm"),
                ("attn1.to_q", "attention.query"),
                ("attn1.to_k", "attention.key"),
                ("attn1.to_v", "attention.value"),
                ("attn1.to_out.0", "attention.output"),
                ("norm3", "feed_forward_norm"),
                ("ff.net.0.proj", "feed_forward.up"),
                ("ff.net.2", "feed_forward.down"),
            ],
        )
        .rename("proj_to_clip_embeddings", "proj_to_clip")
        .rename("proj_in", "proj_in")
        .rename("embedding_proj", "embedding_proj")
        .rename("norm_in", "norm_in")
        .rename("norm_out", "norm_out")
        // The CLIP embedding's statistics, which only an image-embedding prior reads.
        .skip("clip_mean")
        .skip("clip_std")
}

#[derive(Debug, Deserialize)]
struct RendererFields {
    param_shapes: Vec<[i64; 2]>,
    d_latent: i64,
    d_hidden: i64,
    n_output: i64,
    n_hidden_layers: usize,
    insert_direction_at: usize,
    #[serde(deserialize_with = "activation_field")]
    act_fn: Activation,
}

pub(crate) fn renderer_config(source: &Source) -> Result<RendererConfig> {
    let fields: RendererFields = source.json("config.json")?;
    if fields.act_fn != Activation::Silu {
        return Err(Error::unsupported("ShapERenderer act_fn other than silu"));
    }

    // NeRF's encodings: a position's 3 features and 90 sines, and a direction's 3 and 48.
    let (position, direction) = (93, 51);
    let mut inputs = vec![position];
    inputs.extend(iter::repeat_n(fields.d_hidden, fields.n_hidden_layers));
    let mut outputs = vec![fields.d_hidden; fields.n_hidden_layers];
    outputs.push(fields.n_output);
    inputs[fields.insert_direction_at] += direction;
    Ok(RendererConfig {
        projected: fields.param_shapes,
        latent_width: fields.d_latent,
        layers: inputs
            .into_iter()
            .zip(outputs)
            .map(<[_; 2]>::from)
            .collect(),
        insert_direction_at: fields.insert_direction_at,
    })
}

pub(crate) fn renderer_rules(config: &RendererConfig) -> Rules {
    let projections = (0..config.projected.len())
        .map(|index| {
            (
                format!("projections.nerstf__mlp__{index}__weight"),
                index.to_string(),
            )
        })
        .collect::<Vec<_>>();
    let projections = projections
        .iter()
        .map(|(stored, index)| (stored.as_str(), index.as_str()))
        .collect::<Vec<_>>();
    Rules::new()
        .rename_each("params_proj", "params_proj", &projections)
        .rename("mlp.mlp", "mlp")
        .rename("mesh_decoder.cases", "cases")
        .rename("mesh_decoder.masks", "masks")
        // The background of rendered views, which meshes do not read.
        .skip("void.background")
}
