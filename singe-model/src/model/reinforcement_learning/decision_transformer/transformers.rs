//! Reads a `transformers` Decision Transformer snapshot: its config and its tensors' names, its blocks' as GPT-2's.

use serde::Deserialize;
use singe::nn::Activation;

use crate::{
    assembly::PreNormEncoderConfig,
    error::Result,
    format::{RuleSet, transformers::activation_field},
    model::{
        reinforcement_learning::decision_transformer::DecisionTransformerConfig,
        text::gpt2::transformers::{AttentionFields, model_rules},
    },
    source::Source,
};

#[derive(Debug, Deserialize)]
struct Fields {
    state_dim: i64,
    act_dim: i64,
    hidden_size: i64,
    n_layer: i64,
    n_head: i64,
    #[serde(default)]
    n_inner: Option<i64>,
    #[serde(deserialize_with = "activation_field")]
    activation_function: Activation,
    layer_norm_epsilon: f32,
    max_ep_len: i64,
    n_positions: i64,
    action_tanh: bool,
    #[serde(flatten)]
    attention: AttentionFields,
}

pub(crate) fn config(source: &Source) -> Result<DecisionTransformerConfig> {
    let fields: Fields = source.json("config.json")?;
    fields.attention.refuse_other_scaling()?;
    let width = fields.hidden_size;
    Ok(DecisionTransformerConfig {
        encoder: PreNormEncoderConfig {
            width,
            heads: fields.n_head,
            inner: fields.n_inner.unwrap_or(4 * width),
            activation: fields.activation_function,
            eps: fields.layer_norm_epsilon,
            projection_biases: [true; 4],
            causal: true,
            layer_scale: false,
        },
        layers: fields.n_layer,
        state_dim: fields.state_dim,
        action_dim: fields.act_dim,
        episode: fields.max_ep_len,
        max_positions: fields.n_positions,
        action_tanh: fields.action_tanh,
    })
}

pub(crate) fn rules() -> RuleSet {
    RuleSet::new()
        .rename("embed_return", "embeddings.return")
        .rename("embed_state", "embeddings.state")
        .rename("embed_action", "embeddings.action")
        .rename("embed_timestep", "embeddings.timestep")
        .rename("embed_ln", "embeddings.norm")
        // The steps are embedded, never looked up as tokens.
        .skip("encoder.wte")
        .then(model_rules("encoder"))
        .rename("predict_state", "predict.state")
        .rename("predict_action.0", "predict.action")
        .rename("predict_return", "predict.return")
}
