//! Reads a `transformers` Autoformer config and imports its checkpoint.

use serde::Deserialize;
use singe::nn::Activation;

use crate::{
    error::{Error, Result},
    format::{RuleSet, transformers::activation_field},
    model::time_series::autoformer::AutoformerConfig,
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    d_model: i64,
    encoder_attention_heads: i64,
    decoder_attention_heads: i64,
    encoder_layers: i64,
    decoder_layers: i64,
    encoder_ffn_dim: i64,
    decoder_ffn_dim: i64,
    #[serde(deserialize_with = "activation_field")]
    activation_function: Activation,
    context_length: i64,
    prediction_length: i64,
    label_length: i64,
    lags_sequence: Vec<i64>,
    moving_average: i64,
    autocorrelation_factor: f64,
    num_time_features: i64,
    #[serde(default)]
    cardinality: Vec<i64>,
    #[serde(default)]
    embedding_dimension: Vec<i64>,
    num_static_categorical_features: i64,
    num_static_real_features: i64,
    num_dynamic_real_features: i64,
    feature_size: i64,
    input_size: i64,
    scaling: Option<String>,
    distribution_output: String,
}

pub(crate) fn config(source: &Source) -> Result<AutoformerConfig> {
    let fields: Fields = source.json("config.json")?;
    if fields.encoder_attention_heads != fields.decoder_attention_heads
        || fields.input_size != 1
        || fields.num_static_real_features != 0
        || fields.num_dynamic_real_features != 0
        || i64::try_from(fields.cardinality.len()).ok()
            != Some(fields.num_static_categorical_features)
        || fields.scaling.as_deref() != Some("mean")
        || fields.distribution_output != "student_t"
    {
        return Err(Error::unsupported(
            "an Autoformer other than a univariate, mean-scaled Student-T forecaster with static categories only",
        ));
    }
    Ok(AutoformerConfig {
        width: fields.d_model,
        heads: fields.encoder_attention_heads,
        encoder_layers: fields.encoder_layers,
        decoder_layers: fields.decoder_layers,
        encoder_inner: fields.encoder_ffn_dim,
        decoder_inner: fields.decoder_ffn_dim,
        activation: fields.activation_function,
        context: fields.context_length,
        horizon: fields.prediction_length,
        label: fields.label_length,
        lags: fields.lags_sequence,
        moving_average: fields.moving_average,
        autocorrelation_factor: fields.autocorrelation_factor,
        time_features: fields.num_time_features,
        cardinalities: fields.cardinality,
        embedding_dims: fields.embedding_dimension,
        feature_size: fields.feature_size,
        minimum_scale: 1e-10,
    })
}

fn stack_rules(stack: &str) -> RuleSet {
    let (from, to) = (
        format!("model.{stack}.layers.*"),
        format!("{stack}.blocks.*"),
    );
    RuleSet::new()
        .rename(
            &format!("model.{stack}.value_embedding.value_projection"),
            &format!("{stack}.value"),
        )
        .rename(
            &format!("model.{stack}.embed_positions.weight"),
            &format!("{stack}.positions"),
        )
        .rename(
            &format!("model.{stack}.layernorm_embedding"),
            &format!("{stack}.embedding_norm"),
        )
        .rename_each(
            &from,
            &to,
            &[
                ("self_attn.q_proj", "attention.query"),
                ("self_attn.k_proj", "attention.key"),
                ("self_attn.v_proj", "attention.value"),
                ("self_attn.out_proj", "attention.output"),
                ("encoder_attn.q_proj", "cross_attention.query"),
                ("encoder_attn.k_proj", "cross_attention.key"),
                ("encoder_attn.v_proj", "cross_attention.value"),
                ("encoder_attn.out_proj", "cross_attention.output"),
                ("self_attn_layer_norm", "attention_norm"),
                ("encoder_attn_layer_norm", "cross_attention_norm"),
                ("fc1", "feed_forward.up"),
                ("fc2", "feed_forward.down"),
                ("final_layer_norm.layernorm", "norm"),
                ("trend_projection", "trend"),
            ],
        )
}

pub(crate) fn rules() -> RuleSet {
    stack_rules("encoder")
        .then(stack_rules("decoder"))
        .rename(
            "model.decoder.seasonality_projection",
            "decoder.seasonality",
        )
        .rename("model.embedder.embedders", "categories")
        .rename("parameter_projection.proj", "parameters")
}
