//! Reads a `transformers` time series transformer or Informer config and imports its checkpoint.

use serde::Deserialize;
use serde_json::Value;
use singe::nn::Activation;

use crate::{
    error::{Error, Result},
    format::{Rules, transformers::activation_field},
    model::time_series::time_series_transformer::TimeSeriesTransformerConfig,
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
    lags_sequence: Vec<i64>,
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
    scaling: Value,
    distribution_output: String,
    #[serde(default)]
    attention_type: Option<String>,
    #[serde(default)]
    sampling_factor: Option<i64>,
    #[serde(default)]
    distil: bool,
}

pub(crate) fn config(source: &Source) -> Result<TimeSeriesTransformerConfig> {
    let fields: Fields = source.json("config.json")?;
    // `scaling` of `true` or `"mean"` picks `TimeSeriesMeanScaler`.
    let is_mean_scaled =
        matches!(&fields.scaling, Value::Bool(true)) || fields.scaling.as_str() == Some("mean");
    if fields.input_size != 1
        || fields.num_dynamic_real_features != 0
        || i64::try_from(fields.cardinality.len()).ok()
            != Some(fields.num_static_categorical_features)
        || !is_mean_scaled
        || fields.distribution_output != "student_t"
    {
        return Err(Error::unsupported(
            "a time series transformer other than a univariate, mean-scaled Student-T forecaster without dynamic real features",
        ));
    }
    Ok(TimeSeriesTransformerConfig {
        width: fields.d_model,
        encoder_heads: fields.encoder_attention_heads,
        decoder_heads: fields.decoder_attention_heads,
        encoder_layers: fields.encoder_layers,
        decoder_layers: fields.decoder_layers,
        encoder_inner: fields.encoder_ffn_dim,
        decoder_inner: fields.decoder_ffn_dim,
        activation: fields.activation_function,
        context: fields.context_length,
        horizon: fields.prediction_length,
        lags: fields.lags_sequence,
        time_features: fields.num_time_features,
        cardinalities: fields.cardinality,
        embedding_dims: fields.embedding_dimension,
        static_reals: fields.num_static_real_features,
        feature_size: fields.feature_size,
        minimum_scale: 1e-10,
        sparse_factor: match fields.attention_type.as_deref() {
            None | Some("full") => None,
            Some("prob") => Some(fields.sampling_factor.unwrap_or(5)),
            Some(other) => {
                return Err(Error::unsupported(format!(
                    "Informer attention type `{other}`"
                )));
            }
        },
        distil: fields.distil,
    })
}

fn stack_rules(stack: &str) -> Rules {
    let (from, to) = (
        format!("model.{stack}.layers.*"),
        format!("{stack}.blocks.*"),
    );
    Rules::new()
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
                ("final_layer_norm", "feed_forward_norm"),
            ],
        )
}

pub(crate) fn rules() -> Rules {
    stack_rules("encoder")
        // A batch norm's count of the batches it saw in training, which evaluation never reads.
        .skip("model.encoder.conv_layers.*.norm.num_batches_tracked")
        .rename_each(
            "model.encoder.conv_layers.*",
            "encoder.distil.*",
            &[("downConv", "convolution"), ("norm", "norm")],
        )
        .then(stack_rules("decoder"))
        .rename("model.embedder.embedders", "categories")
        .rename("parameter_projection.proj", "parameters")
}
