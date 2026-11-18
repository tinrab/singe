// Reads a `transformers` DETR config, whose backbone is timm's ResNet-50, and imports its checkpoint.

use serde::Deserialize;
use singe::nn::Activation;

use super::{DetrConfig, ResNetConfig};
use crate::{
    error::{Error, Result},
    format::{
        Rules,
        transformers::{activation_field, classifier_config},
    },
    source::Source,
};

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Fields {
    backbone: Option<String>,
    #[serde(default)]
    dilation: bool,
    d_model: i64,
    encoder_layers: i64,
    decoder_layers: i64,
    encoder_attention_heads: i64,
    decoder_attention_heads: i64,
    encoder_ffn_dim: i64,
    decoder_ffn_dim: i64,
    #[serde(deserialize_with = "activation_field")]
    activation_function: Activation,
    num_queries: i64,
    position_embedding_type: String,
}

pub(crate) fn config(source: &Source) -> Result<DetrConfig> {
    let fields: Fields = source.json("config.json")?;
    // timm's ResNet-50 without dilation, and sine positions, as `facebook/detr-resnet-50` has them.
    if fields.backbone.as_deref() != Some("resnet50")
        || fields.dilation
        || fields.position_embedding_type != "sine"
    {
        return Err(Error::unsupported("this DETR backbone"));
    }

    Ok(DetrConfig {
        backbone: ResNetConfig {
            depths: vec![3, 4, 6, 3],
            stem: 64,
        },
        width: fields.d_model,
        encoder_layers: fields.encoder_layers,
        decoder_layers: fields.decoder_layers,
        encoder_heads: fields.encoder_attention_heads,
        decoder_heads: fields.decoder_attention_heads,
        encoder_inner: fields.encoder_ffn_dim,
        decoder_inner: fields.decoder_ffn_dim,
        activation: fields.activation_function,
        queries: fields.num_queries,
        classifier: classifier_config(source)?,
    })
}

pub(crate) fn rules() -> Rules {
    let mut rules = Rules::new()
        // Counts training updates, which evaluation never reads.
        .skip("model.backbone.conv_encoder.model.*.*.downsample.1.num_batches_tracked")
        .rename("model.backbone.conv_encoder.model", "backbone")
        .rename("model.input_projection", "projection")
        .rename("model.query_position_embeddings.weight", "queries")
        .rename("model.decoder.layernorm", "decoder_norm")
        .rename("class_labels_classifier", "classes")
        .rename("bbox_predictor.layers", "boxes");
    for part in ["encoder", "decoder"] {
        let layer = |name: &str| format!("model.{part}.layers.*.{name}");
        let target = |name: &str| format!("{part}.*.{name}");
        for (from, to) in [
            ("self_attn", "attention"),
            ("encoder_attn", "cross_attention"),
        ] {
            for (projection, name) in [
                ("q_proj", "query"),
                ("k_proj", "key"),
                ("v_proj", "value"),
                ("out_proj", "output"),
            ] {
                rules = rules.rename(
                    &layer(&format!("{from}.{projection}")),
                    &target(&format!("{to}.{name}")),
                );
            }
        }

        rules = rules
            .rename(&layer("self_attn_layer_norm"), &target("attention_norm"))
            .rename(
                &layer("encoder_attn_layer_norm"),
                &target("cross_attention_norm"),
            )
            .rename(&layer("fc1"), &target("feed_forward.up"))
            .rename(&layer("fc2"), &target("feed_forward.down"))
            .rename(&layer("final_layer_norm"), &target("feed_forward_norm"));
    }
    rules
}
