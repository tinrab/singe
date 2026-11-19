// Reads a `transformers` ModernVBert snapshot that `sentence-transformers` serves as a multi-vector encoder: its config, the template it renders inputs through, and its tensors' names.

use std::fs;

use serde::Deserialize;
use singe::nn::Activation;

use crate::{
    error::{Error, Result},
    format::{
        Rules, join,
        transformers::{LayerType, activation_field},
    },
    model::multimodal::{
        colmodernvbert::{ColModernVBertConfig, ModernBertConfig},
        smolvlm::transformers::{
            VisionFields, processor, vision_config, vision_rules as tower_rules,
        },
    },
    source::Source,
};

#[derive(Debug, Deserialize)]
struct TextFields {
    vocab_size: i64,
    hidden_size: i64,
    intermediate_size: i64,
    num_attention_heads: i64,
    norm_eps: f32,
    #[serde(deserialize_with = "activation_field")]
    hidden_activation: Activation,
    layer_types: Vec<LayerType>,
    local_attention: i64,
    global_rope_theta: f64,
    local_rope_theta: f64,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
    #[serde(default)]
    norm_bias: bool,
}

#[derive(Debug, Deserialize)]
struct Fields {
    vision_config: VisionFields,
    text_config: TextFields,
    pixel_shuffle_factor: i64,
    image_token_id: i64,
}

#[derive(Debug, Deserialize)]
struct DenseFields {
    out_features: i64,
}

#[derive(Debug, Deserialize)]
struct TemplateChoice {
    chat_template: String,
}

#[derive(Debug, Deserialize)]
struct ProcessingFields {
    chat_template: TemplateChoice,
}

#[derive(Debug, Deserialize)]
struct SentenceFields {
    processing_kwargs: ProcessingFields,
}

pub(crate) fn config(source: &Source) -> Result<ColModernVBertConfig> {
    let fields: Fields = source.json("config.json")?;
    let text = &fields.text_config;
    if text.attention_bias || text.mlp_bias || text.norm_bias {
        return Err(Error::unsupported("a ModernBERT with biases"));
    }

    let local = text
        .layer_types
        .iter()
        .map(|layer| match layer {
            LayerType::FullAttention => Ok(false),
            LayerType::SlidingAttention => Ok(true),
            LayerType::LinearAttention => Err(Error::unsupported("a linear attention layer")),
        })
        .collect::<Result<Vec<_>>>()?;
    let dense: DenseFields = source.json("1_Dense/config.json")?;
    let sentence: SentenceFields = source.json("sentence_bert_config.json")?;
    let name = sentence.processing_kwargs.chat_template.chat_template;
    let path = source
        .directory()
        .join(format!("additional_chat_templates/{name}.jinja"));
    let template = fs::read_to_string(&path).map_err(Error::io(&path))?;
    Ok(ColModernVBertConfig {
        vision: vision_config(&fields.vision_config, fields.pixel_shuffle_factor),
        text: ModernBertConfig {
            vocab_size: text.vocab_size,
            width: text.hidden_size,
            inner: text.intermediate_size,
            heads: text.num_attention_heads,
            eps: text.norm_eps,
            activation: text.hidden_activation,
            local,
            // `local_attention` spans both sides.
            window: text.local_attention / 2,
            global_theta: text.global_rope_theta,
            local_theta: text.local_rope_theta,
        },
        image_token_id: fields.image_token_id,
        embedding: dense.out_features,
        processor: processor(source)?,
        template,
    })
}

// Imports the SigLIP tower and the connector; the tower's pooling head is unused, since the encoder reads every patch.
pub(crate) fn vision_rules() -> Rules {
    Rules::new()
        .skip("model.vision_model.vision_model.head")
        .then(tower_rules(
            "model.vision_model.vision_model",
            "model.connector.modality_projection",
        ))
}

// Imports ModernBERT, splitting its fused projections, and the projection to each token's embedding, which `sentence-transformers`' `1_Dense` holds again.
pub(crate) fn text_rules() -> Rules {
    let text = |name: &str| join("model.text_model", name);
    let layer = |name: &str| join("model.text_model.layers.*", name);
    Rules::new()
        .rename(&text("embeddings.tok_embeddings"), "tokens")
        .rename(&text("embeddings.norm"), "embedding_norm")
        .rename(&layer("attn_norm"), "blocks.*.attention_norm")
        .split(
            &layer("attn.Wqkv"),
            0,
            &[
                "blocks.*.attention.query",
                "blocks.*.attention.key",
                "blocks.*.attention.value",
            ],
        )
        .rename(&layer("attn.Wo"), "blocks.*.attention.output")
        .rename(&layer("mlp_norm"), "blocks.*.feed_forward_norm")
        // The first half is activated, and the second gates it.
        .split(
            &layer("mlp.Wi"),
            0,
            &["blocks.*.feed_forward.gate", "blocks.*.feed_forward.up"],
        )
        .rename(&layer("mlp.Wo"), "blocks.*.feed_forward.down")
        .rename(&text("final_norm"), "norm")
        .rename("custom_text_proj", "projection")
}
