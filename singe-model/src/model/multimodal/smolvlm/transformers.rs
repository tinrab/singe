//! Reads a `transformers` `SmolVLM` snapshot: its config, its processor's settings, and its tensors' names.

use serde::Deserialize;
use serde_json::Value;
use singe::nn::Activation;

use crate::{
    assembly::{PreNormEncoderConfig, VisionTowerConfig, clip_layer_rules, llama_rules},
    error::Result,
    format::{
        Rules, join,
        transformers::{
            DecoderDefaults, DecoderFields, activation_field, default_channels, default_true,
        },
    },
    model::multimodal::smolvlm::{
        SmolVlmConfig, SmolVlmProcessor, SmolVlmVisionConfig, VideoSampling,
    },
    processor::image::Resample,
    source::Source,
};

#[derive(Debug, Deserialize)]
pub(crate) struct VisionFields {
    hidden_size: i64,
    #[serde(default = "inner")]
    intermediate_size: i64,
    #[serde(default = "layers")]
    num_hidden_layers: i64,
    num_attention_heads: i64,
    #[serde(default = "default_channels")]
    num_channels: i64,
    image_size: i64,
    patch_size: i64,
    #[serde(default = "tanh_gelu", deserialize_with = "activation_field")]
    hidden_act: Activation,
    #[serde(default = "eps")]
    layer_norm_eps: f32,
}

const fn inner() -> i64 {
    3072
}

const fn layers() -> i64 {
    12
}

const fn tanh_gelu() -> Activation {
    Activation::Gelu
}

const fn eps() -> f32 {
    1e-6
}

#[derive(Debug, Deserialize)]
struct Fields {
    vision_config: VisionFields,
    /// A Llama config, whose defaults fill what `SmolVLM`'s text configs leave out.
    text_config: DecoderFields,
    scale_factor: i64,
    image_token_id: i64,
    #[serde(default)]
    tie_word_embeddings: bool,
}

#[derive(Debug, Deserialize)]
struct Edge {
    longest_edge: i64,
}

#[derive(Debug, Deserialize)]
struct VideoFields {
    fps: f64,
    max_frames: usize,
    video_size: Edge,
}

#[derive(Debug, Deserialize)]
struct ProcessorFields {
    size: Edge,
    max_image_size: Edge,
    #[serde(default = "default_true")]
    do_image_splitting: bool,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    #[serde(default = "lanczos")]
    resample: i64,
    #[serde(default)]
    video_sampling: Option<VideoFields>,
}

const fn lanczos() -> i64 {
    1
}

#[derive(Debug, Deserialize)]
struct SequenceFields {
    image_seq_len: usize,
}

pub(crate) fn config(source: &Source) -> Result<SmolVlmConfig> {
    let fields: Fields = source.json("config.json")?;
    Ok(SmolVlmConfig {
        vision: vision_config(&fields.vision_config, fields.scale_factor),
        text: fields.text_config.config(DecoderDefaults {
            tied: fields.tie_word_embeddings,
            head_dim: None,
        })?,
        image_token_id: fields.image_token_id,
        processor: processor(source)?,
    })
}

/// Returns the vision tower's config, folding each `scale × scale` square of patches into one row.
pub(crate) const fn vision_config(vision: &VisionFields, scale: i64) -> SmolVlmVisionConfig {
    SmolVlmVisionConfig {
        tower: VisionTowerConfig {
            encoder: PreNormEncoderConfig {
                width: vision.hidden_size,
                heads: vision.num_attention_heads,
                inner: vision.intermediate_size,
                activation: vision.hidden_act,
                eps: vision.layer_norm_eps,
                projection_biases: [true; 4],
                causal: false,
                layer_scale: false,
            },
            layers: vision.num_hidden_layers,
            image_size: vision.image_size,
            patch_size: vision.patch_size,
            channels: vision.num_channels,
        },
        scale,
    }
}

/// Reads the settings of an Idefics3-style processor, which `SmolVLM`'s extends with clips, from `preprocessor_config.json`, `processor_config.json`, and the image tokens in `tokenizer_config.json`.
pub(crate) fn processor(source: &Source) -> Result<SmolVlmProcessor> {
    let processor: ProcessorFields = source.json("preprocessor_config.json")?;
    let resample = Resample::from_pil(processor.resample)?;

    let sequence: SequenceFields = source.json("processor_config.json")?;
    let tokens = source
        .optional_json::<Value>("tokenizer_config.json")?
        .unwrap_or_default();
    let token = |name: &str, default: &str| tokens[name].as_str().unwrap_or(default).to_owned();
    Ok(SmolVlmProcessor {
        longest_edge: processor.size.longest_edge,
        tile: processor.max_image_size.longest_edge,
        splitting: processor.do_image_splitting,
        mean: processor.image_mean,
        std: processor.image_std,
        resample,
        image_tokens: sequence.image_seq_len,
        video: processor.video_sampling.map(|video| VideoSampling {
            fps: video.fps,
            max_frames: video.max_frames,
            edge: video.video_size.longest_edge,
        }),
        image_token: token("image_token", "<image>"),
        fake_image_token: token("fake_image_token", "<fake_token_around_image>"),
        global_image_token: token("global_image_token", "<global-img>"),
    })
}

/// Returns the rules that import the vision tower stored under `prefix` and the connector stored at `connector`, reshaping the learned positions to the layout [`PatchTokens`](singe::nn::PatchTokens) reads.
pub(crate) fn vision_rules(prefix: &str, connector: &str) -> Rules {
    let vision = |name: &str| join(prefix, name);
    Rules::new()
        .rename(&vision("embeddings.patch_embedding"), "embeddings.patches")
        .derive(
            &vision("embeddings.position_embedding.weight"),
            "embeddings.positions",
            |positions| positions.unsqueeze(0),
        )
        .then(clip_layer_rules(&vision("encoder.layers"), "blocks"))
        .rename(&vision("post_layernorm"), "norm")
        .rename(connector, "connector")
}

pub(crate) fn decoder_rules() -> Rules {
    llama_rules("model.text_model")
}
