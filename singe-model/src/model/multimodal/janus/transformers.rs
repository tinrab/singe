//! Reads a `transformers` Janus snapshot: its config, its processor's and generation config's settings, and its tensors' names.

use serde::Deserialize;
use singe::nn::Activation;

use crate::{
    assembly::{PreNormEncoderConfig, VisionTowerConfig, clip_layer_rules, llama_rules},
    error::{Error, Result},
    format::{
        Rules, join,
        transformers::{DecoderDefaults, DecoderFields, activation_field, default_channels},
    },
    head::SamplingFields,
    model::multimodal::janus::{
        JanusConfig, JanusGenerationConfig, JanusPrompt, JanusVisionConfig, vq::JanusVqConfig,
    },
    source::Source,
};

#[derive(Debug, Deserialize)]
struct VisionFields {
    hidden_size: i64,
    num_attention_heads: i64,
    num_hidden_layers: i64,
    mlp_ratio: f64,
    image_size: i64,
    patch_size: i64,
    #[serde(default = "default_channels")]
    num_channels: i64,
    layer_norm_eps: f32,
    #[serde(deserialize_with = "activation_field")]
    hidden_act: Activation,
    attention_bias: bool,
    #[serde(default)]
    use_qk_norm: bool,
    depth: i64,
    projection_dim: i64,
    num_image_tokens: usize,
}

#[derive(Debug, Deserialize)]
struct VqFields {
    num_embeddings: i64,
    embed_dim: i64,
    latent_channels: i64,
    base_channels: i64,
    channel_multiplier: Vec<i64>,
    num_res_blocks: i64,
    num_patches: i64,
    out_channels: i64,
    projection_dim: i64,
    image_token_embed_dim: i64,
    num_hidden_layers: i64,
    #[serde(deserialize_with = "activation_field")]
    hidden_act: Activation,
}

#[derive(Debug, Deserialize)]
struct Fields {
    vision_config: VisionFields,
    text_config: DecoderFields,
    vq_config: VqFields,
    image_token_id: i64,
    tie_word_embeddings: Option<bool>,
}

/// `JanusProcessor`'s settings, with its defaults when a snapshot leaves them out.
#[derive(Debug, Deserialize)]
#[serde(default)]
struct ProcessorFields {
    use_default_system_prompt: bool,
    num_image_tokens: usize,
}

impl Default for ProcessorFields {
    fn default() -> Self {
        Self {
            use_default_system_prompt: false,
            num_image_tokens: 576,
        }
    }
}

#[derive(Debug, Deserialize)]
struct GenerationFields {
    bos_token_id: i64,
    pad_token_id: i64,
    #[serde(default = "guidance_scale")]
    guidance_scale: f32,
    #[serde(flatten)]
    sampling: SamplingFields,
}

/// `JanusForConditionalGeneration.generate`'s default when the generation config names none.
const fn guidance_scale() -> f32 {
    5.0
}

#[derive(Debug, Deserialize)]
struct TokenFields {
    #[serde(rename = "image_token")]
    image: String,
    #[serde(rename = "boi_token")]
    image_start: String,
    #[serde(rename = "eoi_token")]
    image_end: String,
}

pub(crate) fn config(source: &Source) -> Result<JanusConfig> {
    let fields: Fields = source.json("config.json")?;
    let vision = fields.vision_config;
    if vision.use_qk_norm {
        return Err(Error::unsupported(
            "a Janus vision tower that norms its queries and keys",
        ));
    }

    let vq = fields.vq_config;
    let processor = source
        .optional_json::<ProcessorFields>("processor_config.json")?
        .unwrap_or_default();
    let generation: GenerationFields = source.json("generation_config.json")?;
    let tokens: TokenFields = source.json("tokenizer_config.json")?;
    let text = fields.text_config.config(DecoderDefaults {
        tied: fields.tie_word_embeddings.unwrap_or(false),
        head_dim: None,
    })?;
    // `JanusVQVAEAlignerMLP` gives `projection_dim` features, which the decoder reads as its own.
    if vq.image_token_embed_dim != text.width || vq.projection_dim != text.width {
        return Err(Error::unsupported(format!(
            "Janus image token embeddings {} wide and aligned to {} for a decoder {} wide",
            vq.image_token_embed_dim, vq.projection_dim, text.width
        )));
    }

    Ok(JanusConfig {
        vision: JanusVisionConfig {
            tower: VisionTowerConfig {
                encoder: PreNormEncoderConfig {
                    width: vision.hidden_size,
                    heads: vision.num_attention_heads,
                    inner: (vision.hidden_size as f64 * vision.mlp_ratio) as i64,
                    activation: vision.hidden_act,
                    eps: vision.layer_norm_eps,
                    projection_biases: [
                        vision.attention_bias,
                        vision.attention_bias,
                        vision.attention_bias,
                        true,
                    ],
                    causal: false,
                    layer_scale: false,
                },
                layers: vision.num_hidden_layers,
                image_size: vision.image_size,
                patch_size: vision.patch_size,
                channels: vision.num_channels,
            },
            aligner_depth: vision.depth,
            aligner_width: vision.projection_dim,
            activation: vision.hidden_act,
        },
        text,
        vq: JanusVqConfig {
            codebook: vq.num_embeddings,
            entry: vq.embed_dim,
            latent: vq.latent_channels,
            base: vq.base_channels,
            multipliers: vq.channel_multiplier,
            blocks: vq.num_res_blocks,
            side: vq.num_patches,
            channels: vq.out_channels,
        },
        generation: JanusGenerationConfig {
            projection: vq.projection_dim,
            aligner_depth: vq.num_hidden_layers,
            activation: vq.hidden_act,
            tokens: vision.num_image_tokens,
            guidance_scale: generation.guidance_scale,
            sampling: generation.sampling.sampling(),
            bos_token_id: generation.bos_token_id,
            pad_token_id: generation.pad_token_id,
        },
        image_token_id: fields.image_token_id,
        prompt: JanusPrompt {
            system_prompt: processor.use_default_system_prompt,
            image_tokens: processor.num_image_tokens,
            image_token: tokens.image,
            boi_token: tokens.image_start,
            eoi_token: tokens.image_end,
        },
    })
}

/// Imports `from`'s `fc1` and `hidden_layers.N` as `LinearStack` layers `0` and `N + 1` under `to`.
fn aligner_rules(from: &str, to: &str, depth: i64) -> Rules {
    let mut rules = Rules::new().rename(&join(from, "fc1"), &join(to, "0"));
    for index in 0..depth - 1 {
        rules = rules.rename(
            &join(from, &format!("hidden_layers.{index}")),
            &join(to, &(index + 1).to_string()),
        );
    }
    rules
}

/// The understanding part's vision tower and aligner, skipping what only image generation reads.
pub(crate) fn vision_rules(config: &JanusConfig) -> Rules {
    let vision = |name: &str| join("model.vision_model", name);
    let mut rules = Rules::new()
        .rename(&vision("embeddings.patch_embedding"), "embeddings.patches")
        .derive(
            &vision("embeddings.position_embedding.weight"),
            "embeddings.positions",
            |positions| positions.unsqueeze(0),
        )
        .rename_each(
            &vision("encoder.layers.*"),
            "blocks.*",
            &[("self_attn.projection_layer", "attention.output")],
        )
        .then(clip_layer_rules(&vision("encoder.layers"), "blocks"))
        .rename(&vision("post_layernorm"), "norm")
        .then(aligner_rules(
            "model.aligner",
            "aligner",
            config.vision.aligner_depth,
        ));
    for unread in [
        "model.vqmodel",
        "model.generation_embeddings",
        "model.generation_aligner",
        "model.generation_head",
    ] {
        rules = rules.skip(unread);
    }
    rules
}

pub(crate) fn decoder_rules() -> Rules {
    llama_rules("model.language_model")
}

/// The image generation part: the decoder with the generation embeddings, aligner, and head, skipping what only understanding and image encoding read.
pub(crate) fn generation_rules(config: &JanusConfig) -> Rules {
    // Skipped first, since `llama_rules` would take `lm_head` as the head.
    let mut rules = Rules::new();
    for unread in [
        "model.vision_model",
        "model.aligner",
        "lm_head",
        "model.vqmodel",
    ] {
        rules = rules.skip(unread);
    }
    rules
        .rename("model.generation_embeddings", "generation_embeddings")
        .then(aligner_rules(
            "model.generation_aligner",
            "generation_aligner",
            config.generation.aligner_depth,
        ))
        .rename("model.generation_head.proj_out", "norm.projection")
        .rename("model.generation_head.vision_head", "head")
        .rename("model.language_model.norm", "norm.norm")
        .then(llama_rules("model.language_model"))
}

/// Imports `JanusVQVAEAttnBlock` at `from` as a `SpatialAttention` at `to`, its 1 × 1 convolutions as linear layers.
fn spatial_attention_rules(from: &str, to: &str) -> Rules {
    let mut rules = Rules::new().rename(&join(from, "norm"), &join(to, "norm"));
    for (convolution, linear) in [
        ("q", "query"),
        ("k", "key"),
        ("v", "value"),
        ("proj_out", "output"),
    ] {
        let (from, to) = (join(from, convolution), join(to, linear));
        rules = rules
            .derive(&join(&from, "weight"), &join(&to, "weight"), |weight| {
                weight.flatten(1, 3)
            })
            .rename(&join(&from, "bias"), &join(&to, "bias"));
    }
    rules
}

/// The VQ decoder part, skipping the VQ encoder and everything else.
pub(crate) fn vq_rules() -> Rules {
    let decoder = |name: &str| join("model.vqmodel.decoder", name);
    let resnet = [
        ("norm1", "norm1"),
        ("conv1", "conv1"),
        ("norm2", "norm2"),
        ("conv2", "conv2"),
        ("nin_shortcut", "shortcut"),
    ];
    let mut rules = Rules::new()
        .rename("model.vqmodel.quantize.embedding.weight", "codebook")
        .rename("model.vqmodel.post_quant_conv", "post_quant")
        .then(spatial_attention_rules(
            &decoder("mid.attn_1"),
            "middle.attentions.0",
        ))
        .then(spatial_attention_rules(
            &decoder("up.*.attn.*"),
            "levels.*.attentions.*",
        ))
        .rename_each(&decoder("mid.block_1"), "middle.resnets.0", &resnet)
        .rename_each(&decoder("mid.block_2"), "middle.resnets.1", &resnet)
        .rename_each(&decoder("up.*.block.*"), "levels.*.blocks.*", &resnet)
        .rename(&decoder("up.*.upsample.conv"), "levels.*.upsample")
        .rename_each(
            "model.vqmodel.decoder",
            "",
            &[
                ("conv_in", "conv_in"),
                ("norm_out", "norm_out"),
                ("conv_out", "conv_out"),
            ],
        );
    for unread in [
        "model.vqmodel.encoder",
        "model.vqmodel.quant_conv",
        "model.vision_model",
        "model.aligner",
        "model.language_model",
        "model.generation_embeddings",
        "model.generation_aligner",
        "model.generation_head",
        "lm_head",
    ] {
        rules = rules.skip(unread);
    }
    rules
}
