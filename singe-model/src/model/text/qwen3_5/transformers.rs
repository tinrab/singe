//! Reads a `transformers` Qwen3.5 config and imports its checkpoint.
//!
//! A model that reads only text keeps its decoder under `model`; one that also reads images keeps it under `model.language_model`, beside the vision encoder under `model.visual`.
//! Either keeps its multi-token-prediction layer under `mtp`.

use serde::Deserialize;
use serde_json::Value;
use singe::nn::{Activation, GatedDeltaNetSpec};

use crate::{
    assembly::{
        DecoderConfig, ExpertRouting, ExpertsConfig, expert_rules, llama_block_rules, llama_rules,
    },
    error::{Error, Result},
    format::{
        Rules, join,
        transformers::{
            DecoderDefaults, DecoderFields, LayerType, RotaryFields, activation_field,
            default_true, parse,
        },
    },
    model::text::qwen3_5::{
        Draft, ImageSizeBounds, Qwen35Config, Qwen35ImageProcessor, Qwen35Mixer, Qwen35TextConfig,
        Qwen35VisionConfig, Qwen35VisionLanguageConfig,
    },
    source::Source,
};

/// The text decoder's fields: the top level of a model that reads only text, or `text_config` of one that also reads images.
/// Expert settings other than `num_experts` that it leaves out take the defaults of `transformers`' `Qwen3_5MoeTextConfig`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
struct TextFields {
    #[serde(flatten)]
    decoder: DecoderFields,
    linear_conv_kernel_dim: i64,
    linear_key_head_dim: i64,
    linear_num_key_heads: i64,
    linear_value_head_dim: i64,
    linear_num_value_heads: i64,
    /// How often a layer is full attention when the config names no layer types, every fourth by default.
    #[serde(default = "default_full_attention_interval")]
    full_attention_interval: usize,
    #[serde(default = "default_true")]
    attn_output_gate: bool,
    /// Present in a mixture-of-experts model, every block of which has experts.
    #[serde(default)]
    num_experts: Option<i64>,
    #[serde(default = "default_experts_per_token")]
    num_experts_per_tok: i64,
    #[serde(default = "default_expert_inner")]
    moe_intermediate_size: i64,
    #[serde(default = "default_expert_inner")]
    shared_expert_intermediate_size: i64,
}

const fn default_full_attention_interval() -> usize {
    4
}

const fn default_experts_per_token() -> i64 {
    8
}

const fn default_expert_inner() -> i64 {
    512
}

impl TextFields {
    fn config(&self) -> Result<Qwen35TextConfig> {
        // A checkpoint using one of these would load and then compute something else.
        if !self.attn_output_gate {
            return Err(Error::unsupported("full attention without an output gate"));
        }

        let decoder = DecoderConfig {
            head_norms: true,
            ..self.decoder.config(DecoderDefaults {
                tied: false,
                head_dim: Some(256),
            })?
        };
        let interval = self.full_attention_interval.max(1);
        let mixers = self
            .decoder
            .layer_types(|index| {
                if (index + 1).is_multiple_of(interval) {
                    LayerType::FullAttention
                } else {
                    LayerType::LinearAttention
                }
            })?
            .into_iter()
            .map(|layer| match layer {
                LayerType::FullAttention => Ok(Qwen35Mixer::Attention),
                LayerType::LinearAttention => Ok(Qwen35Mixer::DeltaNet),
                LayerType::SlidingAttention => Err(Error::unsupported("a sliding attention layer")),
            })
            .collect::<Result<Vec<_>>>()?;
        let experts = self.num_experts.map(|count| ExpertsConfig {
            count,
            top_k: self.num_experts_per_tok,
            inner: self.moe_intermediate_size,
            shared_inner: self.shared_expert_intermediate_size,
            routing: ExpertRouting::Softmax,
        });
        Ok(Qwen35TextConfig {
            decoder,
            mixers,
            delta_net: GatedDeltaNetSpec {
                key_heads: self.linear_num_key_heads,
                key_dim: self.linear_key_head_dim,
                value_heads: self.linear_num_value_heads,
                value_dim: self.linear_value_head_dim,
                kernel: self.linear_conv_kernel_dim,
            },
            experts,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct VisionFields {
    depth: i64,
    hidden_size: i64,
    #[serde(deserialize_with = "activation_field")]
    hidden_act: Activation,
    intermediate_size: i64,
    num_heads: i64,
    num_position_embeddings: i64,
    out_hidden_size: i64,
    patch_size: i64,
    spatial_merge_size: i64,
    temporal_patch_size: i64,
    in_channels: i64,
    #[serde(default)]
    deepstack_visual_indexes: Vec<usize>,
}

impl VisionFields {
    fn config(&self) -> Result<Qwen35VisionConfig> {
        if !self.deepstack_visual_indexes.is_empty() {
            return Err(Error::unsupported("deepstack visual features"));
        }

        Ok(Qwen35VisionConfig {
            width: self.hidden_size,
            layers: self.depth,
            heads: self.num_heads,
            inner: self.intermediate_size,
            activation: self.hidden_act,
            max_positions: self.num_position_embeddings,
            out_width: self.out_hidden_size,
            patch_size: self.patch_size,
            merge_size: self.spatial_merge_size,
            frames: self.temporal_patch_size,
            channels: self.in_channels,
        })
    }
}

/// The fields of a `config.json` of a model that reads images.
#[derive(Debug, Clone, PartialEq, Deserialize)]
struct VisionLanguageFields {
    text_config: TextFields,
    vision_config: VisionFields,
    image_token_id: i64,
    /// Whether the output head shares the embedding's weight, which `transformers` reads here and not from `text_config`.
    #[serde(default)]
    tie_word_embeddings: bool,
}

/// What the image front end reads from `preprocessor_config.json`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
struct ProcessorFields {
    size: SizeFields,
    image_mean: [f32; 3],
    image_std: [f32; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
struct SizeFields {
    shortest_edge: i64,
    longest_edge: i64,
}

/// Reads `config.json`, and for a model that reads images, `preprocessor_config.json`.
pub(crate) fn config(source: &Source) -> Result<Qwen35Config> {
    let file: Value = source.json("config.json")?;
    if file.get("text_config").is_none() {
        return Ok(Qwen35Config {
            text: parse::<TextFields>(file)?.config()?,
            vision: None,
        });
    }

    let fields: VisionLanguageFields = parse(file)?;
    let processor: ProcessorFields = source.json("preprocessor_config.json")?;
    let mut text = fields.text_config.config()?;
    text.decoder.tied = fields.tie_word_embeddings;
    Ok(Qwen35Config {
        text,
        vision: Some(Qwen35VisionLanguageConfig {
            encoder: fields.vision_config.config()?,
            image_token_id: fields.image_token_id,
            position_sections: sections(&fields.text_config.decoder.rotary)?,
            processor: Qwen35ImageProcessor {
                bounds: ImageSizeBounds {
                    min_pixels: processor.size.shortest_edge,
                    max_pixels: processor.size.longest_edge,
                },
                mean: processor.image_mean,
                std: processor.image_std,
            },
        }),
    })
}

fn sections(rotary: &RotaryFields) -> Result<[i64; 3]> {
    if !rotary.mrope_interleaved() {
        return Err(Error::unsupported(
            "M-RoPE whose sections are not interleaved",
        ));
    }

    let section = rotary
        .mrope_section()
        .ok_or_else(|| Error::unsupported("rope parameters without `mrope_section`"))?;
    <[i64; 3]>::try_from(section)
        .map_err(|_| Error::unsupported(format!("an `mrope_section` of {} axes", section.len())))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Vision {
    Loaded,
    Skipped,
}

pub(crate) fn decoder_rules(config: &Qwen35Config, draft: Draft, vision: Vision) -> Rules {
    let model = if config.vision.is_some() {
        "model.language_model"
    } else {
        "model"
    };
    let layers = join(model, "layers.*");
    let rules = match vision {
        Vision::Loaded => Rules::new(),
        // Decoding text reads no image.
        Vision::Skipped => Rules::new().skip("model.visual"),
    };
    let rules = match draft {
        Draft::Included => rules.then(draft_rules()),
        // The multi-token-prediction layer only drafts tokens, which this load does not.
        Draft::Omitted => rules.skip("mtp"),
    };
    let rules = rules.rename_each(
        &join(&layers, "linear_attn"),
        "blocks.*.delta_net",
        &[
            ("in_proj_qkv", "qkv"),
            ("in_proj_z", "gate"),
            ("in_proj_b", "strength"),
            ("in_proj_a", "decay"),
            ("conv1d", "convolution"),
            ("A_log", "decay_log"),
            ("dt_bias", "decay_bias"),
            ("norm", "norm"),
            ("out_proj", "output"),
        ],
    );
    let rules = match &config.text.experts {
        Some(experts) => rules
            .rename_each(
                &join(&layers, "mlp"),
                "blocks.*.feed_forward",
                &[
                    ("gate", "router"),
                    ("shared_expert_gate", "shared_gate"),
                    ("shared_expert.gate_proj", "shared.gate"),
                    ("shared_expert.up_proj", "shared.up"),
                    ("shared_expert.down_proj", "shared.down"),
                ],
            )
            .then(expert_rules(model, experts.inner)),
        None => rules,
    };
    rules.then(llama_rules(model))
}

/// Returns the rules that import the one multi-token-prediction layer into `draft`.
fn draft_rules() -> Rules {
    Rules::new()
        .rename_each(
            "mtp",
            "draft",
            &[
                ("pre_fc_norm_embedding", "embedding_norm"),
                ("pre_fc_norm_hidden", "hidden_norm"),
                ("fc", "projection"),
                ("norm", "norm"),
            ],
        )
        .then(llama_block_rules("mtp.layers.0", "draft.block"))
}

pub(crate) fn vision_rules() -> Rules {
    let blocks = "model.visual.blocks.*";
    Rules::new()
        .rename_each(
            "model.visual",
            "",
            &[
                ("patch_embed.proj", "embeddings.patches"),
                ("pos_embed", "embeddings.positions"),
            ],
        )
        .rename(&join(blocks, "norm1"), "blocks.*.attention_norm")
        // One weight and one bias hold every head's queries, then keys, then values.
        .split(
            &join(blocks, "attn.qkv"),
            0,
            &[
                "blocks.*.attention.query",
                "blocks.*.attention.key",
                "blocks.*.attention.value",
            ],
        )
        .rename_each(
            blocks,
            "blocks.*",
            &[
                ("attn.proj", "attention.output"),
                ("norm2", "feed_forward_norm"),
                ("mlp.linear_fc1", "feed_forward.up"),
                ("mlp.linear_fc2", "feed_forward.down"),
            ],
        )
        .rename_each(
            "model.visual.merger",
            "merger",
            &[
                ("norm", "norm"),
                ("linear_fc1", "up"),
                ("linear_fc2", "down"),
            ],
        )
}
