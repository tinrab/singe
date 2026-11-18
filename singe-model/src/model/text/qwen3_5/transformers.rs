// Reads a `transformers` Qwen3.5 config and imports its checkpoint.
//
// A model that reads only text keeps its decoder under `model`; one that also reads images keeps it under `model.language_model`, beside the vision encoder under `model.visual`.
// Either keeps its multi-token-prediction layer under `mtp`.

use serde::{Deserialize, de::DeserializeOwned};
use serde_json::Value;
use singe::nn::{Activation, GatedDeltaNetSpec};

use super::{
    ImageSizeBounds, Qwen35Config, Qwen35ExpertsConfig, Qwen35ImageProcessorConfig, Qwen35Mixer,
    Qwen35TextConfig, Qwen35VisionConfig, Qwen35VisionLanguageConfig,
};
use crate::{
    assembly::{DecoderConfig, expert_rules, llama_rules},
    error::{Error, Result},
    format::{
        Rules,
        transformers::{DecoderDefaults, DecoderFields, activation_field},
    },
    source::Source,
};

// The text decoder's fields: the top level of a model that reads only text, or `text_config` of one that also reads images.
#[derive(Debug, Clone, PartialEq, Deserialize)]
struct TextFields {
    #[serde(flatten)]
    decoder: DecoderFields,
    linear_conv_kernel_dim: i64,
    linear_key_head_dim: i64,
    linear_num_key_heads: i64,
    linear_value_head_dim: i64,
    linear_num_value_heads: i64,
    // Every fourth layer is full attention when the config names no layer types.
    #[serde(default = "default_full_attention_interval")]
    full_attention_interval: usize,
    #[serde(default = "default_output_gate")]
    attn_output_gate: bool,
    // Present in a mixture-of-experts model, every block of which has experts.
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

const fn default_output_gate() -> bool {
    true
}

// The defaults `transformers`' `Qwen3_5MoeTextConfig` gives its experts.
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

        let layer_types = self.decoder.layer_types.as_ref();
        if layer_types
            .into_iter()
            .flatten()
            .any(|layer| layer != "full_attention" && layer != "linear_attention")
        {
            return Err(Error::unsupported(
                "a layer that is neither full nor linear attention",
            ));
        }

        let decoder = DecoderConfig {
            head_norms: true,
            ..self.decoder.config(DecoderDefaults {
                tied: false,
                head_dim: Some(256),
            })?
        };
        let layers = usize::try_from(decoder.layers)
            .map_err(|_| Error::unsupported("a negative number of layers"))?;
        let mixers = (0..layers)
            .map(|index| {
                let full = layer_types.map_or_else(
                    || (index + 1).is_multiple_of(self.full_attention_interval.max(1)),
                    |types| {
                        types
                            .get(index)
                            .is_some_and(|layer| layer == "full_attention")
                    },
                );
                if full {
                    Qwen35Mixer::Attention
                } else {
                    Qwen35Mixer::DeltaNet
                }
            })
            .collect();
        let experts = self.num_experts.map(|experts| Qwen35ExpertsConfig {
            experts,
            top_k: self.num_experts_per_tok,
            inner: self.moe_intermediate_size,
            shared_inner: self.shared_expert_intermediate_size,
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
            positions: self.num_position_embeddings,
            out_width: self.out_hidden_size,
            patch_size: self.patch_size,
            merge_size: self.spatial_merge_size,
            frames: self.temporal_patch_size,
            channels: self.in_channels,
        })
    }
}

// The fields of a `config.json` of a model that reads images.
#[derive(Debug, Clone, PartialEq, Deserialize)]
struct VisionLanguageFields {
    text_config: Value,
    vision_config: VisionFields,
    image_token_id: i64,
    // Whether the output head shares the embedding's weight, which `transformers` reads here and not from `text_config`.
    #[serde(default)]
    tie_word_embeddings: bool,
}

// What the image front end reads from `preprocessor_config.json`.
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

fn parse<T: DeserializeOwned>(value: Value) -> Result<T> {
    serde_json::from_value(value).map_err(Error::ConfigFormat)
}

// Reads `config.json`, and for a model that reads images, `preprocessor_config.json`.
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
    let mut text = parse::<TextFields>(fields.text_config.clone())?.config()?;
    text.decoder.tied = fields.tie_word_embeddings;
    Ok(Qwen35Config {
        text,
        vision: Some(Qwen35VisionLanguageConfig {
            encoder: fields.vision_config.config()?,
            image_token_id: fields.image_token_id,
            position_sections: sections(&fields.text_config)?,
            processor: Qwen35ImageProcessorConfig {
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

// Reads `text_config.rope_parameters.mrope_section`, the frequencies each of M-RoPE's axes takes.
fn sections(text: &Value) -> Result<[i64; 3]> {
    let parameters = text
        .get("rope_parameters")
        .or_else(|| text.get("rope_scaling"))
        .ok_or_else(|| Error::unsupported("a text config without rope parameters"))?;
    if parameters.get("mrope_interleaved").and_then(Value::as_bool) != Some(true) {
        return Err(Error::unsupported(
            "M-RoPE whose sections are not interleaved",
        ));
    }

    parse(
        parameters
            .get("mrope_section")
            .cloned()
            .ok_or_else(|| Error::unsupported("rope parameters without `mrope_section`"))?,
    )
}

// A gated delta net's stored names, and Singe's.
const DELTA_NET: [(&str, &str); 9] = [
    ("in_proj_qkv", "qkv"),
    ("in_proj_z", "gate"),
    ("in_proj_b", "strength"),
    ("in_proj_a", "decay"),
    ("conv1d", "convolution"),
    ("A_log", "decay_log"),
    ("dt_bias", "decay_bias"),
    ("norm", "norm"),
    ("out_proj", "output"),
];

// A block of full attention and a dense feed-forward layer's stored names, and Singe's.
const ATTENTION_BLOCK: [(&str, &str); 11] = [
    ("input_layernorm", "attention_norm"),
    ("self_attn.q_proj", "attention.query"),
    ("self_attn.k_proj", "attention.key"),
    ("self_attn.v_proj", "attention.value"),
    ("self_attn.o_proj", "attention.output"),
    ("self_attn.q_norm", "attention.query_norm"),
    ("self_attn.k_norm", "attention.key_norm"),
    ("post_attention_layernorm", "feed_forward_norm"),
    ("mlp.gate_proj", "feed_forward.gate"),
    ("mlp.up_proj", "feed_forward.up"),
    ("mlp.down_proj", "feed_forward.down"),
];

// Imports the text decoder, and its draft layer if `drafting`; `images` says whether the vision encoder loads too, as a part of its own.
pub(crate) fn decoder_rules(config: &Qwen35Config, drafting: bool, images: bool) -> Rules {
    let model = if config.vision.is_some() {
        "model.language_model"
    } else {
        "model"
    };
    let layer = |name: &str| format!("{model}.layers.*.{name}");
    let mut rules = Rules::new();
    if !images {
        // Decoding text reads no image.
        rules = rules.skip("model.visual");
    }

    rules = if drafting {
        rules.then(draft_rules())
    } else {
        // The multi-token-prediction layer only drafts tokens, which this load does not.
        rules.skip("mtp")
    };
    rules = DELTA_NET.iter().fold(rules, |rules, (from, to)| {
        rules.rename(
            &layer(&format!("linear_attn.{from}")),
            &format!("blocks.*.delta_net.{to}"),
        )
    });
    if let Some(experts) = &config.text.experts {
        rules = rules
            .rename(&layer("mlp.gate"), "blocks.*.feed_forward.router")
            .rename(
                &layer("mlp.shared_expert_gate"),
                "blocks.*.feed_forward.shared_gate",
            )
            .rename(
                &layer("mlp.shared_expert.gate_proj"),
                "blocks.*.feed_forward.shared.gate",
            )
            .rename(
                &layer("mlp.shared_expert.up_proj"),
                "blocks.*.feed_forward.shared.up",
            )
            .rename(
                &layer("mlp.shared_expert.down_proj"),
                "blocks.*.feed_forward.shared.down",
            )
            .then(expert_rules(model, experts.inner));
    }
    rules.then(llama_rules(model))
}

// Imports the one multi-token-prediction layer into `draft`.
fn draft_rules() -> Rules {
    let rules = Rules::new()
        .rename("mtp.pre_fc_norm_embedding", "draft.embedding_norm")
        .rename("mtp.pre_fc_norm_hidden", "draft.hidden_norm")
        .rename("mtp.fc", "draft.projection")
        .rename("mtp.norm", "draft.norm");
    ATTENTION_BLOCK.iter().fold(rules, |rules, (from, to)| {
        rules.rename(
            &format!("mtp.layers.0.{from}"),
            &format!("draft.block.{to}"),
        )
    })
}

// Imports the vision encoder.
pub(crate) fn vision_rules() -> Rules {
    let block = |name: &str| format!("model.visual.blocks.*.{name}");
    Rules::new()
        .rename("model.visual.patch_embed.proj", "embeddings.patches")
        .rename("model.visual.pos_embed", "embeddings.positions")
        .rename(&block("norm1"), "blocks.*.attention_norm")
        // One weight and one bias hold every head's queries, then keys, then values.
        .split(
            &block("attn.qkv"),
            0,
            &[
                "blocks.*.attention.query",
                "blocks.*.attention.key",
                "blocks.*.attention.value",
            ],
        )
        .rename(&block("attn.proj"), "blocks.*.attention.output")
        .rename(&block("norm2"), "blocks.*.feed_forward_norm")
        .rename(&block("mlp.linear_fc1"), "blocks.*.feed_forward.up")
        .rename(&block("mlp.linear_fc2"), "blocks.*.feed_forward.down")
        .rename("model.visual.merger.norm", "merger.norm")
        .rename("model.visual.merger.linear_fc1", "merger.up")
        .rename("model.visual.merger.linear_fc2", "merger.down")
}
