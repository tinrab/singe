// What decoder definitions share: the parts of a causal model under Singe's names, and the rules that import Llama-family checkpoints into them.
//
// A causal model's state is `embeddings.tokens`, `blocks.N`, `norm`, and `head`, and a block's is `attention_norm`, `attention` (`query`, `key`, `value`, `output`, and `query_norm` and `key_norm` when heads are normed), `feed_forward_norm`, and `feed_forward` (`gate`, `up`, `down`).

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Device, Error as SingeError, Result as SingeResult, Tensor,
    generation::{DecoderOptions, DraftingModel},
    nn::{
        Activation, CacheSize, CausalLm, CausalLmParts, DecodeCache, Decoder, DecoderModel,
        DecodingLayer, Embedding, GatedMlp, GroupedQueryAttention, GroupedQueryLayers,
        HeadProjection, KvCache, KvCacheSpec, Layer, Linear, ModelState, MultiTokenPrediction,
        PreNormBlock, RotaryFrequencies, RotaryLayout, RotarySpec,
    },
    tensor::AttentionOptions,
};

use crate::{
    format::{Rules, join},
    schema::{Schema, SchemaNorm},
};

// One place for a decoder's sizes, which its attention, cache, and rotary table all derive from.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "each flag is a separate setting of a model"
)]
pub struct DecoderConfig {
    pub vocab_size: i64,
    pub width: i64,
    pub inner: Option<i64>,
    pub layers: i64,
    pub heads: i64,
    pub kv_heads: i64,
    pub head_dim: i64,
    // The leading features of each head the rotation turns.
    pub rotary_dim: i64,
    pub eps: f32,
    pub activation: Activation,
    pub attention_bias: bool,
    pub mlp_bias: bool,
    pub tied: bool,
    // Whether queries and keys are normed per head, as Qwen3's are.
    pub head_norms: bool,
    pub frequencies: RotaryFrequencies,
    pub rotary_layout: RotaryLayout,
}

impl DecoderConfig {
    #[must_use]
    pub const fn rotary(&self) -> RotarySpec {
        RotarySpec {
            head_dim: self.rotary_dim,
            frequencies: self.frequencies,
            layout: self.rotary_layout,
        }
    }

    #[must_use]
    pub const fn attention_spec(&self) -> GroupedAttentionSpec {
        GroupedAttentionSpec {
            width: self.width,
            heads: self.heads,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
            bias: self.attention_bias,
            head_norms: if self.head_norms {
                HeadNorms::Rms(self.eps)
            } else {
                HeadNorms::Omitted
            },
            output_gate: false,
        }
    }

    #[must_use]
    pub const fn kv_cache(&self) -> KvCacheSpec {
        KvCacheSpec {
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
        }
    }
}

// The decoder most Llama-family models are: grouped-query attention and a gated feed-forward layer.
pub type GroupedDecoder = Decoder<GroupedQueryAttention>;

// Builds that decoder on `Device::Meta`.
pub fn grouped_decoder(
    schema: &mut Schema<'_>,
    config: &DecoderConfig,
    options: AttentionOptions,
) -> SingeResult<GroupedDecoder> {
    causal_lm(
        schema,
        config,
        CausalLmSpec::new(config.rotary(), |_| config.kv_cache()),
        |schema, _| {
            pre_norm_block(schema, config, |schema| {
                grouped_attention(schema, config, options)
            })
        },
    )
}

// Registers `feed_forward`'s `gate`, `up`, and `down`.
pub fn gated_mlp(schema: &mut Schema<'_>, config: &DecoderConfig) -> SingeResult<GatedMlp> {
    let width = config.width;
    let inner = config
        .inner
        .ok_or_else(|| SingeError::missing_input("a dense feed-forward layer's width"))?;
    schema.gated_mlp(
        "feed_forward",
        [width, inner, width],
        config.activation,
        config.mlp_bias,
    )
}

// A block with `attention_norm`, `attention`, `feed_forward_norm`, and a gated `feed_forward`.
pub fn pre_norm_block<A, N: SchemaNorm>(
    schema: &mut Schema<'_>,
    config: &DecoderConfig,
    attention: impl FnOnce(&mut Schema<'_>) -> SingeResult<A>,
) -> SingeResult<PreNormBlock<A, GatedMlp, N>> {
    pre_norm_block_with(schema, config, attention, |schema| {
        gated_mlp(schema, config)
    })
}

// A block as `pre_norm_block` makes, whose feed-forward layer is what `feed_forward` registers.
pub fn pre_norm_block_with<A, F, N: SchemaNorm>(
    schema: &mut Schema<'_>,
    config: &DecoderConfig,
    attention: impl FnOnce(&mut Schema<'_>) -> SingeResult<A>,
    feed_forward: impl FnOnce(&mut Schema<'_>) -> SingeResult<F>,
) -> SingeResult<PreNormBlock<A, F, N>> {
    let attention_norm = N::register(schema, "attention_norm", config.width, config.eps)?;
    let attention = attention(&mut schema.scope("attention"))?;
    let feed_forward_norm = N::register(schema, "feed_forward_norm", config.width, config.eps)?;
    let feed_forward = feed_forward(schema)?;
    Ok(PreNormBlock::new(
        attention_norm,
        attention,
        feed_forward_norm,
        feed_forward,
    ))
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HeadNorms {
    Omitted,
    Rms(f32),
    OffsetRms(f32),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroupedAttentionSpec {
    pub width: i64,
    pub heads: i64,
    pub kv_heads: i64,
    pub head_dim: i64,
    pub bias: bool,
    pub head_norms: HeadNorms,
    pub output_gate: bool,
}

impl GroupedAttentionSpec {
    #[must_use]
    pub const fn multi_head(width: i64, heads: i64, bias: bool) -> Self {
        Self {
            width,
            heads,
            kv_heads: heads,
            head_dim: width / heads,
            bias,
            head_norms: HeadNorms::Omitted,
            output_gate: false,
        }
    }
}

pub fn grouped_attention_layers(
    schema: &mut Schema<'_>,
    spec: &GroupedAttentionSpec,
) -> SingeResult<GroupedQueryLayers> {
    let GroupedAttentionSpec {
        width,
        heads,
        kv_heads,
        head_dim,
        bias,
        head_norms,
        output_gate,
    } = *spec;
    let mut head = |name: &str,
                    norm: Option<&str>,
                    heads: i64,
                    features: i64|
     -> SingeResult<HeadProjection> {
        let linear = schema.linear(name, width, features, bias)?;
        let projection = HeadProjection::new(linear, heads, head_dim);
        Ok(match (norm, head_norms) {
            (Some(norm), HeadNorms::Rms(eps)) => {
                projection.with_norm(schema.rms_norm(norm, head_dim, eps)?)
            }
            (Some(norm), HeadNorms::OffsetRms(eps)) => {
                projection.with_norm(schema.offset_rms_norm(norm, head_dim, eps)?)
            }
            _ => projection,
        })
    };
    let query = if output_gate {
        head("query", Some("query_norm"), heads, 2 * heads * head_dim)?.with_output_gate()
    } else {
        head("query", Some("query_norm"), heads, heads * head_dim)?
    };
    let key = head("key", Some("key_norm"), kv_heads, kv_heads * head_dim)?;
    let value = head("value", None, kv_heads, kv_heads * head_dim)?;
    let output = schema.linear("output", heads * head_dim, width, bias)?;
    let layers = GroupedQueryLayers {
        query,
        key,
        value,
        output,
    };
    schema.join_rows(&layers.siblings())?;
    Ok(layers)
}

pub fn grouped_attention(
    schema: &mut Schema<'_>,
    config: &DecoderConfig,
    options: AttentionOptions,
) -> SingeResult<GroupedQueryAttention> {
    GroupedQueryAttention::create(
        grouped_attention_layers(schema, &config.attention_spec())?,
        options,
    )
}

// Options for the causal attention decoders use.
#[must_use]
pub fn causal() -> AttentionOptions {
    AttentionOptions {
        causal: true,
        ..AttentionOptions::default()
    }
}

pub type DraftLayer<'a, B, N> =
    &'a dyn Fn(&mut Schema<'_>) -> SingeResult<MultiTokenPrediction<B, N>>;

// What a causal model takes beside its blocks: its rotary table, each block's cache by index, a scale of its embeddings, and a draft layer.
pub struct CausalLmSpec<'a, C, B: DecodingLayer, N> {
    pub rotary: RotarySpec,
    pub cache: C,
    pub embedding_scale: Option<f32>,
    pub draft: Option<DraftLayer<'a, B, N>>,
}

impl<C, B: DecodingLayer, N> CausalLmSpec<'_, C, B, N>
where
    C: Fn(usize) -> <B::Cache as DecodeCache>::Spec,
{
    pub const fn new(rotary: RotarySpec, cache: C) -> Self {
        Self {
            rotary,
            cache,
            embedding_scale: None,
            draft: None,
        }
    }
}

// Builds a causal model on `Device::Meta`: `embeddings.tokens`, `blocks.N` from `block`, `norm`, and `head`, tied to the embeddings when the config says so, and `draft` from the spec's draft layer.
pub fn causal_lm<B, N, C>(
    schema: &mut Schema<'_>,
    config: &DecoderConfig,
    spec: CausalLmSpec<'_, C, B, N>,
    block: impl Fn(&mut Schema<'_>, usize) -> SingeResult<B>,
) -> SingeResult<CausalLm<B, N>>
where
    B: DecodingLayer,
    N: SchemaNorm + Layer,
    C: Fn(usize) -> <B::Cache as DecodeCache>::Spec,
{
    let table = [config.vocab_size, config.width];
    let embedding = schema.parameter("embeddings.tokens.weight", table)?;
    let blocks = (0..config.layers)
        .zip(0_usize..)
        .map(|(layer, index)| block(&mut schema.scope(format!("blocks.{layer}")), index))
        .collect::<SingeResult<Vec<_>>>()?;
    let norm = N::register(schema, "norm", config.width, config.eps)?;
    let caches = (0..blocks.len()).map(spec.cache).collect();
    let head = if config.tied {
        schema.tie("head.weight", &embedding)?
    } else {
        schema.parameter("head.weight", table)?
    };
    let model = CausalLm::new(CausalLmParts {
        embedding: Embedding::new(embedding),
        embedding_scale: spec.embedding_scale,
        rotary: spec.rotary,
        blocks,
        norm,
        head: Linear::new(head, None),
        caches,
    });
    match spec.draft {
        Some(draft) => Ok(model.with_draft_layer(draft(schema)?)),
        None => Ok(model),
    }
}

// Imports a Llama-family checkpoint whose decoder lies under `model` (`model`, or `model.language_model` in a model that also reads images or audio) into `causal_lm`'s names.
// Rules for a model's other weights go before these, since the first rule that matches a name decides it.
#[must_use]
pub fn llama_rules(model: &str) -> Rules {
    let layers = join(model, "layers.*");
    Rules::new()
        .rename(&join(model, "embed_tokens"), "embeddings.tokens")
        .then(llama_block_rules(&layers, "blocks.*"))
        // Older checkpoints store the rotary frequencies, which the rotary table computes.
        .skip(&join(&layers, "self_attn.rotary_emb"))
        .rename(&join(model, "norm"), "norm")
        .rename("lm_head", "head")
}

#[must_use]
pub fn llama_block_rules(from: &str, to: &str) -> Rules {
    Rules::new().rename_each(
        from,
        to,
        &[
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
        ],
    )
}

// Caches for `blocks` blocks of keys and values, for a hand-written [`DecoderModel`].
pub fn kv_caches(
    blocks: usize,
    data_type: DataType,
    device: Device,
    size: CacheSize,
    spec: KvCacheSpec,
) -> SingeResult<Vec<KvCache>> {
    (0..blocks)
        .map(|_| KvCache::create(device, data_type, size, spec))
        .collect()
}

// Each sequence's tokens' final normed states, `[ΣT, W]` in order, from one uncached step of a decoder over every sequence, as an embedding model reads a causal language model.
pub fn hidden_states<L: DecoderModel + DraftingModel>(
    layer: &L,
    sequences: &[&[i64]],
    state: &ModelState,
) -> SingeResult<Tensor> {
    let device = state.device()?;
    let data_type = state.floating_type()?;
    let longest = sequences
        .iter()
        .map(|tokens| tokens.len())
        .max()
        .unwrap_or(1);
    let capacity = i64::try_from(longest).map_err(|_| SingeError::out_of_range("tokens"))?;
    let mut pages = DecoderOptions::new(sequences.len().max(1), capacity).page_table()?;
    let tables = layer.tables(data_type, device, capacity)?;
    let caches = layer.caches(data_type, device, pages.cache_size()?)?;
    let step = pages.step(device, sequences)?;
    let (_, hidden) = layer.forward_rows(&step, &tables, &caches, state)?;
    Ok(hidden)
}
