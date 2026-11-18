// What decoder definitions share: the parts of a causal model under Singe's names, and the rules that import Llama-family checkpoints into them.
//
// A causal model's state is `embeddings.tokens`, `blocks.N`, `norm`, and `head`, and a block's is `attention_norm`, `attention` (`query`, `key`, `value`, `output`, and `query_norm` and `key_norm` when heads are normed), `feed_forward_norm`, and `feed_forward` (`gate`, `up`, `down`).

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Device, Error as SingeError, Result as SingeResult, Tensor,
    generation::{CachedDecoder, Decoder as GenerationDecoder, DecoderOptions, DraftingModel},
    nn::{
        Activation, CacheSize, CausalLm, CausalLmParts, DecodeCache, Decoder, DecoderModel,
        DecodingLayer, Embedding, GatedMlp, GroupedQueryAttention, GroupedQueryLayers,
        HeadProjection, KvCache, KvCacheSpec, Layer, Linear, Model, ModelState,
        MultiTokenPrediction, PreNormBlock, RotaryFrequencies, RotaryLayout, RotarySpec,
        StateBuilder,
    },
    tensor::AttentionOptions,
};

use crate::{
    error::Result,
    format::Rules,
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
}

impl DecoderConfig {
    // The rotary table of grouped-query attention, which turns each head's first `rotary_dim` features.
    #[must_use]
    pub const fn rotary(&self) -> RotarySpec {
        RotarySpec {
            head_dim: self.rotary_dim,
            frequencies: self.frequencies,
            layout: RotaryLayout::HalfSplit,
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
    config: &DecoderConfig,
    data_type: DataType,
    options: AttentionOptions,
) -> SingeResult<Model<GroupedDecoder>> {
    let (layer, state) = grouped_decoder_parts(config, data_type, options)?;
    Ok(Model::new(layer, state))
}

// The decoder and its state apart, for a model that wraps the decoder, as one reading images does.
pub fn grouped_decoder_parts(
    config: &DecoderConfig,
    data_type: DataType,
    options: AttentionOptions,
) -> SingeResult<(GroupedDecoder, ModelState)> {
    causal_lm(
        config,
        data_type,
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
    let (width, bias) = (config.width, config.mlp_bias);
    let inner = config
        .inner
        .ok_or_else(|| SingeError::missing_input("a dense feed-forward layer's width"))?;
    let mut schema = schema.scope("feed_forward");
    Ok(GatedMlp::new(
        schema.linear("gate", width, inner, bias)?,
        schema.linear("up", width, inner, bias)?,
        schema.linear("down", inner, width, bias)?,
        config.activation,
    ))
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

// Registers `query`, `key`, `value`, and `output`, and with `config.head_norms`, `query_norm` and `key_norm`.
pub fn grouped_attention(
    schema: &mut Schema<'_>,
    config: &DecoderConfig,
    options: AttentionOptions,
) -> SingeResult<GroupedQueryAttention> {
    let (width, head_dim, bias) = (config.width, config.head_dim, config.attention_bias);
    let mut head = |name: &str, norm: Option<&str>, heads: i64| -> SingeResult<HeadProjection> {
        let linear = schema.linear(name, width, heads * head_dim, bias)?;
        let projection = HeadProjection::new(linear, heads, head_dim);
        Ok(match norm.filter(|_| config.head_norms) {
            Some(norm) => projection.with_norm(schema.rms_norm(norm, head_dim, config.eps)?),
            None => projection,
        })
    };
    let query = head("query", Some("query_norm"), config.heads)?;
    let key = head("key", Some("key_norm"), config.kv_heads)?;
    let value = head("value", None, config.kv_heads)?;
    let output = schema.linear("output", config.heads * head_dim, width, bias)?;
    GroupedQueryAttention::create(
        GroupedQueryLayers {
            query,
            key,
            value,
            output,
        },
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
    &'a dyn Fn(&mut StateBuilder) -> SingeResult<MultiTokenPrediction<B, N>>;

// What a causal model takes beside its blocks: its rotary table, each block's cache by index, a scale of its embeddings, and a draft layer.
pub struct CausalLmSpec<'a, C, B: DecodingLayer, N> {
    pub rotary: RotarySpec,
    pub cache: C,
    pub embedding_scale: Option<f32>,
    pub draft: Option<DraftLayer<'a, B, N>>,
}

impl<'a, C, B: DecodingLayer, N> CausalLmSpec<'a, C, B, N>
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

    #[must_use]
    pub const fn with_embedding_scale(mut self, scale: f32) -> Self {
        self.embedding_scale = Some(scale);
        self
    }

    #[must_use]
    pub const fn with_draft(mut self, draft: DraftLayer<'a, B, N>) -> Self {
        self.draft = Some(draft);
        self
    }
}

// Builds a causal model on `Device::Meta`: `embeddings.tokens`, `blocks.N` from `block`, `norm`, and `head`, tied to the embeddings when the config says so, and `draft` from the spec's draft layer.
pub fn causal_lm<B, N, C>(
    config: &DecoderConfig,
    data_type: DataType,
    spec: CausalLmSpec<'_, C, B, N>,
    block: impl Fn(&mut Schema<'_>, usize) -> SingeResult<B>,
) -> SingeResult<(CausalLm<B, N>, ModelState)>
where
    B: DecodingLayer,
    N: SchemaNorm + Layer,
    C: Fn(usize) -> <B::Cache as DecodeCache>::Spec,
{
    let table = [config.vocab_size, config.width];
    let mut builder = StateBuilder::new();
    let (embedding, blocks, norm) = {
        let mut model = Schema::root(&mut builder, data_type);
        let embedding = model.parameter("embeddings.tokens.weight", table)?;
        let blocks = (0..config.layers)
            .zip(0_usize..)
            .map(|(layer, index)| block(&mut model.scope(format!("blocks.{layer}")), index))
            .collect::<SingeResult<Vec<_>>>()?;
        let norm = N::register(&mut model, "norm", config.width, config.eps)?;
        (embedding, blocks, norm)
    };
    let caches = (0..blocks.len()).map(spec.cache).collect();
    let head = if config.tied {
        builder.tie("head.weight", &embedding)?
    } else {
        Schema::root(&mut builder, data_type).parameter("head.weight", table)?
    };
    let mut model = CausalLm::new(CausalLmParts {
        embedding: Embedding::new(embedding),
        embedding_scale: spec.embedding_scale,
        rotary: spec.rotary,
        blocks,
        norm,
        head: Linear::new(head, None),
        caches,
    });
    if let Some(draft) = spec.draft {
        model = model.with_draft_layer(draft(&mut builder)?);
    }

    Ok((model, builder.finish()?))
}

// Imports a Llama-family checkpoint whose decoder lies under `model` (`model`, or `model.language_model` in a model that also reads images or audio) into `causal_lm`'s names.
// Rules for a model's other weights go before these, since the first rule that matches a name decides it.
#[must_use]
pub fn llama_rules(model: &str) -> Rules {
    // An empty prefix reads a bare `*Model`, as embedding models store it.
    let at = |name: &str| {
        if model.is_empty() {
            name.to_owned()
        } else {
            format!("{model}.{name}")
        }
    };
    let layer = |name: &str| at(&format!("layers.*.{name}"));
    Rules::new()
        .rename(&at("embed_tokens"), "embeddings.tokens")
        .rename(&layer("input_layernorm"), "blocks.*.attention_norm")
        .rename(
            &layer("post_attention_layernorm"),
            "blocks.*.feed_forward_norm",
        )
        .rename(&layer("self_attn.q_proj"), "blocks.*.attention.query")
        .rename(&layer("self_attn.k_proj"), "blocks.*.attention.key")
        .rename(&layer("self_attn.v_proj"), "blocks.*.attention.value")
        .rename(&layer("self_attn.o_proj"), "blocks.*.attention.output")
        .rename(&layer("self_attn.q_norm"), "blocks.*.attention.query_norm")
        .rename(&layer("self_attn.k_norm"), "blocks.*.attention.key_norm")
        // Older checkpoints store the rotary frequencies, which the rotary table computes.
        .skip(&layer("self_attn.rotary_emb"))
        .rename(&layer("mlp.gate_proj"), "blocks.*.feed_forward.gate")
        .rename(&layer("mlp.up_proj"), "blocks.*.feed_forward.up")
        .rename(&layer("mlp.down_proj"), "blocks.*.feed_forward.down")
        .rename(&at("norm"), "norm")
        .rename("lm_head", "head")
}

// Returns a decoder over `model`, compiled if asked.
pub fn decoder<L: DecoderModel>(
    model: &Model<L>,
    options: DecoderOptions,
    compile: bool,
) -> Result<Box<dyn GenerationDecoder>> {
    let mut decoder = CachedDecoder::create(model, options)?;
    if compile {
        decoder.compile()?;
    }
    Ok(Box::new(decoder))
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
