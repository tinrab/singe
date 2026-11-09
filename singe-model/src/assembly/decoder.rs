// What decoder definitions share: Hugging Face's Llama-family names and the shape of a causal model.

use singe::{
    DataType, Device, Result as SingeResult,
    generation::{CachedDecoder, Decoder as GenerationDecoder, DecoderOptions},
    nn::{
        Activation, CacheSize, CausalLm, CausalLmParts, DecodeCache, Decoder, DecoderModel,
        DecodingLayer, Embedding, GatedMlp, GroupedQueryAttention, GroupedQueryLayers,
        HeadProjection, KvCache, KvCacheSpec, Layer, Linear, Model, ModelState, PreNormBlock,
        RotaryFrequencies, RotaryLayout, RotarySpec, StateBuilder,
    },
    tensor::AttentionOptions,
};

use crate::{
    error::Result,
    schema::{Schema, SchemaNorm},
};

// One place for a decoder's sizes, which its attention, cache, and rotary table all derive from.
#[derive(Debug, Clone, Copy)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "each flag is a separate setting of a checkpoint"
)]
pub struct DecoderSpec {
    pub data_type: DataType,
    pub vocab_size: i64,
    pub width: i64,
    // The feed-forward layer's width.
    pub inner: i64,
    pub layers: i64,
    pub heads: i64,
    pub kv_heads: i64,
    pub head_dim: i64,
    pub eps: f32,
    pub activation: Activation,
    pub attention_bias: bool,
    pub mlp_bias: bool,
    pub tied: bool,
    // Whether queries and keys take `q_norm` and `k_norm`, as Qwen3's do.
    pub head_norms: bool,
    pub frequencies: RotaryFrequencies,
}

impl DecoderSpec {
    // The rotary table of grouped-query attention, which turns whole heads.
    #[must_use]
    pub const fn rotary(&self) -> RotarySpec {
        RotarySpec {
            head_dim: self.head_dim,
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
    spec: &DecoderSpec,
    options: AttentionOptions,
) -> SingeResult<Model<GroupedDecoder>> {
    let lm = causal_lm(spec, spec.rotary(), spec.kv_cache(), None, |schema, _| {
        pre_norm_block(schema, spec, |schema| {
            grouped_attention(schema, spec, options)
        })
    })?;
    Ok(Model::new(lm.0, lm.1))
}

// Registers `mlp`'s `gate_proj`, `up_proj`, and `down_proj`.
pub fn gated_mlp(schema: &mut Schema<'_>, spec: &DecoderSpec) -> SingeResult<GatedMlp> {
    let (width, inner, bias) = (spec.width, spec.inner, spec.mlp_bias);
    let mut schema = schema.scope("mlp");
    Ok(GatedMlp::new(
        schema.linear("gate_proj", width, inner, bias)?,
        schema.linear("up_proj", width, inner, bias)?,
        schema.linear("down_proj", inner, width, bias)?,
        spec.activation,
    ))
}

// A block with `input_layernorm`, `self_attn`, `post_attention_layernorm`, and `mlp`.
pub fn pre_norm_block<A, N: SchemaNorm>(
    schema: &mut Schema<'_>,
    spec: &DecoderSpec,
    attention: impl FnOnce(&mut Schema<'_>) -> SingeResult<A>,
) -> SingeResult<PreNormBlock<A, GatedMlp, N>> {
    pre_norm_block_with(schema, spec, attention, |schema| gated_mlp(schema, spec))
}

// A block as `pre_norm_block` makes, whose `mlp` is what `feed_forward` registers.
pub fn pre_norm_block_with<A, F, N: SchemaNorm>(
    schema: &mut Schema<'_>,
    spec: &DecoderSpec,
    attention: impl FnOnce(&mut Schema<'_>) -> SingeResult<A>,
    feed_forward: impl FnOnce(&mut Schema<'_>) -> SingeResult<F>,
) -> SingeResult<PreNormBlock<A, F, N>> {
    let attention_norm = N::register(schema, "input_layernorm", spec.width, spec.eps)?;
    let attention = attention(&mut schema.scope("self_attn"))?;
    let feed_forward_norm = N::register(schema, "post_attention_layernorm", spec.width, spec.eps)?;
    let feed_forward = feed_forward(schema)?;
    Ok(PreNormBlock::new(
        attention_norm,
        attention,
        feed_forward_norm,
        feed_forward,
    ))
}

// Registers `q_proj`, `k_proj`, `v_proj`, and `o_proj`, and with `spec.head_norms`, `q_norm` and `k_norm`.
pub fn grouped_attention(
    schema: &mut Schema<'_>,
    spec: &DecoderSpec,
    options: AttentionOptions,
) -> SingeResult<GroupedQueryAttention> {
    let (width, head_dim, bias) = (spec.width, spec.head_dim, spec.attention_bias);
    let mut head = |name: &str, norm: Option<&str>, heads: i64| -> SingeResult<HeadProjection> {
        let linear = schema.linear(name, width, heads * head_dim, bias)?;
        let norm = norm
            .filter(|_| spec.head_norms)
            .map(|norm| schema.rms_norm(norm, head_dim, spec.eps))
            .transpose()?;
        Ok(HeadProjection::new(linear, norm, heads, head_dim))
    };
    let query = head("q_proj", Some("q_norm"), spec.heads)?;
    let key = head("k_proj", Some("k_norm"), spec.kv_heads)?;
    let value = head("v_proj", None, spec.kv_heads)?;
    let output = schema.linear("o_proj", spec.heads * head_dim, width, bias)?;
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

// Builds a causal model on `Device::Meta` under Llama-family names: `model.embed_tokens`, `model.layers.N`, `model.norm`, and `lm_head`.
pub fn causal_lm<B, N>(
    spec: &DecoderSpec,
    rotary: RotarySpec,
    cache: <B::Cache as DecodeCache>::Spec,
    embedding_scale: Option<f32>,
    block: impl Fn(&mut Schema<'_>, usize) -> SingeResult<B>,
) -> SingeResult<(CausalLm<B, N>, ModelState)>
where
    B: DecodingLayer,
    N: SchemaNorm + Layer,
{
    let table = [spec.vocab_size, spec.width];
    let mut builder = StateBuilder::new();
    let (embedding, blocks, norm) = {
        let mut model = Schema::new(builder.scope("model"), spec.data_type);
        let embedding = model.parameter("embed_tokens.weight", table)?;
        let blocks = (0..spec.layers)
            .zip(0_usize..)
            .map(|(layer, index)| block(&mut model.scope(format!("layers.{layer}")), index))
            .collect::<SingeResult<Vec<_>>>()?;
        let norm = N::register(&mut model, "norm", spec.width, spec.eps)?;
        (embedding, blocks, norm)
    };
    let head = if spec.tied {
        builder.tie("lm_head.weight", &embedding)?
    } else {
        Schema::root(&mut builder, spec.data_type).parameter("lm_head.weight", table)?
    };
    Ok((
        CausalLm::new(CausalLmParts {
            embedding: Embedding::new(embedding),
            embedding_scale,
            rotary,
            blocks,
            norm,
            head: Linear::new(head, None),
            cache,
        }),
        builder.finish()?,
    ))
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
