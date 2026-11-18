// Qwen3.5's text decoder: blocks of gated full attention or gated delta nets, a gated feed-forward layer or a mixture of experts, norms that scale by one plus their weight, and a multi-token-prediction draft layer.
//
// A block is `attention_norm`, `attention` or `delta_net`, `feed_forward_norm`, and `feed_forward`; the draft layer is `draft`.

use singe::{
    DataType, Result as SingeResult,
    nn::{
        Decoder, DecoderFeedForward, GatedDeltaNet, GatedDeltaNetLayers, GatedDeltaNetSpec,
        GatedMlp, GroupedQueryAttention, GroupedQueryLayers, HeadProjection, HybridCacheSpec,
        HybridLayer, MixtureOfExperts, Model, ModelState, MoeLayers, MoeRouting,
        MultiTokenPrediction, OffsetRmsNorm, PreNormBlock, StateBuilder,
    },
};

use super::{Qwen35ExpertsConfig, Qwen35TextConfig};
use crate::{
    assembly::{CausalLmSpec, DecoderConfig, DraftLayer, causal, causal_lm, gated_mlp},
    schema::Schema,
};

// A layer of Qwen3.5: gated full attention, or a gated delta net.
pub type Qwen35Layer = HybridLayer<GroupedQueryAttention, GatedDeltaNet>;

pub type Qwen35Block = PreNormBlock<Qwen35Layer, DecoderFeedForward, OffsetRmsNorm>;

pub type Qwen35Layers = Decoder<Qwen35Layer, DecoderFeedForward, OffsetRmsNorm>;

pub type Qwen35 = Model<Qwen35Layers>;

// Registers `query`, which gives each head its query and output gate, `key`, `value`, `output`, and the offset norms `query_norm` and `key_norm`.
fn gated_attention(
    schema: &mut Schema<'_>,
    config: &DecoderConfig,
) -> SingeResult<GroupedQueryAttention> {
    let (width, head_dim, bias, eps) = (
        config.width,
        config.head_dim,
        config.attention_bias,
        config.eps,
    );
    let query = HeadProjection::new(
        schema.linear("query", width, 2 * config.heads * head_dim, bias)?,
        config.heads,
        head_dim,
    )
    .with_output_gate()
    .with_norm(schema.offset_rms_norm("query_norm", head_dim, eps)?);
    let key = HeadProjection::new(
        schema.linear("key", width, config.kv_heads * head_dim, bias)?,
        config.kv_heads,
        head_dim,
    )
    .with_norm(schema.offset_rms_norm("key_norm", head_dim, eps)?);
    let value = HeadProjection::new(
        schema.linear("value", width, config.kv_heads * head_dim, bias)?,
        config.kv_heads,
        head_dim,
    );
    let output = schema.linear("output", config.heads * head_dim, width, bias)?;
    GroupedQueryAttention::create(
        GroupedQueryLayers {
            query,
            key,
            value,
            output,
        },
        causal(),
    )
}

// Registers the delta net's projections `qkv`, `gate`, `strength`, and `decay`, its `convolution`, `decay_log`, `decay_bias`, the gated `norm`, and `output`.
fn delta_net(
    schema: &mut Schema<'_>,
    config: &DecoderConfig,
    net: GatedDeltaNetSpec,
) -> SingeResult<GatedDeltaNet> {
    let width = config.width;
    let value_width = net.value_heads * net.value_dim;
    let layers = GatedDeltaNetLayers {
        qkv: schema.linear("qkv", width, net.channels(), false)?,
        gate: schema.linear("gate", width, value_width, false)?,
        strength: schema.linear("strength", width, net.value_heads, false)?,
        decay: schema.linear("decay", width, net.value_heads, false)?,
        convolution: schema.parameter("convolution.weight", [net.channels(), 1, net.kernel])?,
        decay_log: schema.parameter("decay_log", [net.value_heads])?,
        decay_bias: schema.parameter("decay_bias", [net.value_heads])?,
        norm: schema.rms_norm("norm", net.value_dim, config.eps)?,
        output: schema.linear("output", value_width, width, false)?,
    };
    GatedDeltaNet::create(layers, net)
}

// Registers `feed_forward`'s `router`, the packed `experts.gate_up` and `experts.down`, `shared_gate`, and the `shared` expert's `gate`, `up`, and `down`.
fn mixture_of_experts(
    schema: &mut Schema<'_>,
    experts: &Qwen35ExpertsConfig,
    config: &DecoderConfig,
) -> SingeResult<MixtureOfExperts> {
    let (width, inner, count) = (config.width, experts.inner, experts.experts);
    let mut feed_forward = schema.scope("feed_forward");
    let router = feed_forward.parameter("router.weight", [count, width])?;
    let gate_up = feed_forward.parameter("experts.gate_up", [count, 2 * inner, width])?;
    let down = feed_forward.parameter("experts.down", [count, width, inner])?;
    let shared_gate = feed_forward.linear("shared_gate", width, 1, false)?;
    let shared_inner = experts.shared_inner;
    let mut shared = feed_forward.scope("shared");
    let shared = GatedMlp::new(
        shared.linear("gate", width, shared_inner, false)?,
        shared.linear("up", width, shared_inner, false)?,
        shared.linear("down", shared_inner, width, false)?,
        config.activation,
    );
    Ok(MixtureOfExperts::new(
        MoeLayers {
            router,
            gate_up,
            down,
            shared,
            shared_gate: Some(Box::new(shared_gate)),
        },
        MoeRouting::Softmax {
            top_k: experts.top_k,
        },
    ))
}

// Registers block `index`, or with none, the draft layer's block, which is full attention.
fn block(
    schema: &mut Schema<'_>,
    config: &Qwen35TextConfig,
    index: Option<usize>,
) -> SingeResult<Qwen35Block> {
    let decoder = &config.decoder;
    let attention_norm = schema.offset_rms_norm("attention_norm", decoder.width, decoder.eps)?;
    let layer = if index.is_none_or(|index| config.is_full_attention(index)) {
        HybridLayer::Attention(gated_attention(&mut schema.scope("attention"), decoder)?)
    } else {
        HybridLayer::State(delta_net(
            &mut schema.scope("delta_net"),
            decoder,
            config.delta_net,
        )?)
    };
    let feed_forward_norm =
        schema.offset_rms_norm("feed_forward_norm", decoder.width, decoder.eps)?;
    let feed_forward = match &config.experts {
        Some(experts) => DecoderFeedForward::Moe(mixture_of_experts(schema, experts, decoder)?),
        None => DecoderFeedForward::Dense(gated_mlp(schema, decoder)?),
    };
    Ok(PreNormBlock::new(
        attention_norm,
        layer,
        feed_forward_norm,
        feed_forward,
    ))
}

// Registers `draft`: the norms of the next token's embedding and of the hidden state, the `projection` of both, a `block`, and its `norm`.
fn draft_layer(
    builder: &mut StateBuilder,
    config: &Qwen35TextConfig,
    data_type: DataType,
) -> SingeResult<MultiTokenPrediction<Qwen35Block, OffsetRmsNorm>> {
    let decoder = &config.decoder;
    let (width, eps) = (decoder.width, decoder.eps);
    let mut draft = Schema::new(builder.scope("draft"), data_type);
    Ok(MultiTokenPrediction {
        embedding_norm: draft.offset_rms_norm("embedding_norm", width, eps)?,
        hidden_norm: draft.offset_rms_norm("hidden_norm", width, eps)?,
        projection: draft.linear("projection", 2 * width, width, false)?,
        block: block(&mut draft.scope("block"), config, None)?,
        cache: HybridCacheSpec::Attention(decoder.kv_cache()),
        norm: draft.offset_rms_norm("norm", width, eps)?,
        width,
    })
}

// Builds Qwen3.5's decoder layers and their state on `Device::Meta`, with its draft layer if `drafting`.
pub(crate) fn decoder(
    config: &Qwen35TextConfig,
    data_type: DataType,
    drafting: bool,
) -> SingeResult<(Qwen35Layers, ModelState)> {
    let decoder = &config.decoder;
    let net = config.delta_net;
    let draft: DraftLayer<'_, Qwen35Block, OffsetRmsNorm> =
        &|builder: &mut StateBuilder| draft_layer(builder, config, data_type);
    let spec: CausalLmSpec<'_, _, Qwen35Block, OffsetRmsNorm> =
        CausalLmSpec::new(decoder.rotary(), |index| {
            if config.is_full_attention(index) {
                HybridCacheSpec::Attention(decoder.kv_cache())
            } else {
                HybridCacheSpec::State(net.cache())
            }
        });
    let spec = if drafting {
        spec.with_draft(draft)
    } else {
        spec
    };
    causal_lm(decoder, data_type, spec, |schema, index| {
        block(schema, config, Some(index))
    })
}
