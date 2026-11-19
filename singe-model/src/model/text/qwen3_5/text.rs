// Qwen3.5's text decoder: blocks of gated full attention or gated delta nets, a gated feed-forward layer or a mixture of experts, norms that scale by one plus their weight, and a multi-token-prediction draft layer.
//
// A block is `attention_norm`, `attention` or `delta_net`, `feed_forward_norm`, and `feed_forward`; the draft layer is `draft`.

use singe::{
    Result as SingeResult,
    nn::{
        Decoder, DecoderFeedForward, GatedDeltaNet, GatedDeltaNetLayers, GatedDeltaNetSpec,
        GroupedQueryAttention, HybridCacheSpec, HybridLayer, Model, MultiTokenPrediction,
        OffsetRmsNorm, PreNormBlock,
    },
};

use crate::{
    assembly::{
        CausalLmSpec, DecoderConfig, DraftLayer, GroupedAttentionSpec, HeadNorms, causal,
        causal_lm, gated_mlp, grouped_attention_layers, mixture_of_experts,
    },
    model::text::qwen3_5::{Draft, Qwen35TextConfig},
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
    let spec = GroupedAttentionSpec {
        head_norms: HeadNorms::OffsetRms(config.eps),
        output_gate: true,
        ..config.attention_spec()
    };
    GroupedQueryAttention::create(grouped_attention_layers(schema, &spec)?, causal())
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
        Some(experts) => DecoderFeedForward::Moe(mixture_of_experts(
            schema,
            experts,
            decoder.width,
            decoder.activation,
        )?),
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
    schema: &mut Schema<'_>,
    config: &Qwen35TextConfig,
) -> SingeResult<MultiTokenPrediction<Qwen35Block, OffsetRmsNorm>> {
    let decoder = &config.decoder;
    let (width, eps) = (decoder.width, decoder.eps);
    let mut draft = schema.scope("draft");
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

pub(crate) fn decoder(
    schema: &mut Schema<'_>,
    config: &Qwen35TextConfig,
    draft: Draft,
) -> SingeResult<Qwen35Layers> {
    let decoder = &config.decoder;
    let net = config.delta_net;
    let layer: DraftLayer<'_, Qwen35Block, OffsetRmsNorm> = &|schema| draft_layer(schema, config);
    let spec: CausalLmSpec<'_, _, Qwen35Block, OffsetRmsNorm> =
        CausalLmSpec::new(decoder.rotary(), |index| {
            if config.is_full_attention(index) {
                HybridCacheSpec::Attention(decoder.kv_cache())
            } else {
                HybridCacheSpec::State(net.cache())
            }
        });
    let spec = match draft {
        Draft::Omitted => spec,
        Draft::Included => CausalLmSpec {
            draft: Some(layer),
            ..spec
        },
    };
    causal_lm(schema, decoder, spec, |schema, index| {
        block(schema, config, Some(index))
    })
}
