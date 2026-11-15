use std::path::Path;

use serde::Deserialize;
use serde_json::Value;
use singe::{
    DataType, Device, Result as SingeResult,
    nn::{
        Decoder, DecoderFeedForward, GatedDeltaNet, GatedDeltaNetLayers, GatedDeltaNetSpec,
        GatedMlp, GroupedQueryAttention, GroupedQueryLayers, HeadProjection, HybridCacheSpec,
        HybridLayer, MixtureOfExperts, Model, ModelState, MoeLayers, MoeRouting,
        MultiTokenPrediction, OffsetRmsNorm, PreNormBlock, StateBuilder,
    },
};

use crate::{
    assembly::{DecoderSpec, DraftLayer, causal, causal_lm_in, gated_mlp, load_expert_weights},
    config::{DecoderConfig, Snapshot, unsupported},
    error::{Error, Result},
    schema::Schema,
};

// The text decoder's fields of a Qwen3.5 `config.json`: its top level for a text model, or its `text_config` for one that also reads images.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Qwen35TextConfig {
    #[serde(flatten)]
    pub decoder: DecoderConfig,
    pub linear_conv_kernel_dim: i64,
    pub linear_key_head_dim: i64,
    pub linear_num_key_heads: i64,
    pub linear_value_head_dim: i64,
    pub linear_num_value_heads: i64,
    // Every fourth layer is full attention when the config names no layer types.
    #[serde(default = "default_full_attention_interval")]
    pub full_attention_interval: usize,
    #[serde(default = "default_output_gate")]
    pub attn_output_gate: bool,
    #[serde(default)]
    pub num_experts: Option<i64>,
    #[serde(default)]
    pub num_experts_per_tok: i64,
    #[serde(default)]
    pub moe_intermediate_size: i64,
    #[serde(default)]
    pub shared_expert_intermediate_size: i64,
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,
}

const fn default_full_attention_interval() -> usize {
    4
}

const fn default_output_gate() -> bool {
    true
}

impl Qwen35TextConfig {
    // A checkpoint using one of these would load and then compute something else.
    pub(crate) fn validate(&self) -> Result<()> {
        if !self.attn_output_gate {
            return Err(unsupported("full attention without an output gate"));
        }

        if let Some(layers) = &self.decoder.layer_types
            && layers
                .iter()
                .any(|layer| layer != "full_attention" && layer != "linear_attention")
        {
            return Err(unsupported(
                "a layer that is neither full nor linear attention",
            ));
        }
        Ok(())
    }

    // Whether layer `index` is full attention rather than linear attention.
    fn is_full_attention(&self, index: usize) -> bool {
        self.decoder.layer_types.as_ref().map_or_else(
            || (index + 1).is_multiple_of(self.full_attention_interval.max(1)),
            |layers| {
                layers
                    .get(index)
                    .is_some_and(|layer| layer == "full_attention")
            },
        )
    }

    const fn delta_net(&self) -> GatedDeltaNetSpec {
        GatedDeltaNetSpec {
            key_heads: self.linear_num_key_heads,
            key_dim: self.linear_key_head_dim,
            value_heads: self.linear_num_value_heads,
            value_dim: self.linear_value_head_dim,
            kernel: self.linear_conv_kernel_dim,
        }
    }
}

// A layer of Qwen3.5: gated full attention, or a gated delta net.
pub type Qwen35Layer = HybridLayer<GroupedQueryAttention, GatedDeltaNet>;

pub type Qwen35Block = PreNormBlock<Qwen35Layer, DecoderFeedForward, OffsetRmsNorm>;

// Qwen3.5's text decoder: blocks of gated full attention or gated delta nets, a gated feed-forward layer or a mixture of experts, and norms that scale by one plus their weight.
pub type Qwen35Layers = Decoder<Qwen35Layer, DecoderFeedForward, OffsetRmsNorm>;

pub type Qwen35 = Model<Qwen35Layers>;

// Registers `q_proj`, which gives each head its query and output gate, `k_proj`, `v_proj`, `o_proj`, and the offset norms `q_norm` and `k_norm`.
fn gated_attention(
    schema: &mut Schema<'_>,
    spec: &DecoderSpec,
) -> SingeResult<GroupedQueryAttention> {
    let (width, head_dim, bias, eps) = (spec.width, spec.head_dim, spec.attention_bias, spec.eps);
    let query = HeadProjection::new(
        schema.linear("q_proj", width, 2 * spec.heads * head_dim, bias)?,
        spec.heads,
        head_dim,
    )
    .with_output_gate()
    .with_norm(schema.offset_rms_norm("q_norm", head_dim, eps)?);
    let key = HeadProjection::new(
        schema.linear("k_proj", width, spec.kv_heads * head_dim, bias)?,
        spec.kv_heads,
        head_dim,
    )
    .with_norm(schema.offset_rms_norm("k_norm", head_dim, eps)?);
    let value = HeadProjection::new(
        schema.linear("v_proj", width, spec.kv_heads * head_dim, bias)?,
        spec.kv_heads,
        head_dim,
    );
    let output = schema.linear("o_proj", spec.heads * head_dim, width, bias)?;
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

// Registers the delta net's projections `in_proj_qkv`, `in_proj_z`, `in_proj_b`, and `in_proj_a`, its `conv1d`, `A_log`, `dt_bias`, the gated `norm`, and `out_proj`.
fn delta_net(
    schema: &mut Schema<'_>,
    spec: &DecoderSpec,
    net: GatedDeltaNetSpec,
) -> SingeResult<GatedDeltaNet> {
    let width = spec.width;
    let value_width = net.value_heads * net.value_dim;
    let layers = GatedDeltaNetLayers {
        qkv: schema.linear("in_proj_qkv", width, net.channels(), false)?,
        gate: schema.linear("in_proj_z", width, value_width, false)?,
        strength: schema.linear("in_proj_b", width, net.value_heads, false)?,
        decay: schema.linear("in_proj_a", width, net.value_heads, false)?,
        convolution: schema.parameter("conv1d.weight", [net.channels(), 1, net.kernel])?,
        decay_log: schema.parameter("A_log", [net.value_heads])?,
        decay_bias: schema.parameter("dt_bias", [net.value_heads])?,
        norm: schema.rms_norm("norm", net.value_dim, spec.eps)?,
        output: schema.linear("out_proj", value_width, width, false)?,
    };
    GatedDeltaNet::create(layers, net)
}

fn mixture_of_experts(
    schema: &mut Schema<'_>,
    config: &Qwen35TextConfig,
    experts: i64,
    spec: &DecoderSpec,
) -> SingeResult<MixtureOfExperts> {
    let (width, inner) = (spec.width, config.moe_intermediate_size);
    let mut mlp = schema.scope("mlp");
    let router = mlp.parameter("gate.weight", [experts, width])?;
    let gate_up = mlp.parameter("experts.gate_up_proj", [experts, 2 * inner, width])?;
    let down = mlp.parameter("experts.down_proj", [experts, width, inner])?;
    let shared_gate = mlp.linear("shared_expert_gate", width, 1, false)?;
    let shared_inner = config.shared_expert_intermediate_size;
    let mut shared = mlp.scope("shared_expert");
    let shared = GatedMlp::new(
        shared.linear("gate_proj", width, shared_inner, false)?,
        shared.linear("up_proj", width, shared_inner, false)?,
        shared.linear("down_proj", shared_inner, width, false)?,
        spec.activation,
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
            top_k: config.num_experts_per_tok,
        },
    ))
}

// Builds Qwen3.5's decoder on `Device::Meta`, under `scope`, with its draft layer if `drafting`.
fn schema(
    config: &Qwen35TextConfig,
    scope: &str,
    data_type: DataType,
    drafting: bool,
) -> Result<Qwen35> {
    let (decoder, state) = decoder(config, scope, data_type, drafting)?;
    Ok(Model::new(decoder, state))
}

fn block(
    schema: &mut Schema<'_>,
    config: &Qwen35TextConfig,
    spec: &DecoderSpec,
    index: Option<usize>,
) -> SingeResult<PreNormBlock<Qwen35Layer, DecoderFeedForward, OffsetRmsNorm>> {
    let attention_norm = schema.offset_rms_norm("input_layernorm", spec.width, spec.eps)?;
    let layer = if index.is_none_or(|index| config.is_full_attention(index)) {
        HybridLayer::Attention(gated_attention(&mut schema.scope("self_attn"), spec)?)
    } else {
        HybridLayer::State(delta_net(
            &mut schema.scope("linear_attn"),
            spec,
            config.delta_net(),
        )?)
    };
    let feed_forward_norm =
        schema.offset_rms_norm("post_attention_layernorm", spec.width, spec.eps)?;
    let dense = index.is_some_and(|index| config.mlp_only_layers.contains(&index));
    let feed_forward = match config.num_experts {
        Some(experts) if !dense => {
            DecoderFeedForward::Moe(mixture_of_experts(schema, config, experts, spec)?)
        }
        _ => DecoderFeedForward::Dense(gated_mlp(schema, spec)?),
    };
    Ok(PreNormBlock::new(
        attention_norm,
        layer,
        feed_forward_norm,
        feed_forward,
    ))
}

fn draft_layer(
    builder: &mut StateBuilder,
    config: &Qwen35TextConfig,
    spec: &DecoderSpec,
) -> SingeResult<MultiTokenPrediction<Qwen35Block, OffsetRmsNorm>> {
    let mut mtp = Schema::new(builder.scope("mtp"), spec.data_type);
    let width = spec.width;
    Ok(MultiTokenPrediction {
        embedding_norm: mtp.offset_rms_norm("pre_fc_norm_embedding", width, spec.eps)?,
        hidden_norm: mtp.offset_rms_norm("pre_fc_norm_hidden", width, spec.eps)?,
        projection: mtp.linear("fc", 2 * width, width, false)?,
        block: block(&mut mtp.scope("layers.0"), config, spec, None)?,
        cache: HybridCacheSpec::Attention(spec.kv_cache()),
        norm: mtp.offset_rms_norm("norm", width, spec.eps)?,
        width,
    })
}

// Builds Qwen3.5's decoder layers and their state on `Device::Meta`, under `scope`, with its draft layer if `drafting`.
pub(crate) fn decoder(
    config: &Qwen35TextConfig,
    scope: &str,
    data_type: DataType,
    drafting: bool,
) -> Result<(Qwen35Layers, ModelState)> {
    let spec = config.decoder.spec(data_type, true)?;
    let net = config.delta_net();
    let draft = |builder: &mut StateBuilder| draft_layer(builder, config, &spec);
    let (decoder, state) = causal_lm_in(
        scope,
        &spec,
        spec.rotary(),
        |index| {
            if config.is_full_attention(index) {
                HybridCacheSpec::Attention(spec.kv_cache())
            } else {
                HybridCacheSpec::State(net.cache())
            }
        },
        None,
        |schema, index| block(schema, config, &spec, Some(index)),
        drafting.then_some(&draft as DraftLayer<'_, _, _>),
    )?;
    Ok((decoder, state))
}

// Loads a Hugging Face snapshot directory, converting the weights to `data_type`.
//
// A snapshot that also reads images keeps its text decoder under `model.language_model`, and its vision encoder and multi-token-prediction layer, which decoding text does not read, are skipped.
pub fn load(snapshot: impl AsRef<Path>, device: Device, data_type: DataType) -> Result<Qwen35> {
    load_with(snapshot.as_ref(), device, data_type, false)
}

pub fn load_drafting(
    snapshot: impl AsRef<Path>,
    device: Device,
    data_type: DataType,
) -> Result<Qwen35> {
    load_with(snapshot.as_ref(), device, data_type, true)
}

fn load_with(
    snapshot: &Path,
    device: Device,
    data_type: DataType,
    drafting: bool,
) -> Result<Qwen35> {
    let config: Value = Snapshot::new(snapshot).config()?;
    let (text, scope) = config
        .get("text_config")
        .cloned()
        .map_or_else(|| (config, "model"), |text| (text, "model.language_model"));
    let config: Qwen35TextConfig = serde_json::from_value(text).map_err(Error::ConfigFormat)?;
    config.validate()?;
    if drafting && config.num_experts.is_some() {
        return Err(unsupported("a draft layer of experts"));
    }

    let names = move |name: &str| {
        (!name.starts_with("model.visual.") && (drafting || !name.starts_with("mtp.")))
            .then(|| name.to_owned())
    };
    load_expert_weights(
        &schema(&config, scope, data_type, drafting)?,
        snapshot,
        device,
        &names,
        config.moe_intermediate_size,
    )
}
