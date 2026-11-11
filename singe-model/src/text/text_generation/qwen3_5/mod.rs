use std::path::Path;

use serde::Deserialize;
use serde_json::Value;
use singe::{
    DataType, Device, Result as SingeResult,
    nn::{
        Decoder, GatedDeltaNet, GatedDeltaNetLayers, GatedDeltaNetSpec, GatedMlp,
        GroupedQueryAttention, GroupedQueryLayers, HeadProjection, HybridCacheSpec, HybridLayer,
        Model, OffsetRmsNorm, PreNormBlock,
    },
};

use crate::{
    assembly::{DecoderSpec, causal, causal_lm_in, gated_mlp, load_weights},
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
}

const fn default_full_attention_interval() -> usize {
    4
}

const fn default_output_gate() -> bool {
    true
}

impl Qwen35TextConfig {
    // A checkpoint using one of these would load and then compute something else.
    fn validate(&self) -> Result<()> {
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

// Qwen3.5's text decoder: blocks of gated full attention or gated delta nets, a gated feed-forward layer, and norms that scale by one plus their weight.
pub type Qwen35Layers = Decoder<Qwen35Layer, GatedMlp, OffsetRmsNorm>;

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

// Builds Qwen3.5's decoder on `Device::Meta`, under `scope`.
fn schema(config: &Qwen35TextConfig, scope: &str, data_type: DataType) -> Result<Qwen35> {
    let spec = config.decoder.spec(data_type, true)?;
    let net = config.delta_net();
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
        |schema, index| {
            let attention_norm = schema.offset_rms_norm("input_layernorm", spec.width, spec.eps)?;
            let layer = if config.is_full_attention(index) {
                HybridLayer::Attention(gated_attention(&mut schema.scope("self_attn"), &spec)?)
            } else {
                HybridLayer::State(delta_net(&mut schema.scope("linear_attn"), &spec, net)?)
            };
            let feed_forward_norm =
                schema.offset_rms_norm("post_attention_layernorm", spec.width, spec.eps)?;
            let feed_forward = gated_mlp(schema, &spec)?;
            Ok(PreNormBlock::new(
                attention_norm,
                layer,
                feed_forward_norm,
                feed_forward,
            ))
        },
    )?;
    Ok(Model::new(decoder, state))
}

// Loads a Hugging Face snapshot directory, converting the weights to `data_type`.
//
// A snapshot that also reads images keeps its text decoder under `model.language_model`, and its vision encoder and multi-token-prediction layer, which decoding text does not read, are skipped.
pub fn load(snapshot: impl AsRef<Path>, device: Device, data_type: DataType) -> Result<Qwen35> {
    let config: Value = Snapshot::new(&snapshot).config()?;
    let (text, scope) = config
        .get("text_config")
        .cloned()
        .map_or_else(|| (config, "model"), |text| (text, "model.language_model"));
    let config: Qwen35TextConfig = serde_json::from_value(text).map_err(Error::ConfigFormat)?;
    config.validate()?;
    let names = |name: &str| {
        (!name.starts_with("model.visual.") && !name.starts_with("mtp.")).then(|| name.to_owned())
    };
    load_weights(
        &schema(&config, scope, data_type)?,
        snapshot,
        device,
        Some(&names),
    )
}
