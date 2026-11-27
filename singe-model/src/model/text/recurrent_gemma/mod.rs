//! `RecurrentGemma`: Griffin's blocks of real-gated linear recurrent units or sliding-window multi-query attention, each followed by a gated feed-forward layer, with Gemma's norms, a scaled embedding, and capped logits.
//!
//! A block is `attention_norm`, `attention` or `recurrent`, `feed_forward_norm`, and `feed_forward`.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult,
    nn::{
        Decoder, GatedMlp, GroupedQueryAttention, HybridCacheSpec, HybridLayer, Model,
        OffsetRmsNorm, PreNormBlock, RgLru, RgLruLayers, RgLruSpec,
    },
    tensor::AttentionOptions,
};

use crate::{
    assembly::{
        CausalLmSpec, DecoderConfig, GroupedAttentionSpec, causal, causal_lm, gated_mlp,
        grouped_attention_layers,
    },
    error::Result,
    format::Loader,
    head::HeadLoader,
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecurrentGemmaConfig {
    pub decoder: DecoderConfig,
    pub recurrent: RgLruSpec,
    /// The blocks whose layer is attention; the others are recurrent.
    pub attention_layers: Vec<usize>,
    /// The tokens up to its own each query attends to.
    pub window: i64,
    /// What the embeddings are multiplied by, which `transformers` rounds to `BF16` whatever the model's type.
    pub embedding_scale: f32,
    /// The cap `tanh` holds the logits under.
    pub logit_cap: f32,
}

pub type RecurrentGemmaLayer = HybridLayer<GroupedQueryAttention, RgLru>;

pub type RecurrentGemmaLayers = Decoder<RecurrentGemmaLayer, GatedMlp, OffsetRmsNorm>;

pub type RecurrentGemma = Model<RecurrentGemmaLayers>;

/// Registers the recurrent layer's `input`, `gate`, `convolution`, both gates' block-diagonal weights and biases, `rate`, and `output`.
fn recurrent(schema: &mut Schema<'_>, config: &RecurrentGemmaConfig) -> SingeResult<RgLru> {
    let (width, spec) = (config.decoder.width, config.recurrent);
    let block = spec.width / spec.blocks;
    let mut gate = |name: &str| -> SingeResult<[_; 2]> {
        Ok([
            schema.parameter(&format!("{name}.weight"), [spec.blocks, block, block])?,
            schema.parameter(&format!("{name}.bias"), [spec.blocks, block])?,
        ])
    };
    let input_gate = gate("input_gate")?;
    let recurrence_gate = gate("recurrence_gate")?;
    let layers = RgLruLayers {
        input: schema.linear("input", width, spec.width, true)?,
        gate: schema.linear("gate", width, spec.width, true)?,
        convolution: schema.parameter("convolution.weight", [spec.width, 1, spec.kernel])?,
        convolution_bias: Some(schema.parameter("convolution.bias", [spec.width])?),
        input_gate,
        recurrence_gate,
        rate: schema.parameter("rate", [spec.width])?,
        output: schema.linear("output", spec.width, width, true)?,
    };
    RgLru::create(layers, spec)
}

fn block(
    schema: &mut Schema<'_>,
    config: &RecurrentGemmaConfig,
    index: usize,
) -> SingeResult<PreNormBlock<RecurrentGemmaLayer, GatedMlp, OffsetRmsNorm>> {
    let decoder = &config.decoder;
    let attention_norm = schema.offset_rms_norm("attention_norm", decoder.width, decoder.eps)?;
    let layer = if config.attention_layers.contains(&index) {
        let spec = decoder.attention_spec();
        let layers = grouped_attention_layers(
            &mut schema.scope("attention"),
            &GroupedAttentionSpec {
                output_bias: true,
                ..spec
            },
        )?;
        let options = AttentionOptions {
            window: Some(config.window),
            ..causal()
        };
        HybridLayer::Attention(GroupedQueryAttention::create(layers, options)?)
    } else {
        HybridLayer::State(recurrent(&mut schema.scope("recurrent"), config)?)
    };
    let feed_forward_norm =
        schema.offset_rms_norm("feed_forward_norm", decoder.width, decoder.eps)?;
    let feed_forward = gated_mlp(schema, decoder)?;
    Ok(PreNormBlock::new(
        attention_norm,
        layer,
        feed_forward_norm,
        feed_forward,
    ))
}

/// Builds `RecurrentGemma`'s decoder on [`singe::Device::Meta`].
pub fn layers(
    schema: &mut Schema<'_>,
    config: &RecurrentGemmaConfig,
) -> SingeResult<RecurrentGemmaLayers> {
    let decoder = &config.decoder;
    let spec = CausalLmSpec {
        embedding_scale: Some(config.embedding_scale),
        logit_cap: Some(config.logit_cap),
        ..CausalLmSpec::new(Some(decoder.rotary()), |index| {
            if config.attention_layers.contains(&index) {
                HybridCacheSpec::Attention(decoder.kv_cache())
            } else {
                HybridCacheSpec::State(config.recurrent.cache())
            }
        })
    };
    causal_lm(schema, decoder, spec, |schema, index| {
        block(schema, config, index)
    })
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "recurrent_gemma",
    formats: &[(Format::Transformers, &["recurrent_gemma"])],
    loaders: &[HeadLoader::CausalLm(|loader| Ok(Box::new(load(loader)?)))],
};

pub fn load(loader: &mut Loader<'_>) -> Result<RecurrentGemma> {
    let config: RecurrentGemmaConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| layers(schema, &config))?;
    loader.load_whole(&model, &transformers::rules())
}
