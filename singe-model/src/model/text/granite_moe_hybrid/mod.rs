//! Granite-4.0-H: blocks of Mamba2 mixers or attention without positions, each followed by a gated feed-forward layer, with scaled embeddings, residual branches, and logits.
//!
//! A block is `attention_norm`, `attention` or `mixer`, `feed_forward_norm`, and `feed_forward`.
//! The residual multiplier is folded into the output projection of each branch when the checkpoint loads.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult,
    nn::{
        Decoder, GatedMlp, GroupedQueryAttention, HybridCacheSpec, HybridLayer, Mamba2Mixer,
        Mamba2Spec, Model, PreNormBlock,
    },
    tensor::AttentionOptions,
};

use crate::{
    assembly::{
        CausalLmSpec, DecoderConfig, Mamba2Layout, causal, causal_lm, gated_mlp, grouped_attention,
        mamba2_mixer,
    },
    error::Result,
    format::Loader,
    head::HeadLoader,
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraniteMoeHybridConfig {
    pub decoder: DecoderConfig,
    pub mixer: Mamba2Spec,
    /// The blocks whose layer is attention; the others are mixers.
    pub attention_layers: Vec<usize>,
    /// Whether attention rotates its queries and keys, which Granite-4.0-H does not.
    pub rotary: bool,
    pub mixer_bias: bool,
    pub convolution_bias: bool,
    pub attention_scale: f32,
    pub embedding_scale: f32,
    pub logit_scale: f32,
    pub residual_scale: f32,
}

pub type GraniteMoeHybridLayer = HybridLayer<GroupedQueryAttention, Mamba2Mixer>;

pub type GraniteMoeHybridLayers = Decoder<GraniteMoeHybridLayer, GatedMlp>;

pub type GraniteMoeHybrid = Model<GraniteMoeHybridLayers>;

fn block(
    schema: &mut Schema<'_>,
    config: &GraniteMoeHybridConfig,
    index: usize,
) -> SingeResult<PreNormBlock<GraniteMoeHybridLayer, GatedMlp>> {
    let decoder = &config.decoder;
    let attention_norm = schema.rms_norm("attention_norm", decoder.width, decoder.eps)?;
    let layer = if config.attention_layers.contains(&index) {
        let options = AttentionOptions {
            scale: Some(config.attention_scale),
            ..causal()
        };
        HybridLayer::Attention(grouped_attention(
            &mut schema.scope("attention"),
            decoder,
            options,
        )?)
    } else {
        HybridLayer::State(mamba2_mixer(
            &mut schema.scope("mixer"),
            decoder.width,
            config.mixer,
            decoder.eps,
            Mamba2Layout {
                projection_bias: config.mixer_bias,
                convolution_bias: config.convolution_bias,
                norm: true,
            },
        )?)
    };
    let feed_forward_norm = schema.rms_norm("feed_forward_norm", decoder.width, decoder.eps)?;
    let feed_forward = gated_mlp(schema, decoder)?;
    Ok(PreNormBlock::new(
        attention_norm,
        layer,
        feed_forward_norm,
        feed_forward,
    ))
}

/// Builds Granite-4.0-H's decoder on [`singe::Device::Meta`].
pub fn layers(
    schema: &mut Schema<'_>,
    config: &GraniteMoeHybridConfig,
) -> SingeResult<GraniteMoeHybridLayers> {
    let decoder = &config.decoder;
    let rotary = config.rotary.then(|| decoder.rotary());
    let spec = CausalLmSpec {
        embedding_scale: Some(config.embedding_scale),
        logit_scale: Some(config.logit_scale),
        ..CausalLmSpec::new(rotary, |index| {
            if config.attention_layers.contains(&index) {
                HybridCacheSpec::Attention(decoder.kv_cache())
            } else {
                HybridCacheSpec::State(config.mixer.cache())
            }
        })
    };
    causal_lm(schema, decoder, spec, |schema, index| {
        block(schema, config, index)
    })
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "granitemoehybrid",
    formats: &[(Format::Transformers, &["granitemoehybrid"])],
    loaders: &[HeadLoader::CausalLm(|loader| Ok(Box::new(load(loader)?)))],
};

pub fn load(loader: &mut Loader<'_>) -> Result<GraniteMoeHybrid> {
    let config: GraniteMoeHybridConfig = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| layers(schema, &config))?;
    loader.load_whole(&model, &transformers::rules(config.residual_scale))
}
