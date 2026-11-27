//! Falcon-H1: blocks that run attention and a Mamba2 mixer side by side on the same normed input and add both to the residual, then a gated feed-forward layer.
//!
//! A block is `attention_norm`, `attention` and `mixer`, `feed_forward_norm`, and `feed_forward`.
//! The μP multipliers inside a block are folded into its weights when the checkpoint loads; the embedding and logit multipliers stay scales of the model.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult,
    nn::{
        Decoder, GatedMlp, GroupedQueryAttention, HybridCacheSpec, HybridLayer, Mamba2Mixer,
        Mamba2Spec, Model, PreNormBlock,
    },
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

/// The multipliers Falcon-H1 applies inside its blocks, which loading folds into the weights they follow.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FalconH1Multipliers {
    pub attention_in: f32,
    pub attention_out: f32,
    pub key: f32,
    pub mixer_in: f32,
    pub mixer_out: f32,
    /// The mixer's input projection's sections: the gate, the features, the input and output projections of the state, and the step.
    pub sections: [f32; 5],
    pub gate: f32,
    pub down: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FalconH1Config {
    pub decoder: DecoderConfig,
    pub mixer: Mamba2Spec,
    /// Whether the mixer norms its gated output, over groups of features.
    pub mixer_norm: bool,
    pub mixer_bias: bool,
    pub convolution_bias: bool,
    pub embedding_scale: f32,
    pub logit_scale: f32,
    pub multipliers: FalconH1Multipliers,
}

pub type FalconH1Layer = HybridLayer<GroupedQueryAttention, Mamba2Mixer>;

pub type FalconH1Layers = Decoder<FalconH1Layer, GatedMlp>;

pub type FalconH1 = Model<FalconH1Layers>;

fn block(
    schema: &mut Schema<'_>,
    config: &FalconH1Config,
) -> SingeResult<PreNormBlock<FalconH1Layer, GatedMlp>> {
    let decoder = &config.decoder;
    let attention_norm = schema.rms_norm("attention_norm", decoder.width, decoder.eps)?;
    let attention = grouped_attention(&mut schema.scope("attention"), decoder, causal())?;
    let mixer = mamba2_mixer(
        &mut schema.scope("mixer"),
        decoder.width,
        config.mixer,
        decoder.eps,
        Mamba2Layout {
            projection_bias: config.mixer_bias,
            convolution_bias: config.convolution_bias,
            norm: config.mixer_norm,
        },
    )?;
    let feed_forward_norm = schema.rms_norm("feed_forward_norm", decoder.width, decoder.eps)?;
    let feed_forward = gated_mlp(schema, decoder)?;
    Ok(PreNormBlock::new(
        attention_norm,
        HybridLayer::Parallel(attention, mixer),
        feed_forward_norm,
        feed_forward,
    ))
}

/// Builds Falcon-H1's decoder on [`singe::Device::Meta`].
pub fn layers(schema: &mut Schema<'_>, config: &FalconH1Config) -> SingeResult<FalconH1Layers> {
    let decoder = &config.decoder;
    let cache = HybridCacheSpec::Parallel(decoder.kv_cache(), config.mixer.cache());
    let spec = CausalLmSpec {
        embedding_scale: Some(config.embedding_scale),
        logit_scale: Some(config.logit_scale),
        ..CausalLmSpec::new(Some(decoder.rotary()), |_| cache)
    };
    causal_lm(schema, decoder, spec, |schema, _| block(schema, config))
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "falcon_h1",
    formats: &[(Format::Transformers, &["falcon_h1"])],
    loaders: &[HeadLoader::CausalLm(|loader| Ok(Box::new(load(loader)?)))],
};

pub fn load(loader: &mut Loader<'_>) -> Result<FalconH1> {
    let config: FalconH1Config = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| layers(schema, &config))?;
    loader.load_whole(&model, &transformers::rules(&config))
}
