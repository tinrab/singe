//! Gemma 2: four norms per block, a scaled embedding, a norm that scales by one plus its weight, sliding-window attention in alternate blocks, and capped attention scores and logits.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult,
    nn::{Decoder, GatedMlp, GroupedQueryAttention, Model, OffsetRmsNorm},
    tensor::AttentionOptions,
};

use crate::{
    assembly::{CausalLmSpec, DecoderConfig, causal, causal_lm, grouped_attention, pre_norm_block},
    error::Result,
    format::Loader,
    head::HeadLoader,
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gemma2Config {
    pub decoder: DecoderConfig,
    /// What attention multiplies its scores by.
    pub attention_scale: f32,
    /// Each block's window, whose keys are that many tokens up to each query's own, or `None` for full attention.
    pub windows: Vec<Option<i64>>,
    /// The cap `tanh` holds attention scores under, or `None` for no cap.
    pub attention_softcap: Option<f32>,
    /// The cap `tanh` holds the logits under, or `None` for no cap.
    pub logit_softcap: Option<f32>,
}

impl Gemma2Config {
    fn window(&self, index: usize) -> Option<i64> {
        self.windows.get(index).copied().flatten()
    }
}

pub type Gemma2Layers = Decoder<GroupedQueryAttention, GatedMlp, OffsetRmsNorm>;

/// Builds Gemma 2's decoder on [`singe::Device::Meta`].
pub fn layers(schema: &mut Schema<'_>, config: &Gemma2Config) -> SingeResult<Gemma2Layers> {
    let decoder = &config.decoder;
    let (width, eps) = (decoder.width, decoder.eps);
    let options = AttentionOptions {
        scale: Some(config.attention_scale),
        softcap: config.attention_softcap,
        ..causal()
    };
    causal_lm(
        schema,
        decoder,
        CausalLmSpec {
            embedding_scale: Some((width as f64).sqrt() as f32),
            logit_cap: config.logit_softcap,
            ..CausalLmSpec::new(Some(decoder.rotary()), |_| decoder.kv_cache())
        },
        |schema, index| {
            let block = pre_norm_block(schema, decoder, |schema| {
                grouped_attention(
                    schema,
                    decoder,
                    AttentionOptions {
                        window: config.window(index),
                        ..options
                    },
                )
            })?;
            Ok(block.with_output_norms(
                schema.offset_rms_norm("attention_output_norm", width, eps)?,
                schema.offset_rms_norm("feed_forward_output_norm", width, eps)?,
            ))
        },
    )
}

pub type Gemma2 = Model<Gemma2Layers>;

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "gemma2",
    formats: &[(Format::Transformers, &["gemma2"])],
    loaders: &[HeadLoader::CausalLm(|loader| Ok(Box::new(load(loader)?)))],
};

pub fn load(loader: &mut Loader<'_>) -> Result<Gemma2> {
    let config: Gemma2Config = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| layers(schema, &config))?;
    loader.load_whole(&model, &transformers::rules())
}
