//! Gemma 2: four norms per block, a scaled embedding, a norm that scales by one plus its weight, sliding-window attention in alternate blocks, and capped attention scores and logits.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Device, Result as SingeResult, Tensor,
    nn::{
        CacheSize, DecodeStep, Decoder, DecoderModel, GatedMlp, GroupedQueryAttention, KvCache,
        Model, ModelState, OffsetRmsNorm, RotaryTable,
    },
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

/// Gemma 2's layers: the shared decoder, to which their [`DecoderModel`] implementation passes every call, and the cap on the logits that decoder returns.
#[derive(Debug)]
pub struct Gemma2Layers {
    decoder: Decoder<GroupedQueryAttention, GatedMlp, OffsetRmsNorm>,
    final_cap: Option<f32>,
}

impl DecoderModel for Gemma2Layers {
    type Cache = KvCache;

    type Tables = RotaryTable;

    fn tables(
        &self,
        data_type: DataType,
        device: Device,
        capacity: i64,
    ) -> SingeResult<RotaryTable> {
        self.decoder.tables(data_type, device, capacity)
    }

    fn caches(
        &self,
        data_type: DataType,
        device: Device,
        size: CacheSize,
    ) -> SingeResult<Vec<KvCache>> {
        self.decoder.caches(data_type, device, size)
    }

    fn forward(
        &self,
        step: &DecodeStep,
        tables: &RotaryTable,
        caches: &[KvCache],
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let logits = self.decoder.forward(step, tables, caches, state)?;
        match self.final_cap {
            Some(cap) => logits.mul_scalar(cap.recip())?.tanh()?.mul_scalar(cap),
            None => Ok(logits),
        }
    }
}

impl Gemma2Layers {
    pub fn create(schema: &mut Schema<'_>, config: &Gemma2Config) -> SingeResult<Self> {
        let decoder = &config.decoder;
        let (width, eps) = (decoder.width, decoder.eps);
        let options = AttentionOptions {
            scale: Some(config.attention_scale),
            softcap: config.attention_softcap,
            ..causal()
        };
        let layers = causal_lm(
            schema,
            decoder,
            CausalLmSpec {
                embedding_scale: Some((width as f64).sqrt() as f32),
                ..CausalLmSpec::new(decoder.rotary(), |_| decoder.kv_cache())
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
        )?;
        Ok(Self {
            decoder: layers,
            final_cap: config.logit_softcap,
        })
    }
}

pub type Gemma2 = Model<Gemma2Layers>;

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "gemma2",
    formats: &[(Format::Transformers, &["gemma2"])],
    loaders: &[HeadLoader::CausalLm(|loader| Ok(Box::new(load(loader)?)))],
};

pub fn load(loader: &mut Loader<'_>) -> Result<Gemma2> {
    let config: Gemma2Config = loader.config(transformers::config)?;
    let model = Schema::model(loader.options().data_type, |schema| {
        Gemma2Layers::create(schema, &config)
    })?;
    loader.load_whole(&model, &transformers::rules())
}
