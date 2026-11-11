use std::path::Path;

use serde::Deserialize;
use singe::{
    DataType, Device, Result as SingeResult, Tensor,
    nn::{
        CacheSize, DecodeStep, Decoder, DecoderModel, GatedMlp, GroupedQueryAttention, KvCache,
        Model, ModelState, OffsetRmsNorm, PreNormBlock, RotaryTable,
    },
    tensor::AttentionOptions,
};

use crate::{
    assembly::{causal, causal_lm, gated_mlp, grouped_attention, load_weights},
    config::{DecoderConfig, Snapshot, unsupported},
    error::Result,
};

// The fields of a Hugging Face `config.json` this definition reads.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Gemma2Config {
    #[serde(flatten)]
    pub decoder: DecoderConfig,
    pub query_pre_attn_scalar: f64,
    #[serde(default)]
    pub sliding_window: Option<i64>,
    #[serde(default)]
    pub attn_logit_softcapping: Option<f32>,
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,
}

impl Gemma2Config {
    // A checkpoint using one of these would load and then compute something else.
    fn validate(&self) -> Result<()> {
        if self
            .decoder
            .layer_types
            .iter()
            .flatten()
            .any(|layer| layer != "full_attention" && layer != "sliding_attention")
        {
            return Err(unsupported("a layer that is neither full nor sliding"));
        }
        Ok(())
    }

    // The window of layer `index`, whose keys are that many tokens up to each query's own.
    fn window(&self, index: usize) -> Option<i64> {
        let layer = self.decoder.layer_types.as_ref()?.get(index)?;
        (layer == "sliding_attention")
            .then_some(self.sliding_window)
            .flatten()
    }
}

// Gemma 2: four norms per block, a scaled embedding, a norm that scales by one plus its weight, and optionally capped logits.
//
// It is a shared decoder except for the cap, so its `DecoderModel` passes to that decoder and caps what comes back.
#[derive(Debug)]
pub struct Gemma2Layers {
    decoder: Decoder<GroupedQueryAttention, GatedMlp, OffsetRmsNorm>,
    final_cap: Option<f32>,
}

pub type Gemma2 = Model<Gemma2Layers>;

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

#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "scales are defined in F32, as the references compute them"
)]
fn schema(config: &Gemma2Config, data_type: DataType) -> Result<Gemma2> {
    let spec = config.decoder.spec(data_type, true)?;
    let (width, eps) = (spec.width, spec.eps);
    let options = AttentionOptions {
        scale: Some(config.query_pre_attn_scalar.sqrt().recip() as f32),
        softcap: config.attn_logit_softcapping,
        ..causal()
    };
    let (decoder, state) = causal_lm(
        &spec,
        spec.rotary(),
        |_| spec.kv_cache(),
        Some((width as f64).sqrt() as f32),
        |schema, index| {
            let attention_norm = schema.offset_rms_norm("input_layernorm", width, eps)?;
            let attention = grouped_attention(
                &mut schema.scope("self_attn"),
                &spec,
                AttentionOptions {
                    window: config.window(index),
                    ..options
                },
            )?;
            let attention_output =
                schema.offset_rms_norm("post_attention_layernorm", width, eps)?;
            let feed_forward_norm =
                schema.offset_rms_norm("pre_feedforward_layernorm", width, eps)?;
            let feed_forward = gated_mlp(schema, &spec)?;
            let feed_forward_output =
                schema.offset_rms_norm("post_feedforward_layernorm", width, eps)?;
            Ok(
                PreNormBlock::new(attention_norm, attention, feed_forward_norm, feed_forward)
                    .with_output_norms(attention_output, feed_forward_output),
            )
        },
    )?;
    Ok(Model::new(
        Gemma2Layers {
            decoder,
            final_cap: config.final_logit_softcapping,
        },
        state,
    ))
}

// Loads a Hugging Face snapshot directory, converting the weights to `data_type`.
pub fn load(snapshot: impl AsRef<Path>, device: Device, data_type: DataType) -> Result<Gemma2> {
    let config: Gemma2Config = Snapshot::new(&snapshot).config()?;
    config.validate()?;
    load_weights(&schema(&config, data_type)?, snapshot, device, None)
}
