//! ESM-2: a pre-norm encoder with rotary positions over protein sequences, whose embeddings drop masked tokens and scale the rest by how many were masked, as its training did.

mod transformers;

use serde::{Deserialize, Serialize};
use singe::{
    DataType, Error as SingeError, Result as SingeResult, Tensor,
    graph::capture,
    nn::{
        Activation, DenseAttentionInput, Layer, LayerNorm, ModelState, RotaryFrequencies,
        RotaryLayout, RotarySpec, RotaryTable, StateKey, TokenBatch,
    },
};

use crate::{
    assembly::{PreNormEncoderBlock, PreNormEncoderConfig, pre_norm_encoder_block},
    head::{EncoderBody, EncoderSizes, encoder_heads},
    model::ModelEntry,
    schema::Schema,
    source::Format,
};

/// The share of tokens ESM's training masked, `0.15 · 0.8`, which token dropout scales embeddings for.
const TRAINING_MASK_RATIO: f64 = 0.15 * 0.8;

#[derive(Debug, Clone, Copy)]
pub struct Esm;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EsmConfig {
    pub vocab_size: i64,
    pub width: i64,
    pub layers: i64,
    pub heads: i64,
    pub inner: i64,
    pub activation: Activation,
    pub eps: f32,
    pub rotary_base: f64,
    /// The mask token, whose embeddings token dropout zeroes, when the model drops them.
    pub dropped_token: Option<i64>,
}

/// ESM-2's encoder, from tokens to each token's last hidden state, `[B, T, W]`.
#[derive(Debug, Clone)]
pub struct EsmEncoder {
    tokens: StateKey,
    blocks: Vec<PreNormEncoderBlock>,
    norm: LayerNorm,
    config: EsmConfig,
}

impl EsmEncoder {
    pub fn create(schema: &mut Schema<'_>, config: &EsmConfig) -> SingeResult<Self> {
        let block = PreNormEncoderConfig {
            width: config.width,
            heads: config.heads,
            inner: config.inner,
            activation: config.activation,
            eps: config.eps,
            projection_biases: [true; 4],
            causal: false,
            layer_scale: false,
        };
        let blocks = (0..config.layers)
            .map(|index| {
                pre_norm_encoder_block(&mut schema.scope(format!("blocks.{index}")), &block)
            })
            .collect::<SingeResult<Vec<_>>>()?;
        Ok(Self {
            tokens: schema.parameter(
                "embeddings.tokens.weight",
                [config.vocab_size, config.width],
            )?,
            blocks,
            norm: schema.layer_norm("norm", config.width, config.eps)?,
            config: config.clone(),
        })
    }

    /// Returns the embeddings of `input`, with masked tokens zeroed and every embedding scaled by `0.88 / (1 − masked / length)` when the model drops them, then zeroed past each sequence.
    fn embed(&self, input: &TokenBatch, state: &ModelState) -> SingeResult<Tensor> {
        let tokens = input.tokens();
        let table = state.tensor(&self.tokens)?;
        let mut hidden = table
            .index_select(0, &tokens.flatten(0, 1)?)?
            .reshape_with(&[
                tokens.dimension(0)?,
                tokens.dimension(1)?,
                self.config.width.into(),
            ])?;
        let data_type = hidden.data_type();
        if let Some(dropped) = self.config.dropped_token {
            let device = tokens.device();
            let dropped = capture::scalar(device, dropped)?;
            let is_dropped = tokens.equal(&dropped)?;
            let kept = is_dropped.logical_not()?.cast(data_type)?.unsqueeze(2)?;
            let share = is_dropped
                .cast(DataType::F32)?
                .sum(&[1], false)?
                .div(&input.lengths().cast(DataType::F32)?)?;
            let scale = share
                .rsub_scalar(1.0)?
                .unsqueeze(1)?
                .unsqueeze(2)?
                .cast(data_type)?;
            hidden = hidden
                .mul(&kept)?
                .mul_scalar(1.0 - TRAINING_MASK_RATIO)?
                .div(&scale)?;
        }
        hidden.mul(&input.mask()?.cast(data_type)?.unsqueeze(2)?)
    }
}

impl Layer<TokenBatch> for EsmEncoder {
    fn forward(&self, input: &TokenBatch, state: &ModelState) -> SingeResult<Tensor> {
        let mut hidden = self.embed(input, state)?;
        let &[_, length] = input.tokens().shape().dimensions() else {
            return Err(SingeError::invalid_argument("ESM tokens, [B, T]"));
        };
        let device = hidden.device();
        let positions = Tensor::range(device, 0_i64, length)?;
        let angles = RotaryTable::create(
            device,
            hidden.data_type(),
            RotarySpec {
                head_dim: self.config.width / self.config.heads,
                frequencies: RotaryFrequencies::new(self.config.rotary_base),
                layout: RotaryLayout::HalfSplit,
            },
            length,
        )?
        .angles(&positions)?;
        let key_lengths = input.key_lengths()?;
        for block in &self.blocks {
            hidden = block.forward(
                &DenseAttentionInput {
                    hidden: &hidden,
                    angles: Some(&angles),
                    key_lengths: Some(&key_lengths),
                    mask: None,
                },
                state,
            )?;
        }
        hidden.apply(&self.norm, state)
    }
}

/// Creates ESM-2's encoder as the body heads sit on.
pub fn body(schema: &mut Schema<'_>, config: &EsmConfig) -> SingeResult<EncoderBody> {
    let encoder = EsmEncoder::create(schema, config)?;
    Ok(EncoderBody {
        sizes: EncoderSizes {
            width: config.width,
            vocab_size: config.vocab_size,
            eps: config.eps,
            activation: config.activation,
        },
        tokens: encoder.tokens.clone(),
        layers: Box::new(encoder),
    })
}

pub(crate) const MODEL: ModelEntry = ModelEntry {
    name: "esm",
    formats: &[(Format::Transformers, &["esm"])],
    loaders: &encoder_heads::<Esm>(),
};
