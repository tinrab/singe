//! The encoder block that norms before attention and before the feed-forward, which Whisper, Qwen3-ASR's audio encoder, the vision transformers, and CLIP repeat, under Singe's names `attention_norm`, `attention` (`query`, `key`, `value`, `output`), `feed_forward_norm`, and `feed_forward` (`up`, `down`).
//! It also holds a stack of these blocks, and the rules that import `transformers`' layouts of them.

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult, Tensor,
    nn::{
        Activation, DenseAttentionInput, Layer, LayerNorm, LayerScale, Mlp, ModelState,
        PreNormBlock, SelfAttention,
    },
};

use crate::{
    format::{RuleSet, join},
    schema::Schema,
};

pub type PreNormEncoderBlock = PreNormBlock<SelfAttention, Mlp, LayerNorm>;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PreNormEncoderConfig {
    pub width: i64,
    pub heads: i64,
    pub inner: i64,
    pub activation: Activation,
    pub eps: f32,
    pub projection_biases: [bool; 4],
    /// Whether each token attends only to those before it, as CLIP's text encoder does.
    pub causal: bool,
    pub layer_scale: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinalNorm {
    Included,
    Omitted,
}

pub fn pre_norm_encoder_block(
    schema: &mut Schema<'_>,
    config: &PreNormEncoderConfig,
) -> SingeResult<PreNormEncoderBlock> {
    let width = config.width;
    let [query, key, value, output] = schema
        .scope("attention")
        .attention_projections(width, config.projection_biases)?;
    let mut attention = SelfAttention::create(query, key, value, output, config.heads, width)?;
    if config.causal {
        attention = attention.causal();
    }

    let feed_forward = schema.mlp(
        "feed_forward",
        [width, config.inner, width],
        config.activation,
        true,
    )?;
    let block = PreNormBlock::new(
        schema.layer_norm("attention_norm", width, config.eps)?,
        attention,
        schema.layer_norm("feed_forward_norm", width, config.eps)?,
        feed_forward,
    );
    if !config.layer_scale {
        return Ok(block);
    }

    Ok(block.with_layer_scales(
        LayerScale::new(schema.parameter("attention_scale.scale", [width])?),
        LayerScale::new(schema.parameter("feed_forward_scale.scale", [width])?),
    ))
}

/// A stack of pre-norm blocks, `blocks.N`, and, where the model has one, a final `norm`, from tokens `[B, T, W]` to their last hidden states.
#[derive(Debug, Clone)]
pub struct PreNormEncoder {
    blocks: Vec<PreNormEncoderBlock>,
    norm: Option<LayerNorm>,
}

impl PreNormEncoder {
    pub fn create(
        schema: &mut Schema<'_>,
        config: &PreNormEncoderConfig,
        layers: i64,
        final_norm: FinalNorm,
    ) -> SingeResult<Self> {
        let blocks = (0..layers)
            .map(|layer| {
                pre_norm_encoder_block(&mut schema.scope(format!("blocks.{layer}")), config)
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let norm = match final_norm {
            FinalNorm::Included => Some(schema.layer_norm("norm", config.width, config.eps)?),
            FinalNorm::Omitted => None,
        };
        Ok(Self { blocks, norm })
    }

    /// Runs the blocks over `hidden`, each sequence attending to its first `key_lengths` tokens when given.
    pub fn encode(
        &self,
        hidden: &Tensor,
        key_lengths: Option<&Tensor>,
        state: &ModelState,
    ) -> SingeResult<Tensor> {
        let mut hidden = hidden.clone();
        for block in &self.blocks {
            hidden = run_block(block, &hidden, key_lengths, state)?;
        }
        self.normed(hidden, state)
    }

    pub fn features(
        &self,
        hidden: &Tensor,
        stages: &[usize],
        state: &ModelState,
    ) -> SingeResult<Vec<Tensor>> {
        let mut hidden = hidden.clone();
        let mut features = Vec::with_capacity(stages.len());
        for (index, block) in self.blocks.iter().enumerate() {
            hidden = run_block(block, &hidden, None, state)?;
            if stages.contains(&(index + 1)) {
                features.push(self.normed(hidden.clone(), state)?);
            }
        }
        Ok(features)
    }

    fn normed(&self, hidden: Tensor, state: &ModelState) -> SingeResult<Tensor> {
        match &self.norm {
            Some(norm) => hidden.apply(norm, state),
            None => Ok(hidden),
        }
    }
}

fn run_block(
    block: &PreNormEncoderBlock,
    hidden: &Tensor,
    key_lengths: Option<&Tensor>,
    state: &ModelState,
) -> SingeResult<Tensor> {
    block.forward(
        &DenseAttentionInput {
            hidden,
            angles: None,
            key_lengths,
            mask: None,
        },
        state,
    )
}

impl Layer for PreNormEncoder {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        self.encode(input, None, state)
    }
}

/// Imports `transformers`' Whisper-style layers under `from` (such as `model.encoder.layers`) into encoder blocks under `to` (such as `blocks`).
#[must_use]
pub fn pre_norm_encoder_rules(from: &str, to: &str) -> RuleSet {
    RuleSet::new().rename_each(
        &join(from, "*"),
        &join(to, "*"),
        &[
            ("self_attn_layer_norm", "attention_norm"),
            ("self_attn.q_proj", "attention.query"),
            ("self_attn.k_proj", "attention.key"),
            ("self_attn.v_proj", "attention.value"),
            ("self_attn.out_proj", "attention.output"),
            ("final_layer_norm", "feed_forward_norm"),
            ("fc1", "feed_forward.up"),
            ("fc2", "feed_forward.down"),
        ],
    )
}

/// Imports `transformers`' `ViT`-style layers under `from` (such as `vit.encoder.layer`), which `ViT`, YOLOS, and `VideoMAE` store, into encoder blocks under `to`.
#[must_use]
pub fn vit_layer_rules(from: &str, to: &str) -> RuleSet {
    RuleSet::new().rename_each(
        &join(from, "*"),
        &join(to, "*"),
        &[
            ("layernorm_before", "attention_norm"),
            ("attention.attention.query", "attention.query"),
            ("attention.attention.key", "attention.key"),
            ("attention.attention.value", "attention.value"),
            ("attention.output.dense", "attention.output"),
            ("layernorm_after", "feed_forward_norm"),
            ("intermediate.dense", "feed_forward.up"),
            ("output.dense", "feed_forward.down"),
        ],
    )
}

/// Imports `transformers`' CLIP-style layers under `from` (such as `text_model.encoder.layers`), which CLIP's and `OWL-ViT`'s towers store, into encoder blocks under `to`.
#[must_use]
pub fn clip_layer_rules(from: &str, to: &str) -> RuleSet {
    RuleSet::new().rename_each(
        &join(from, "*"),
        &join(to, "*"),
        &[
            ("layer_norm1", "attention_norm"),
            ("self_attn.q_proj", "attention.query"),
            ("self_attn.k_proj", "attention.key"),
            ("self_attn.v_proj", "attention.value"),
            ("self_attn.out_proj", "attention.output"),
            ("layer_norm2", "feed_forward_norm"),
            ("mlp.fc1", "feed_forward.up"),
            ("mlp.fc2", "feed_forward.down"),
        ],
    )
}
