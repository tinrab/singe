use singe::{
    Result as SingeResult,
    nn::{Activation, LayerNorm, Mlp, PreNormBlock, SelfAttention},
};

use crate::schema::Schema;

pub type EncoderBlock = PreNormBlock<SelfAttention, Mlp, LayerNorm>;

const LAYER_NORM_EPS: f32 = 1e-5;

#[derive(Debug, Clone, Copy)]
pub struct EncoderSpec {
    pub width: i64,
    pub heads: i64,
    pub inner: i64,
    pub activation: Activation,
    pub key_bias: bool,
}

pub fn encoder_block(schema: &mut Schema<'_>, spec: &EncoderSpec) -> SingeResult<EncoderBlock> {
    // Whisper's encoder block, which Qwen3-ASR's audio encoder repeats: every projection has a bias, but Whisper's keys.
    let width = spec.width;
    let mut attention = schema.scope("self_attn");
    let attention = SelfAttention::create(
        attention.linear("q_proj", width, width, true)?,
        attention.linear("k_proj", width, width, spec.key_bias)?,
        attention.linear("v_proj", width, width, true)?,
        attention.linear("out_proj", width, width, true)?,
        spec.heads,
        width,
    )?;
    Ok(PreNormBlock::new(
        schema.layer_norm("self_attn_layer_norm", width, LAYER_NORM_EPS)?,
        attention,
        schema.layer_norm("final_layer_norm", width, LAYER_NORM_EPS)?,
        Mlp::new(
            schema.linear("fc1", width, spec.inner, true)?,
            schema.linear("fc2", spec.inner, width, true)?,
            spec.activation,
        ),
    ))
}
