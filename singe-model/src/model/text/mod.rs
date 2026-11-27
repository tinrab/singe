//! Models `transformers` documents among its text models.

pub mod bart;
pub mod bert;
pub mod big_bird;
pub mod bitnet;
pub mod deberta_v2;
pub mod deepseek_v3;
pub mod distilbert;
pub mod esm;
pub mod falcon_h1;
pub mod gemma2;
pub mod gpt2;
pub mod granite_moe_hybrid;
pub mod llama;
pub mod longformer;
pub mod mamba;
pub mod qwen3;
pub mod qwen3_5;
pub mod recurrent_gemma;
pub mod roberta;
pub mod t5;

use crate::model::ModelEntry;

pub(crate) const MODELS: &[&ModelEntry] = &[
    &bart::MODEL,
    &bert::MODEL,
    &big_bird::MODEL,
    &bitnet::MODEL,
    &deberta_v2::MODEL,
    &deepseek_v3::MODEL,
    &distilbert::MODEL,
    &esm::MODEL,
    &falcon_h1::MODEL,
    &gemma2::MODEL,
    &gpt2::MODEL,
    &granite_moe_hybrid::MODEL,
    &llama::MODEL,
    &longformer::MODEL,
    &mamba::MODEL,
    &qwen3::MODEL,
    &qwen3_5::MODEL,
    &recurrent_gemma::MODEL,
    &roberta::MODEL,
    &t5::MODEL,
];
