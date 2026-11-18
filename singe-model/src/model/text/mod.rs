// Models `transformers` documents among its text models.

pub mod bart;
pub mod bert;
pub mod deepseek_v3;
pub mod distilbert;
pub mod gemma2;
pub mod gpt2;
pub mod llama;
pub mod qwen3;
pub mod qwen3_5;
pub mod roberta;
pub mod t5;

use super::ModelEntry;

pub(crate) const MODELS: &[&ModelEntry] = &[
    &bart::MODEL,
    &bert::MODEL,
    &deepseek_v3::MODEL,
    &distilbert::MODEL,
    &gemma2::MODEL,
    &gpt2::MODEL,
    &llama::MODEL,
    &qwen3::MODEL,
    &qwen3_5::MODEL,
    &roberta::MODEL,
    &t5::MODEL,
];
