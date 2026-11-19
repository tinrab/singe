// Verifications of the models under `model::text`.

mod bert;
mod deepseek_v3;
mod distilbert;
mod gemma2;
mod gpt2;
mod llama;
mod qwen3;
mod qwen3_5;
mod qwen3_embedding;
mod roberta;
mod t5;
mod tapex;
mod xlm_roberta;

use crate::support::verifications;

verifications!(
    "text",
    [
        bert,
        deepseek_v3,
        distilbert,
        gemma2,
        gpt2,
        llama,
        qwen3,
        qwen3_5,
        qwen3_embedding,
        roberta,
        t5,
        tapex,
        xlm_roberta,
    ]
);
