// Verifications of the models under `model::text`.

mod bert;
mod big_bird;
mod bitnet;
mod deberta_v2;
mod deepseek_v3;
mod distilbert;
mod esm;
mod falcon_h1;
mod gemma2;
mod gpt2;
mod granite_moe_hybrid;
mod llama;
mod longformer;
mod longt5;
mod mamba;
mod qwen3;
mod qwen3_5;
mod qwen3_embedding;
mod recurrent_gemma;
mod roberta;
mod switch_transformers;
mod t5;
mod tapex;
mod xlm_roberta;

use crate::support::verifications;

verifications!(
    "text",
    [
        bert,
        big_bird,
        bitnet,
        deberta_v2,
        deepseek_v3,
        distilbert,
        esm,
        falcon_h1,
        gemma2,
        gpt2,
        granite_moe_hybrid,
        llama,
        longformer,
        longt5,
        mamba,
        qwen3,
        qwen3_5,
        recurrent_gemma,
        qwen3_embedding,
        roberta,
        switch_transformers,
        t5,
        tapex,
        xlm_roberta,
    ]
);
