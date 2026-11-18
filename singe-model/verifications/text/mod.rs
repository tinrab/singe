// Verifications of the models under `model::text`.

#[path = "bert/verify.rs"]
mod bert;
#[path = "distilbert/verify.rs"]
mod distilbert;
#[path = "gpt2/verify.rs"]
mod gpt2;
#[path = "llama/verify.rs"]
mod llama;
#[path = "qwen3/verify.rs"]
mod qwen3;
#[path = "qwen3_5/verify.rs"]
mod qwen3_5;
#[path = "qwen3_embedding/verify.rs"]
mod qwen3_embedding;
#[path = "roberta/verify.rs"]
mod roberta;
#[path = "t5/verify.rs"]
mod t5;
#[path = "tapex/verify.rs"]
mod tapex;
#[path = "xlm_roberta/verify.rs"]
mod xlm_roberta;

use crate::support::Verification;

pub const VERIFICATIONS: &[Verification] = &[
    Verification {
        name: "text/bert",
        verify: bert::verify,
    },
    Verification {
        name: "text/distilbert",
        verify: distilbert::verify,
    },
    Verification {
        name: "text/gpt2",
        verify: gpt2::verify,
    },
    Verification {
        name: "text/llama",
        verify: llama::verify,
    },
    Verification {
        name: "text/qwen3",
        verify: qwen3::verify,
    },
    Verification {
        name: "text/qwen3_5",
        verify: qwen3_5::verify,
    },
    Verification {
        name: "text/qwen3_embedding",
        verify: qwen3_embedding::verify,
    },
    Verification {
        name: "text/roberta",
        verify: roberta::verify,
    },
    Verification {
        name: "text/t5",
        verify: t5::verify,
    },
    Verification {
        name: "text/tapex",
        verify: tapex::verify,
    },
    Verification {
        name: "text/xlm_roberta",
        verify: xlm_roberta::verify,
    },
];
