// Verifications of the models under `model::multimodal`.

#[path = "clip/verify.rs"]
mod clip;
#[path = "colmodernvbert/verify.rs"]
mod colmodernvbert;
#[path = "donut/verify.rs"]
mod donut;
#[path = "owlvit/verify.rs"]
mod owlvit;
#[path = "smolvlm/verify.rs"]
mod smolvlm;
#[path = "trocr/verify.rs"]
mod trocr;

use crate::support::Verification;

pub const VERIFICATIONS: &[Verification] = &[
    Verification {
        name: "multimodal/clip",
        verify: clip::verify,
    },
    Verification {
        name: "multimodal/colmodernvbert",
        verify: colmodernvbert::verify,
    },
    Verification {
        name: "multimodal/donut",
        verify: donut::verify,
    },
    Verification {
        name: "multimodal/owlvit",
        verify: owlvit::verify,
    },
    Verification {
        name: "multimodal/smolvlm",
        verify: smolvlm::verify,
    },
    Verification {
        name: "multimodal/trocr",
        verify: trocr::verify,
    },
];
