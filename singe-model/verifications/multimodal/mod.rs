// Verifications of the models under `model::multimodal`.

mod clip;
mod colmodernvbert;
mod donut;
mod florence2;
mod grounding_dino;
mod janus;
mod owlvit;
mod siglip;
mod smolvlm;
mod trocr;

use crate::support::verifications;

verifications!(
    "multimodal",
    [
        clip,
        colmodernvbert,
        donut,
        florence2,
        grounding_dino,
        janus,
        owlvit,
        siglip,
        smolvlm,
        trocr
    ]
);
