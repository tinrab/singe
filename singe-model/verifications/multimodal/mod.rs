// Verifications of the models under `model::multimodal`.

mod clip;
mod colmodernvbert;
mod donut;
mod owlvit;
mod smolvlm;
mod trocr;

use crate::support::verifications;

verifications!(
    "multimodal",
    [clip, colmodernvbert, donut, owlvit, smolvlm, trocr]
);
