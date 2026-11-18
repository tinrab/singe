// Verifications of the models under `model::video`.

#[path = "videomae/verify.rs"]
mod videomae;

use crate::support::Verification;

pub const VERIFICATIONS: &[Verification] = &[Verification {
    name: "video/videomae",
    verify: videomae::verify,
}];
