// Verifications of the models under `model::tabular`.

#[path = "mitra/verify.rs"]
mod mitra;

use crate::support::Verification;

pub const VERIFICATIONS: &[Verification] = &[Verification {
    name: "tabular/mitra",
    verify: mitra::verify,
}];
