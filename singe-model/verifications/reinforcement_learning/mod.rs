// Verifications of the models under `model::reinforcement_learning`.

#[path = "decision_transformer/verify.rs"]
mod decision_transformer;

use crate::support::Verification;

pub const VERIFICATIONS: &[Verification] = &[Verification {
    name: "reinforcement_learning/decision_transformer",
    verify: decision_transformer::verify,
}];
