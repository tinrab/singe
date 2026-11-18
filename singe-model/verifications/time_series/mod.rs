// Verifications of the models under `model::time_series`.

#[path = "timesfm2_5/verify.rs"]
mod timesfm2_5;

use crate::support::Verification;

pub const VERIFICATIONS: &[Verification] = &[Verification {
    name: "time_series/timesfm2_5",
    verify: timesfm2_5::verify,
}];
