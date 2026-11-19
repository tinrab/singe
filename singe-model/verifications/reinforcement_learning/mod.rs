// Verifications of the models under `model::reinforcement_learning`.

mod decision_transformer;

use crate::support::verifications;

verifications!("reinforcement_learning", [decision_transformer]);
