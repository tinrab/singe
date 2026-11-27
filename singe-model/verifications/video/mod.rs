// Verifications of the models under `model::video`.

mod timesformer;
mod videomae;
mod vjepa2;

use crate::support::verifications;

verifications!("video", [timesformer, videomae, vjepa2]);
