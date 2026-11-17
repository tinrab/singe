use serde::{Deserialize, Serialize};

/// Optional cap on the probability-gradient workspace used by floating attention backward.
/// On Hopper, a deterministic request without a bias gradient uses ordered dQ accumulation when this cap is smaller than the padded dP requirement.
/// This cap is independent of the total allocation limit in `CompileConfig`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttentionBackwardWorkspaceLimit {
    /// Does not cap probability-gradient workspace independently of total compile limits.
    Unlimited,
    /// Caps probability-gradient workspace at the supplied byte count; zero requests avoiding that workspace where supported.
    Bytes(usize),
}
