use std::env::{self, VarError};

use serde::{Deserialize, Serialize};

use crate::{error::Result, frontend::error::Error as FrontendError, utility::to_usize};

const DP_WORKSPACE_LIMIT_ENV: &str = "CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT";

/// Optional cap on the probability-gradient workspace used by floating attention backward.
/// On Hopper, a deterministic request without a bias gradient uses ordered dQ accumulation
/// when this cap is smaller than the padded dP requirement.
/// This cap is independent of the total allocation limit in `CompileConfig`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttentionBackwardWorkspaceLimit {
    /// Does not cap probability-gradient workspace independently of total compile limits.
    Unlimited,
    /// Caps probability-gradient workspace at the supplied byte count; zero requests avoiding that workspace where supported.
    Bytes(usize),
}

impl AttentionBackwardWorkspaceLimit {
    pub(crate) fn from_environment() -> Result<Option<Self>> {
        let value = match env::var(DP_WORKSPACE_LIMIT_ENV) {
            Ok(value) => value,
            Err(VarError::NotPresent) => return Ok(None),
            Err(VarError::NotUnicode(value)) => {
                return Err(FrontendError::SdpaWorkspaceLimitInvalid {
                    value: value.to_string_lossy().into_owned(),
                }
                .into());
            }
        };
        let limit = match value.trim().parse::<i64>() {
            Ok(-1) => Self::Unlimited,
            Ok(bytes) if bytes >= 0 => Self::Bytes(to_usize(
                bytes,
                "sdpa probability gradient workspace limit",
            )?),
            _ => return Err(FrontendError::SdpaWorkspaceLimitInvalid { value }.into()),
        };
        Ok(Some(limit))
    }
}
