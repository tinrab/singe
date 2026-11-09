use std::sync::atomic::{AtomicU64, Ordering};

use crate::activity;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Stable identity for a tensor allocation/view handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(u64);

impl TensorId {
    pub(crate) fn generate() -> Self {
        activity::descriptor();
        Self(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }
}
