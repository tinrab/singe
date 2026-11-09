use crate::backend::cuda::runtime::{CublasLtMatmulAlgorithmEntry, CublasLtMatmulAlgorithmKey};

/// Returns the candidate with the lowest wave count, breaking equal-wave ties with fewer workspace bytes.
/// Returns `None` for an empty iterator; equally preferred entries retain iterator order.
pub fn best_cublaslt_matmul_algorithm_entry<'a>(
    entries: impl IntoIterator<Item = &'a CublasLtMatmulAlgorithmEntry>,
) -> Option<&'a CublasLtMatmulAlgorithmEntry> {
    entries.into_iter().fold(None, |best, entry| match best {
        Some(best) if !entry.is_preferred_to(best) => Some(best),
        _ => Some(entry),
    })
}

/// Selects the preferred entry among keys whose recorded workspace limit does not exceed `workspace_limit_bytes`.
/// Preference uses wave count, then the entry's required workspace bytes.
/// Returns `None` if no key qualifies.
pub fn best_cublaslt_matmul_algorithm_entry_with_workspace_limit<'a>(
    entries: impl IntoIterator<
        Item = (
            &'a CublasLtMatmulAlgorithmKey,
            &'a CublasLtMatmulAlgorithmEntry,
        ),
    >,
    workspace_limit_bytes: usize,
) -> Option<&'a CublasLtMatmulAlgorithmEntry> {
    best_cublaslt_matmul_algorithm_entry(
        entries
            .into_iter()
            .filter(|(key, _)| key.workspace_limit_bytes() <= workspace_limit_bytes)
            .map(|(_, entry)| entry),
    )
}
