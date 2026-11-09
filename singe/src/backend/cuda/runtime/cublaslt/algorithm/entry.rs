use singe_cublas::{lt::matmul::MatmulAlgorithm, types::ComputeType};

use crate::backend::cuda::runtime::CublasLtMatmulAlgorithmFingerprint;

/// A cached cuBLASLt algorithm with its compute type, workspace requirement, and selection metadata.
#[derive(Debug, Clone)]
pub struct CublasLtMatmulAlgorithmEntry {
    algorithm: MatmulAlgorithm,
    compute_type: ComputeType,
    workspace_bytes: usize,
    waves_count: f32,
    supports_strided_batch: bool,
    supports_pointer_array_batch: bool,
    fingerprint: Option<CublasLtMatmulAlgorithmFingerprint>,
}

impl CublasLtMatmulAlgorithmEntry {
    #[cfg(test)]
    #[must_use]
    pub fn new(
        algorithm: MatmulAlgorithm,
        compute_type: ComputeType,
        workspace_bytes: usize,
        waves_count: f32,
        supports_strided_batch: bool,
        supports_pointer_array_batch: bool,
    ) -> Self {
        Self {
            algorithm,
            compute_type,
            workspace_bytes,
            waves_count,
            supports_strided_batch,
            supports_pointer_array_batch,
            fingerprint: None,
        }
    }

    /// Stores an algorithm and its resource, batch-support, and fingerprint metadata.
    /// The supplied metadata is retained without a native compatibility check.
    #[must_use]
    pub fn new_with_fingerprint(
        algorithm: MatmulAlgorithm,
        compute_type: ComputeType,
        workspace_bytes: usize,
        waves_count: f32,
        supports_strided_batch: bool,
        supports_pointer_array_batch: bool,
        fingerprint: CublasLtMatmulAlgorithmFingerprint,
    ) -> Self {
        Self {
            algorithm,
            compute_type,
            workspace_bytes,
            waves_count,
            supports_strided_batch,
            supports_pointer_array_batch,
            fingerprint: Some(fingerprint),
        }
    }

    /// Returns the retained native algorithm configuration.
    #[must_use]
    pub fn algorithm(&self) -> &MatmulAlgorithm {
        &self.algorithm
    }

    /// Returns the compute type associated with this algorithm.
    #[must_use]
    pub fn compute_type(&self) -> ComputeType {
        self.compute_type
    }

    /// Returns the required workspace capacity in bytes.
    #[must_use]
    pub fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    /// Returns the device utilization estimate supplied for this algorithm.
    #[must_use]
    pub fn waves_count(&self) -> f32 {
        self.waves_count
    }

    /// Returns whether the algorithm requires a nonzero workspace capacity.
    #[must_use]
    pub fn requires_workspace(&self) -> bool {
        self.workspace_bytes != 0
    }

    /// Returns the recorded support flag for strided matrix batches.
    #[must_use]
    pub fn supports_strided_batch(&self) -> bool {
        self.supports_strided_batch
    }

    /// Returns the recorded support flag for batches described by device pointer arrays.
    #[must_use]
    pub fn supports_pointer_array_batch(&self) -> bool {
        self.supports_pointer_array_batch
    }

    /// Returns the recorded algorithm fingerprint, if supplied.
    #[must_use]
    pub fn fingerprint(&self) -> Option<CublasLtMatmulAlgorithmFingerprint> {
        self.fingerprint
    }

    /// Returns whether this entry has a lower wave count, or an equal wave count and fewer workspace bytes.
    /// The comparison uses exact floating-point equality and does not use measured execution time.
    #[must_use]
    #[expect(
        clippy::float_cmp,
        reason = "exact equality selects an identity, specialization, or deterministic tie; approximate equality changes semantics"
    )]
    pub fn is_preferred_to(&self, other: &Self) -> bool {
        self.waves_count < other.waves_count
            || (self.waves_count == other.waves_count
                && self.workspace_bytes < other.workspace_bytes)
    }
}
