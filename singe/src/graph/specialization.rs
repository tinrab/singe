use std::hash::{Hash, Hasher};

use crate::{
    device::Device,
    graph::{
        ComputePrecision, Determinism, GraphExecutionMode, GraphFingerprint, NumericalPolicy,
        ProgramFingerprint, StateFreezePolicy, StateRevision, fingerprint::StableHasher,
    },
};

/// Process-local prepared specialization.
/// Logical identity is separate from immutable state publication, physical guards, native context, and providers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GraphSpecializationKey {
    pub(crate) program: ProgramFingerprint,
    pub(crate) state: StateRevision,
    pub(crate) device: Device,
    pub(crate) execution_mode: GraphExecutionMode,
    pub(crate) numerics: NumericalPolicy,
    pub(crate) freeze: StateFreezePolicy,
    pub(crate) bound_graph: GraphFingerprint,
    pub(crate) native_context: Option<usize>,
    pub(crate) provider_digest: Option<u64>,
}

impl GraphSpecializationKey {
    /// Returns the compute precision policy retained by this specialization.
    #[must_use]
    pub fn compute_precision(&self) -> ComputePrecision {
        self.numerics.precision
    }

    /// Returns the determinism requirement retained by this specialization.
    #[must_use]
    pub fn determinism(&self) -> Determinism {
        self.numerics.determinism
    }

    /// Hashes the complete process-local specialization key into a 64-bit digest.
    /// Native context identity and other process-local fields make this unsuitable as a persistent program identity.
    #[must_use]
    pub fn digest(&self) -> u64 {
        let mut hasher = StableHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    /// Returns the logical program fingerprint.
    #[must_use]
    pub fn program(&self) -> ProgramFingerprint {
        self.program
    }

    /// Returns the immutable state publication revision.
    #[must_use]
    pub fn state(&self) -> StateRevision {
        self.state
    }

    /// Returns the execution device.
    #[must_use]
    pub fn device(&self) -> Device {
        self.device
    }

    /// Returns the process-local native context identity, if preparation recorded one.
    #[must_use]
    pub fn native_context_identity(&self) -> Option<usize> {
        self.native_context
    }
}
