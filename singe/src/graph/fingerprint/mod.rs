mod compute;
mod operation;
mod program;
mod shape;
mod shapes;
mod stable;
mod values;

pub use self::{program::fingerprint_program, stable::StableHasher};
use crate::graph::GraphCompileOptions;

const GRAPH_FINGERPRINT_VERSION: u64 = 3;

/// Stable graph structure fingerprint used as backend-neutral cache-key input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphFingerprint {
    pub(crate) value: u64,
    pub(crate) cacheable: bool,
}

/// Options included in graph fingerprinting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphFingerprintOptions {
    compiler_version: u64,
    compile: GraphCompileOptions,
}

impl GraphFingerprint {
    /// Returns the stable fingerprint value.
    #[must_use]
    pub fn value(self) -> u64 {
        self.value
    }

    /// Returns whether the fingerprint can be used for a cross-graph cache entry.
    #[must_use]
    pub fn is_cacheable(self) -> bool {
        self.cacheable
    }
}

impl GraphFingerprintOptions {
    /// Creates the default graph fingerprint configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_compile_options(GraphCompileOptions::new())
    }

    /// Creates a fingerprint configuration from graph compile options.
    #[must_use]
    pub fn with_compile_options(compile: GraphCompileOptions) -> Self {
        Self {
            compiler_version: GRAPH_FINGERPRINT_VERSION,
            compile,
        }
    }

    /// Returns a copy of this configuration with `compiler_version`.
    #[cfg(test)]
    #[must_use]
    pub fn with_compiler_version(mut self, compiler_version: u64) -> Self {
        self.compiler_version = compiler_version;
        self
    }

    /// Returns the compiler version salt included in the fingerprint.
    #[cfg(test)]
    #[must_use]
    pub fn compiler_version(self) -> u64 {
        self.compiler_version
    }

    /// Returns the compile configuration.
    pub(crate) fn compile(self) -> GraphCompileOptions {
        self.compile
    }
}

impl Default for GraphFingerprintOptions {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
