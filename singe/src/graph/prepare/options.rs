#[cfg(feature = "cuda")]
use crate::graph::GraphCudaOptions;
use crate::{
    device::Device,
    graph::{
        ComputePrecision, Determinism, GraphCompileOptions, NumericalPolicy,
        prepare::{GraphExecutionMode, GraphPrepareOptions},
    },
    state::StateFreezePolicy,
    tensor::Tensor,
};

impl GraphPrepareOptions {
    /// Creates preparation settings for `device`, using compiled mode, cache capacity eight, no additional state freezing, and default compilation and numerical policies.
    #[must_use]
    pub fn new(device: Device) -> Self {
        Self {
            device,
            mode: GraphExecutionMode::Compiled,
            compile: GraphCompileOptions::new(),
            freeze: StateFreezePolicy::None,
            cache_capacity: 8,
            numerics: NumericalPolicy::default(),
            #[cfg(feature = "cuda")]
            cuda: GraphCudaOptions::new(),
        }
    }

    /// Selects eager dispatch or native compiled replay.
    #[must_use]
    pub fn with_execution_mode(mut self, mode: GraphExecutionMode) -> Self {
        self.mode = mode;
        self
    }

    /// Returns the requested eager or compiled execution mode.
    #[must_use]
    pub fn execution_mode(&self) -> GraphExecutionMode {
        self.mode
    }

    /// Sets the numerical compute-precision policy.
    #[must_use]
    pub fn with_compute_precision(mut self, precision: ComputePrecision) -> Self {
        self.numerics.precision = precision;
        self
    }

    /// Returns the numerical compute-precision policy.
    #[must_use]
    pub fn compute_precision(&self) -> ComputePrecision {
        self.numerics.precision
    }

    /// Sets the requested determinism policy.
    #[must_use]
    pub fn with_determinism(mut self, determinism: Determinism) -> Self {
        self.numerics.determinism = determinism;
        self
    }

    /// Returns the requested determinism policy.
    #[must_use]
    pub fn determinism(&self) -> Determinism {
        self.numerics.determinism
    }

    /// Sets graph compilation and rewrite options.
    #[must_use]
    pub fn with_compile_options(mut self, compile: GraphCompileOptions) -> Self {
        self.compile = compile;
        self
    }

    /// Returns the target execution device.
    #[must_use]
    pub fn device(&self) -> Device {
        self.device
    }

    /// Maximum number of concrete specializations, including the active plan.
    /// Preparation rejects zero capacity.
    /// Older inactive plans are evicted first.
    #[must_use]
    pub fn with_cache_capacity(mut self, capacity: usize) -> Self {
        self.cache_capacity = capacity;
        self
    }

    /// Sets which bound state roles may become constants during preparation.
    #[must_use]
    pub fn with_freeze_policy(mut self, freeze: StateFreezePolicy) -> Self {
        self.freeze = freeze;
        self
    }

    /// Sets CUDA lowering, dynamic-shape, and graph-recording policies.
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn with_cuda_options(mut self, cuda: GraphCudaOptions) -> Self {
        // A record may run sibling matmuls as one, which compiling does before any lowering reads records.
        if cuda.merges_projections() {
            self.compile = self.compile.with_merged_projections(true);
        }

        self.cuda = cuda;
        self
    }

    // Reads the runtime inputs named in `inputs` where those tensors are, on a backend that would otherwise copy them on each run.
    // Runs must pass the same tensors, whose contents may change between runs.
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn with_inputs_in_place(
        mut self,
        inputs: impl IntoIterator<Item = (String, Tensor)>,
    ) -> Self {
        self.cuda = self.cuda.with_inputs_in_place(inputs);
        self
    }

    // Reads the runtime inputs named in `inputs` where those tensors are, which is how every backend without CUDA reads them.
    #[cfg(not(feature = "cuda"))]
    #[must_use]
    pub fn with_inputs_in_place(self, _inputs: impl IntoIterator<Item = (String, Tensor)>) -> Self {
        self
    }
}
