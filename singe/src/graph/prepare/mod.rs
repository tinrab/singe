mod cache;
#[cfg(feature = "cuda")]
mod cuda;
mod entry;
mod executor;
mod memory;
mod options;
mod plan;

use std::time::Duration;

#[cfg(feature = "cuda")]
use crate::backend::cuda::{CudaCompiledGraph, CudaStream, compiled::plan::memory::GraphArena};
use crate::{
    device::Device,
    graph::{
        CompiledGraph, Graph, GraphCompileOptions, GraphProgram, GraphRunStats,
        GraphSpecializationKey, NumericalPolicy, StateFreezePolicy, StateSnapshot,
    },
};

use self::cache::{CachedPlan, RuntimeMetadataKey};
#[cfg(feature = "cuda")]
pub use self::cuda::GraphCudaOptions;
pub use self::{cache::GraphExecutionCacheStats, memory::GraphExecutionMemoryStats};

/// Execution mechanism selected during preparation.
/// CUDA eager execution dispatches operations through the same checked IR.
/// Compiled execution requires native replay support and never falls back.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum GraphExecutionMode {
    /// Dispatches checked graph operations eagerly on the selected device.
    Eager,
    /// Requests native replay on CUDA; CPU execution uses the analyzed graph.
    #[default]
    Compiled,
}

/// Target and graph rewrite policy for reusable execution.
/// CUDA compilation failures are returned to the caller; preparation never falls back to CPU.
#[derive(Debug, Clone)]
pub struct GraphPrepareOptions {
    device: Device,
    mode: GraphExecutionMode,
    compile: GraphCompileOptions,
    freeze: StateFreezePolicy,
    cache_capacity: usize,
    numerics: NumericalPolicy,
    #[cfg(feature = "cuda")]
    cuda: GraphCudaOptions,
}

// Replay buffers several executors share, which must run one at a time; only CUDA has any.
#[derive(Debug, Clone, Default)]
pub struct SharedBuffers {
    #[cfg(feature = "cuda")]
    arena: GraphArena,
}

impl GraphPrepareOptions {
    // Prepares an executor whose buffers are `buffers`.
    #[cfg(feature = "cuda")]
    pub(crate) fn sharing_buffers(mut self, buffers: &SharedBuffers) -> Self {
        self.cuda.shared_arena = Some(buffers.arena.clone());
        self
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn sharing_buffers(self, _buffers: &SharedBuffers) -> Self {
        self
    }
}

/// One execution instance.
/// Analysis and bound tensors are immutable and shared; CUDA replay storage belongs exclusively to this instance.
#[derive(Debug)]
pub struct GraphExecutor {
    program: GraphProgram,
    state: StateSnapshot,
    template: Graph,
    options: GraphPrepareOptions,
    active: PreparedPlan,
    active_key: RuntimeMetadataKey,
    inactive: Vec<CachedPlan>,
    cache_stats: GraphExecutionCacheStats,
    cache_clock: u64,
    preparation_time: Duration,
    runs: u64,
    last_run: Option<GraphRunStats>,
}

/// Immutable preparation shared between workers.
/// Each worker creates its own native execution resources on the thread that will run them.
#[derive(Debug, Clone)]
pub struct GraphExecutionPlan {
    compiled: CompiledGraph,
    active_key: RuntimeMetadataKey,
    options: GraphPrepareOptions,
    program: GraphProgram,
    state: StateSnapshot,
    template: Graph,
}

/// One compiled analysis, its prepared backend resources, and the specialization identity tying them together.
#[derive(Debug)]
pub struct PreparedPlan {
    compiled: CompiledGraph,
    backend: PreparedBackend,
    specialization: GraphSpecializationKey,
}

#[derive(Debug)]
enum PreparedBackend {
    Cpu,
    #[cfg(feature = "cuda")]
    CudaEager {
        stream: CudaStream,
    },
    #[cfg(feature = "cuda")]
    Cuda {
        graph: Box<CudaCompiledGraph>,
        stream: CudaStream,
    },
}
