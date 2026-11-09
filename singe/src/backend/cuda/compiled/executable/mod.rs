mod buffers;
mod layout;
mod replay;
mod report;
mod stats;
mod steps;

use std::{
    collections::{HashMap, HashSet},
    fmt,
    sync::{Arc, Mutex},
};

use singe_cuda::graph::{ExecutableGraph as NativeExecutableGraph, Graph as NativeCudaGraph};
use singe_cufft::plan::Plan;

#[cfg(test)]
pub use self::report::GraphCompileInputLayoutRequirement;
pub use self::{
    buffers::{
        GraphOutputBufferAllocation, GraphOutputBufferSummary, GraphReusableBufferAllocation,
        GraphReusableBufferSummary,
    },
    replay::GraphReplayStats,
    report::{GraphCompileProviderDecision, GraphCompileReport},
    stats::CudaCompiledGraphStats,
    steps::{CudnnGraphStep, CutensorGraphStep},
};
use self::{
    buffers::{output_buffer_allocation_records, output_buffer_summaries, output_value_tensors},
    replay::PendingRuns,
    report::build_cuda_graph_compile_report,
};
use crate::{
    backend::cuda::{
        compiled::plan::{
            cache_key::{
                ConcreteShapeBucket, DynamicShapePolicy, GraphCacheKey, SymbolicDimensionKey,
            },
            fusion::FusionPlan,
            lowering::{LoweringPlan, RealizedLoweringPlan},
        },
        cudnn_backend::CudnnBackendExecutable,
        diagnostics::GraphProviderFailures,
        graph_binding::InPlaceWrites,
        graph_value::CudaGraphValue,
        runtime::CudaRuntime,
        tuning::LibraryRecorder,
    },
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    graph::{
        guards::GraphGuards,
        ir::{Graph, ValueId},
    },
    shape::Shape,
    tensor::Tensor,
};

// What a compiled graph keeps alive for the library work its nodes captured: child graphs, and the cuFFT plans whose tables and work areas captured transforms read.
pub enum LibraryResource {
    Graph(NativeCudaGraph),
    FftPlan(Plan),
}

impl fmt::Debug for LibraryResource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Graph(_) => "graph",
            Self::FftPlan(_) => "fft plan",
        })
    }
}

/// Compiled Singe graph lowered to CUDA-native graph work where possible.
#[derive(Debug)]
pub struct CudaCompiledGraph {
    ir: Arc<Graph>,
    cache_key: GraphCacheKey,
    stats: CudaCompiledGraphStats,
    replay_stats: GraphReplayStats,
    report: GraphCompileReport,
    runtime: Arc<CudaRuntime>,
    device: Device,
    guards: GraphGuards,
    execution_order: Vec<ValueId>,
    // The template graph and library child graphs own the nodes the executable was built from.
    _native_graph: Option<NativeCudaGraph>,
    executable: Option<Arc<Mutex<NativeExecutableGraph>>>,
    cudnn_steps: Vec<CudnnGraphStep>,
    cutensor_steps: Vec<CutensorGraphStep>,
    _library_resources: Vec<LibraryResource>,
    pending_runs: PendingRuns,
    values: HashMap<ValueId, CudaGraphValue>,
    outputs: Vec<ValueId>,
    output_buffers: Vec<GraphOutputBufferSummary>,
    output_buffer_allocations: Vec<GraphOutputBufferAllocation>,
    reusable_buffers: Vec<GraphReusableBufferSummary>,
    reusable_buffer_allocations: Vec<GraphReusableBufferAllocation>,
    private: GraphPrivateBuffers,
}

/// Device buffers that a compiled graph's kernels use besides its values, and the runtime inputs it reads in place.
#[derive(Debug, Default)]
pub struct GraphPrivateBuffers {
    /// Runtime inputs the graph reads where they are instead of copying them into placeholders; replays must pass these same tensors.
    pub(crate) inputs_in_place: HashMap<ValueId, Tensor>,
    // What the graph's scatters write in place of fresh buffers, given those inputs.
    pub(crate) in_place_writes: InPlaceWrites,
    // Where lowering lists what libraries could run each node with, for tuning.
    pub(crate) library_recorder: Option<LibraryRecorder>,
    /// The names of the inputs bound in place whose written values the graph itself copies back, as its last nodes.
    pub(crate) inputs_written_in_graph: HashSet<String>,
    // The positions of the outputs that hold those values, which already sit in their inputs and need no copy out.
    pub(crate) outputs_written_in_graph: HashSet<usize>,
    /// The status word checking kernels report data errors through, read after each replay completes.
    pub(crate) status: Option<Tensor>,
    /// Scratch space for kernels such as `argsort`, which replays reuse one after another.
    pub(crate) scratch: Vec<Tensor>,
}

impl GraphPrivateBuffers {
    /// Returns the status word, allocating it on `device` the first time.
    pub(crate) fn status(&mut self, device: Device) -> Result<&Tensor> {
        if self.status.is_none() {
            self.status = Some(Tensor::zeroed_on(
                device,
                DataType::I32,
                Shape::create([1])?,
            )?);
        }
        self.status
            .as_ref()
            .ok_or_else(|| Error::missing_node("status"))
    }
}

impl Drop for CudaCompiledGraph {
    fn drop(&mut self) {
        self.pending_runs.synchronize_before_drop();
    }
}

impl CudaCompiledGraph {
    /// Returns the accumulated byte count of graph tensor allocations.
    #[must_use]
    pub fn graph_allocation_bytes(&self) -> usize {
        self.stats.graph_allocation_bytes()
    }

    /// Returns the cumulative bytes supplied by reused graph buffers.
    #[must_use]
    pub fn reused_buffer_bytes(&self) -> usize {
        self.stats.reused_buffer_bytes()
    }

    /// Returns the number of retained entries in the graph's reusable buffer pool.
    #[must_use]
    pub fn reusable_buffer_pool_entries(&self) -> usize {
        self.stats.reusable_buffer_pool_entries()
    }

    /// Returns the combined recorded workspace byte count for CUDA library operations.
    #[must_use]
    pub fn library_workspace_bytes(&self) -> usize {
        self.stats.library_workspace_bytes()
    }

    /// Returns lowering and memory-planning statistics for this compiled CUDA graph.
    #[must_use]
    pub fn stats(&self) -> CudaCompiledGraphStats {
        debug_assert_eq!(
            self.reusable_buffer_allocations.len(),
            self.stats.reusable_buffer_pool_entries()
        );
        debug_assert!(
            self.reusable_buffer_allocations
                .iter()
                .all(|allocation| allocation.tensor().data_type() == allocation.data_type())
        );
        debug_assert_eq!(
            self.output_buffer_allocations.len(),
            self.output_buffers.len()
        );
        debug_assert!(
            self.output_buffer_allocations
                .iter()
                .zip(&self.output_buffers)
                .all(|(allocation, summary)| {
                    allocation.value() == summary.value()
                        && allocation.data_type() == summary.data_type()
                        && allocation.layout() == summary.layout()
                        && allocation.bytes() == summary.bytes()
                        && allocation
                            .tensor()
                            .is_some_and(|tensor| tensor.data_type() == allocation.data_type())
                })
        );
        self.stats.with_pending_runs(
            self.pending_runs.len(),
            self.pending_runs.held_input_count(),
            self.pending_runs.held_output_count(),
        )
    }

    /// Returns replay statistics accumulated since this CUDA graph was compiled.
    #[must_use]
    pub fn replay_stats(&self) -> GraphReplayStats {
        self.replay_stats
    }
}

impl CudaCompiledGraph {
    pub(crate) fn native_context_identity(&self) -> usize {
        self.runtime.context().as_raw() as usize
    }

    /// Returns the normalized graph IR used to compile this CUDA graph.
    #[must_use]
    pub fn ir(&self) -> &Graph {
        &self.ir
    }

    /// Returns the CUDA cache identity retained when this executable was compiled.
    #[must_use]
    pub fn cache_key(&self) -> &GraphCacheKey {
        &self.cache_key
    }

    /// Returns whether this executable is suitable for CUDA graph cache reuse.
    #[must_use]
    pub fn is_cacheable(&self) -> bool {
        self.cache_key.is_cacheable()
    }

    /// Returns the dynamic-shape cache policy used to compile this executable.
    #[must_use]
    pub fn dynamic_shape_policy(&self) -> &DynamicShapePolicy {
        self.cache_key.dynamic_shape_policy()
    }

    /// Returns symbolic dimension metadata included in this executable's cache identity.
    #[must_use]
    pub fn symbolic_dimensions(&self) -> &[SymbolicDimensionKey] {
        self.cache_key.symbolic_dimensions()
    }

    /// Returns the concrete dynamic-shape bucket used to record this executable, if any.
    #[must_use]
    pub fn shape_bucket(&self) -> Option<&ConcreteShapeBucket> {
        self.cache_key.shape_bucket()
    }

    /// Returns the successful compile report for this CUDA graph executable.
    #[must_use]
    pub fn report(&self) -> &GraphCompileReport {
        &self.report
    }
}

impl CudaCompiledGraph {
    /// Returns allocation-compatible graph output buffers retained by this graph.
    #[must_use]
    pub fn output_buffers(&self) -> &[GraphOutputBufferSummary] {
        &self.output_buffers
    }

    /// Returns output metadata and weak references to graph-owned output values.
    #[must_use]
    pub fn output_buffer_allocations(&self) -> &[GraphOutputBufferAllocation] {
        &self.output_buffer_allocations
    }

    /// Returns allocation-compatible reusable graph-private work buffers retained by this graph.
    #[must_use]
    pub fn reusable_buffers(&self) -> &[GraphReusableBufferSummary] {
        &self.reusable_buffers
    }

    /// Returns records that retain graph-private reusable buffer values.
    #[must_use]
    pub fn reusable_buffer_allocations(&self) -> &[GraphReusableBufferAllocation] {
        &self.reusable_buffer_allocations
    }
}

// What a compiled graph is made from; the caller keeps these mutually consistent.
pub struct CudaCompiledGraphParts {
    pub ir: Arc<Graph>,
    pub cache_key: GraphCacheKey,
    pub stats: CudaCompiledGraphStats,
    pub report: GraphCompileReport,
    pub runtime: Arc<CudaRuntime>,
    pub device: Device,
    pub guards: GraphGuards,
    pub execution_order: Vec<ValueId>,
    pub native_graph: Option<NativeCudaGraph>,
    pub executable: Option<NativeExecutableGraph>,
    pub cudnn_steps: Vec<CudnnGraphStep>,
    pub cutensor_steps: Vec<CutensorGraphStep>,
    pub library_resources: Vec<LibraryResource>,
    pub values: HashMap<ValueId, CudaGraphValue>,
    pub outputs: Vec<ValueId>,
    pub output_buffers: Vec<GraphOutputBufferSummary>,
    pub output_buffer_allocations: Vec<GraphOutputBufferAllocation>,
    pub reusable_buffers: Vec<GraphReusableBufferSummary>,
    pub reusable_buffer_allocations: Vec<GraphReusableBufferAllocation>,
    pub private: GraphPrivateBuffers,
}

impl CudaCompiledGraph {
    /// Takes compiled IR, native graphs, executable resources, buffer records, and output mappings.
    /// Replay counters and pending submissions start empty.
    /// The caller supplies mutually consistent metadata and retained resources; this constructor does not validate them.
    #[must_use]
    pub fn new(parts: CudaCompiledGraphParts) -> Self {
        let CudaCompiledGraphParts {
            ir,
            cache_key,
            stats,
            report,
            runtime,
            device,
            guards,
            execution_order,
            native_graph,
            executable,
            cudnn_steps,
            cutensor_steps,
            library_resources,
            values,
            outputs,
            output_buffers,
            output_buffer_allocations,
            reusable_buffers,
            reusable_buffer_allocations,
            private,
        } = parts;
        Self {
            ir,
            cache_key,
            stats,
            replay_stats: GraphReplayStats::default(),
            report,
            runtime,
            device,
            guards,
            execution_order,
            _native_graph: native_graph,
            executable: executable.map(|executable| Arc::new(Mutex::new(executable))),
            cudnn_steps,
            cutensor_steps,
            _library_resources: library_resources,
            pending_runs: PendingRuns::default(),
            values,
            outputs,
            output_buffers,
            output_buffer_allocations,
            reusable_buffers,
            reusable_buffer_allocations,
            private,
        }
    }
}

impl CudaCompiledGraph {
    pub(crate) fn compile_report(
        ir: &Graph,
        execution_order: &[ValueId],
        lowering_plan: &LoweringPlan,
        realized_lowering: &RealizedLoweringPlan,
        provider_failures: &HashMap<ValueId, GraphProviderFailures>,
        stats: CudaCompiledGraphStats,
        cache_key: &GraphCacheKey,
        fusion_plan: FusionPlan,
    ) -> Result<GraphCompileReport> {
        build_cuda_graph_compile_report(
            ir,
            execution_order,
            lowering_plan,
            realized_lowering,
            provider_failures,
            stats,
            cache_key,
            fusion_plan,
        )
    }

    /// Adds the executable's retained workspace byte count to cuDNN graph statistics.
    ///
    /// # Errors
    ///
    /// Returns an error if the accumulated workspace byte count overflows.
    pub fn record_cudnn_workspace(
        stats: &mut CudaCompiledGraphStats,
        executable: &CudnnBackendExecutable,
    ) -> Result<()> {
        stats.add_cudnn_workspace_bytes(executable.workspace_byte_size())
    }

    /// Builds output buffer metadata in the supplied output order.
    ///
    /// # Errors
    ///
    /// Returns an error if a retained output value is missing or its logical byte count overflows.
    pub fn output_buffer_summaries(
        values: &HashMap<ValueId, CudaGraphValue>,
        outputs: &[ValueId],
    ) -> Result<Vec<GraphOutputBufferSummary>> {
        output_buffer_summaries(values, outputs)
    }

    /// Builds output buffer records with weak graph-value references in the supplied output order.
    ///
    /// # Errors
    ///
    /// Returns an error if a retained output value is missing or its logical byte count overflows.
    pub fn output_buffer_allocation_records(
        values: &HashMap<ValueId, CudaGraphValue>,
        outputs: &[ValueId],
    ) -> Result<Vec<GraphOutputBufferAllocation>> {
        output_buffer_allocation_records(values, outputs)
    }

    // Returns whether replay already leaves the value written into input `name` in that input.
    pub(crate) fn writes_input_in_graph(&self, name: &str) -> bool {
        self.private.inputs_written_in_graph.contains(name)
    }

    pub(crate) fn output_value_tensors<'a>(
        values: &'a HashMap<ValueId, CudaGraphValue>,
        outputs: &[ValueId],
    ) -> Result<Vec<(ValueId, &'a CudaGraphValue)>> {
        output_value_tensors(values, outputs)
    }
}
