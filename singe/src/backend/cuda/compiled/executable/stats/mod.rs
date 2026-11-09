mod recording;

/// Execution statistics collected for a compiled CUDA graph.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CudaCompiledGraphStats {
    /// Number of graph input placeholder nodes.
    input_nodes: usize,
    /// Number of graph inputs backed by static initializer tensors.
    static_input_nodes: usize,
    /// Number of constant tensor nodes embedded in the graph.
    constant_nodes: usize,
    /// Number of metadata-only alias/view nodes.
    alias_nodes: usize,
    /// Number of kernel child graph nodes.
    kernel_child_graph_nodes: usize,
    /// Number of cuBLAS child graph nodes.
    cublas_child_graph_nodes: usize,
    /// Number of cuDNN backend child graph nodes.
    cudnn_backend_graph_nodes: usize,
    /// Number of cuTENSOR child graph nodes.
    cutensor_child_graph_nodes: usize,
    /// Number of native CUDA graph memcpy nodes.
    memcpy_nodes: usize,
    /// Number of native CUDA graph memset nodes.
    memset_nodes: usize,
    /// Bytes allocated for cuDNN backend graph workspaces owned for the compiled graph lifetime.
    cudnn_workspace_bytes: usize,
    /// Bytes allocated for cuBLASLt graph workspaces owned for the compiled graph lifetime.
    cublaslt_workspace_bytes: usize,
    /// Bytes allocated for cuTENSOR graph workspaces owned for the compiled graph lifetime.
    cutensor_workspace_bytes: usize,
    /// Number of graph work buffers reused by the memory planner.
    reused_buffers: usize,
    /// Bytes allocated for graph-private placeholders and work buffers.
    graph_allocation_bytes: usize,
    /// Planned bytes served from reusable graph work buffers instead of new graph allocations.
    reused_buffer_bytes: usize,
    /// Number of alias groups protected from graph work-buffer reuse.
    protected_buffer_groups: usize,
    /// Number of protected alias groups containing input placeholders.
    protected_input_buffer_groups: usize,
    /// Number of protected alias groups containing constants.
    protected_constant_buffer_groups: usize,
    /// Number of protected alias groups containing graph outputs.
    protected_output_buffer_groups: usize,
    /// Number of alias groups released into reusable graph work buffers.
    released_buffer_groups: usize,
    /// Number of graph-private reusable buffers currently retained by the compiled graph.
    reusable_buffer_pool_entries: usize,
    /// Number of input tensors held by asynchronous pending runs.
    pending_run_inputs: usize,
    /// Number of output tensors held by asynchronous pending runs.
    pending_run_outputs: usize,
    /// Current number of asynchronous `run_on` submissions still tracked by the compiled graph.
    pending_runs: usize,
}

impl CudaCompiledGraphStats {
    /// Returns graph work buffers reused by the memory planner.
    #[must_use]
    pub fn reused_buffers(&self) -> usize {
        self.reused_buffers
    }

    /// Returns graph-private placeholder and work-buffer allocation bytes.
    #[must_use]
    pub fn graph_allocation_bytes(&self) -> usize {
        self.graph_allocation_bytes
    }

    /// Returns planned bytes served from reusable graph work buffers.
    #[must_use]
    pub fn reused_buffer_bytes(&self) -> usize {
        self.reused_buffer_bytes
    }

    /// Returns alias groups protected from graph work-buffer reuse.
    #[must_use]
    pub fn protected_buffer_groups(&self) -> usize {
        self.protected_buffer_groups
    }

    /// Returns protected alias groups containing input placeholders.
    #[must_use]
    pub fn protected_input_buffer_groups(&self) -> usize {
        self.protected_input_buffer_groups
    }

    /// Returns protected alias groups containing constants.
    #[must_use]
    pub fn protected_constant_buffer_groups(&self) -> usize {
        self.protected_constant_buffer_groups
    }

    /// Returns protected alias groups containing graph outputs.
    #[must_use]
    pub fn protected_output_buffer_groups(&self) -> usize {
        self.protected_output_buffer_groups
    }

    /// Returns alias groups released into reusable graph work buffers.
    #[must_use]
    pub fn released_buffer_groups(&self) -> usize {
        self.released_buffer_groups
    }

    /// Returns graph-private reusable buffers retained by the compiled graph.
    #[must_use]
    pub fn reusable_buffer_pool_entries(&self) -> usize {
        self.reusable_buffer_pool_entries
    }
}

impl CudaCompiledGraphStats {
    /// Returns graph input placeholder nodes.
    #[must_use]
    pub fn input_nodes(&self) -> usize {
        self.input_nodes
    }

    /// Returns graph inputs backed by static initializer tensors.
    #[must_use]
    pub fn static_input_nodes(&self) -> usize {
        self.static_input_nodes
    }

    /// Returns constant tensor nodes embedded in the graph.
    #[must_use]
    pub fn constant_nodes(&self) -> usize {
        self.constant_nodes
    }

    /// Returns metadata-only alias/view nodes.
    #[must_use]
    pub fn alias_nodes(&self) -> usize {
        self.alias_nodes
    }

    /// Returns kernel child graph nodes.
    #[must_use]
    pub fn kernel_child_graph_nodes(&self) -> usize {
        self.kernel_child_graph_nodes
    }

    /// Returns cuBLAS child graph nodes.
    #[must_use]
    pub fn cublas_child_graph_nodes(&self) -> usize {
        self.cublas_child_graph_nodes
    }

    /// Returns cuDNN backend child graph nodes.
    #[must_use]
    pub fn cudnn_backend_graph_nodes(&self) -> usize {
        self.cudnn_backend_graph_nodes
    }

    /// Returns cuTENSOR child graph nodes.
    #[must_use]
    pub fn cutensor_child_graph_nodes(&self) -> usize {
        self.cutensor_child_graph_nodes
    }

    /// Returns native CUDA graph memcpy nodes.
    #[must_use]
    pub fn memcpy_nodes(&self) -> usize {
        self.memcpy_nodes
    }

    /// Returns native CUDA graph memset nodes.
    #[must_use]
    pub fn memset_nodes(&self) -> usize {
        self.memset_nodes
    }

    /// Returns all library child graph nodes embedded in the native CUDA graph.
    #[must_use]
    pub fn child_graph_nodes(&self) -> usize {
        self.kernel_child_graph_nodes
            .saturating_add(self.cublas_child_graph_nodes)
            .saturating_add(self.cudnn_backend_graph_nodes)
            .saturating_add(self.cutensor_child_graph_nodes)
    }

    /// Returns native provider nodes materialized by CUDA graph lowering.
    #[must_use]
    pub fn native_provider_nodes(&self) -> usize {
        self.kernel_child_graph_nodes
            .saturating_add(self.cublas_child_graph_nodes)
            .saturating_add(self.cudnn_backend_graph_nodes)
            .saturating_add(self.cutensor_child_graph_nodes)
            .saturating_add(self.memcpy_nodes)
            .saturating_add(self.memset_nodes)
    }
}

impl CudaCompiledGraphStats {
    /// Returns input tensors held by asynchronous pending runs.
    #[must_use]
    pub fn pending_run_inputs(&self) -> usize {
        self.pending_run_inputs
    }

    /// Returns output tensors held by asynchronous pending runs.
    #[must_use]
    pub fn pending_run_outputs(&self) -> usize {
        self.pending_run_outputs
    }

    /// Returns asynchronous `run_on` submissions still tracked by the compiled graph.
    #[must_use]
    pub fn pending_runs(&self) -> usize {
        self.pending_runs
    }
}

impl CudaCompiledGraphStats {
    /// Returns bytes allocated for cuDNN backend graph workspaces.
    #[must_use]
    pub fn cudnn_workspace_bytes(&self) -> usize {
        self.cudnn_workspace_bytes
    }

    /// Returns bytes allocated for cuBLASLt graph workspaces.
    #[must_use]
    pub fn cublaslt_workspace_bytes(&self) -> usize {
        self.cublaslt_workspace_bytes
    }

    /// Returns bytes allocated for cuTENSOR graph workspaces.
    #[must_use]
    pub fn cutensor_workspace_bytes(&self) -> usize {
        self.cutensor_workspace_bytes
    }

    /// Returns all library workspace bytes owned for the compiled graph lifetime.
    #[must_use]
    pub fn library_workspace_bytes(&self) -> usize {
        self.cudnn_workspace_bytes
            .saturating_add(self.cublaslt_workspace_bytes)
            .saturating_add(self.cutensor_workspace_bytes)
    }
}
