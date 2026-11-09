use crate::{
    backend::cuda::compiled::executable::CudaCompiledGraphStats, error::Result,
    utility::checked_add,
};

impl CudaCompiledGraphStats {
    /// Counts one input node and adds its graph allocation byte count.
    ///
    /// # Errors
    ///
    /// Returns an error if the accumulated graph allocation byte count overflows.
    /// On this error, neither the byte count nor the input node count changes.
    pub fn record_input_node(&mut self, graph_allocation_bytes: usize) -> Result<()> {
        self.graph_allocation_bytes = checked_add(
            self.graph_allocation_bytes,
            graph_allocation_bytes,
            "cuda graph allocation bytes",
        )?;
        self.input_nodes += 1;
        Ok(())
    }

    /// Counts one static input node.
    pub fn record_static_input_node(&mut self) {
        self.static_input_nodes += 1;
    }

    /// Counts one constant node.
    pub fn record_constant_node(&mut self) {
        self.constant_nodes += 1;
    }

    /// Counts one storage alias node.
    pub fn record_alias_node(&mut self) {
        self.alias_nodes += 1;
    }

    /// Counts one kernel child graph node.
    pub fn record_kernel_child_graph_node(&mut self) {
        self.kernel_child_graph_nodes += 1;
    }

    /// Counts one cuBLAS child graph node.
    pub fn record_cublas_child_graph_node(&mut self) {
        self.cublas_child_graph_nodes += 1;
    }

    /// Counts one cuDNN backend graph node.
    pub fn record_cudnn_backend_graph_node(&mut self) {
        self.cudnn_backend_graph_nodes += 1;
    }

    /// Counts one cuTENSOR child graph node.
    pub fn record_cutensor_child_graph_node(&mut self) {
        self.cutensor_child_graph_nodes += 1;
    }

    /// Counts one memory copy node.
    pub fn record_memcpy_node(&mut self) {
        self.memcpy_nodes += 1;
    }

    /// Counts one memory fill node.
    pub fn record_memset_node(&mut self) {
        self.memset_nodes += 1;
    }

    /// Adds `byte_size` to the accumulated cuDNN workspace byte count.
    ///
    /// # Errors
    ///
    /// Returns an error if the accumulated workspace byte count overflows.
    /// The previous count remains unchanged on error.
    pub fn add_cudnn_workspace_bytes(&mut self, byte_size: usize) -> Result<()> {
        self.cudnn_workspace_bytes = checked_add(
            self.cudnn_workspace_bytes,
            byte_size,
            "cuda graph cuDNN workspace bytes",
        )?;
        Ok(())
    }

    /// Adds `byte_size` to the accumulated cuBLASLt workspace byte count.
    ///
    /// # Errors
    ///
    /// Returns an error if the accumulated workspace byte count overflows.
    /// The previous count remains unchanged on error.
    pub fn add_cublaslt_workspace_bytes(&mut self, byte_size: usize) -> Result<()> {
        self.cublaslt_workspace_bytes = checked_add(
            self.cublaslt_workspace_bytes,
            byte_size,
            "cuda graph cuBLASLt workspace bytes",
        )?;
        Ok(())
    }

    /// Adds `byte_size` to the accumulated cuTENSOR workspace byte count.
    ///
    /// # Errors
    ///
    /// Returns an error if the accumulated workspace byte count overflows.
    /// The previous count remains unchanged on error.
    pub fn add_cutensor_workspace_bytes(&mut self, byte_size: usize) -> Result<()> {
        self.cutensor_workspace_bytes = checked_add(
            self.cutensor_workspace_bytes,
            byte_size,
            "cuda graph cuTENSOR workspace bytes",
        )?;
        Ok(())
    }

    /// Replaces memory planner counters and adds `graph_allocation_bytes` to the allocation total.
    /// The supplied reused-buffer byte count replaces the previous value.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph allocation byte total overflows.
    /// Group, reuse, and pool entry counters have already been replaced on this error, but allocation and reused-buffer byte totals retain their previous values.
    pub fn record_memory_plan(
        &mut self,
        reused_buffers: usize,
        protected_buffer_groups: usize,
        protected_input_buffer_groups: usize,
        protected_constant_buffer_groups: usize,
        protected_output_buffer_groups: usize,
        released_buffer_groups: usize,
        reusable_buffer_pool_entries: usize,
        graph_allocation_bytes: usize,
        reused_buffer_bytes: usize,
    ) -> Result<()> {
        self.reused_buffers = reused_buffers;
        self.protected_buffer_groups = protected_buffer_groups;
        self.protected_input_buffer_groups = protected_input_buffer_groups;
        self.protected_constant_buffer_groups = protected_constant_buffer_groups;
        self.protected_output_buffer_groups = protected_output_buffer_groups;
        self.released_buffer_groups = released_buffer_groups;
        self.reusable_buffer_pool_entries = reusable_buffer_pool_entries;
        self.graph_allocation_bytes = checked_add(
            self.graph_allocation_bytes,
            graph_allocation_bytes,
            "cuda graph allocation bytes",
        )?;
        self.reused_buffer_bytes = reused_buffer_bytes;
        Ok(())
    }

    /// Returns these statistics with pending run, input, and output counts replaced.
    /// The supplied counts describe retained asynchronous runs and their tensor values.
    #[must_use]
    pub fn with_pending_runs(
        mut self,
        pending_runs: usize,
        pending_run_inputs: usize,
        pending_run_outputs: usize,
    ) -> Self {
        self.pending_runs = pending_runs;
        self.pending_run_inputs = pending_run_inputs;
        self.pending_run_outputs = pending_run_outputs;
        self
    }
}
