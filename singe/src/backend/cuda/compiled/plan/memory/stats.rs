use crate::{
    backend::cuda::compiled::{
        executable::{
            CudaCompiledGraphStats, GraphReusableBufferAllocation, GraphReusableBufferSummary,
        },
        plan::memory::GraphMemoryPlanner,
    },
    error::Result,
    utility::checked_mul,
};

impl GraphMemoryPlanner {
    /// Returns the number of buffer requests satisfied by reusing a free buffer.
    #[must_use]
    pub fn reused_buffers(&self) -> usize {
        self.reused_buffers
    }

    /// Returns the total bytes allocated for new planner buffers.
    #[must_use]
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }

    /// Returns the cumulative bytes supplied by reused buffers.
    #[must_use]
    pub fn reused_buffer_bytes(&self) -> usize {
        self.reused_buffer_bytes
    }

    /// Returns the number of storage groups protected from planner reuse.
    #[must_use]
    pub fn protected_groups(&self) -> usize {
        self.protected_groups.len()
    }

    /// Returns the number of protected input storage groups.
    #[must_use]
    pub fn protected_input_groups(&self) -> usize {
        self.protected_input_groups.len()
    }

    /// Returns the number of protected constant storage groups.
    #[must_use]
    pub fn protected_constant_groups(&self) -> usize {
        self.protected_constant_groups.len()
    }

    /// Returns the number of protected output storage groups.
    #[must_use]
    pub fn protected_output_groups(&self) -> usize {
        self.protected_output_groups.len()
    }

    /// Returns the number of distinct storage groups released by the planner.
    #[must_use]
    pub fn released_groups(&self) -> usize {
        self.released_groups.len()
    }

    /// Returns the number of buffer entries currently in the free pool.
    pub fn reusable_buffer_pool_entries(&self) -> usize {
        self.free.values().map(Vec::len).sum()
    }

    /// Copies planner counters into graph statistics and adds the planner allocation byte total.
    ///
    /// # Errors
    ///
    /// Returns an error if the accumulated graph allocation byte count overflows.
    /// Group, reuse, and pool entry counters have already been copied on this error.
    pub fn record_stats(&self, stats: &mut CudaCompiledGraphStats) -> Result<()> {
        stats.record_memory_plan(
            self.reused_buffers(),
            self.protected_groups(),
            self.protected_input_groups(),
            self.protected_constant_groups(),
            self.protected_output_groups(),
            self.released_groups(),
            self.reusable_buffer_pool_entries(),
            self.allocated_bytes(),
            self.reused_buffer_bytes(),
        )
    }

    /// Summarizes nonempty free buffer buckets by data type and layout.
    /// Each summary reports an entry count and its total logical byte count.
    /// Results are sorted by data type, dimensions, strides, and storage offset.
    ///
    /// # Errors
    ///
    /// Returns an error if a buffer byte count or a bucket total overflows.
    pub fn reusable_buffer_summaries(&self) -> Result<Vec<GraphReusableBufferSummary>> {
        summaries(self)
    }

    /// Returns descriptors that retain each free buffer allocation.
    /// Results are sorted by data type, dimensions, strides, storage offset, and byte count.
    ///
    /// # Errors
    ///
    /// Returns an error if a buffer shape byte count cannot be represented.
    pub fn reusable_buffer_allocations(&self) -> Result<Vec<GraphReusableBufferAllocation>> {
        allocations(self)
    }
}

pub fn summaries(planner: &GraphMemoryPlanner) -> Result<Vec<GraphReusableBufferSummary>> {
    let mut summaries = Vec::with_capacity(planner.free.len());
    for (key, buffers) in &planner.free {
        let entries = buffers.len();
        if entries == 0 {
            continue;
        }

        let buffer_bytes = checked_mul(
            key.layout().shape().element_count(),
            key.data_type().size_bytes(),
            "cuda graph reusable buffer byte size",
        )?;
        let bytes = checked_mul(
            buffer_bytes,
            entries,
            "cuda graph reusable buffer summary byte size",
        )?;
        summaries.push(GraphReusableBufferSummary::new(
            key.data_type(),
            key.layout().clone(),
            entries,
            bytes,
        ));
    }

    summaries.sort_by(|left, right| {
        (left.data_type(), left.layout()).cmp(&(right.data_type(), right.layout()))
    });
    Ok(summaries)
}

pub fn allocations(planner: &GraphMemoryPlanner) -> Result<Vec<GraphReusableBufferAllocation>> {
    let mut allocations = Vec::new();
    for (key, buffers) in &planner.free {
        let bytes = checked_mul(
            key.layout().shape().element_count(),
            key.data_type().size_bytes(),
            "cuda graph reusable buffer allocation byte size",
        )?;
        for buffer in buffers {
            allocations.push(GraphReusableBufferAllocation::new(
                key.data_type(),
                key.layout().clone(),
                bytes,
                buffer.tensor.clone(),
            ));
        }
    }

    allocations.sort_by(|left, right| {
        (left.data_type(), left.layout(), left.bytes()).cmp(&(
            right.data_type(),
            right.layout(),
            right.bytes(),
        ))
    });
    Ok(allocations)
}
