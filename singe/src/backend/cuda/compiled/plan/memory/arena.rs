use std::sync::{Arc, Mutex};

use singe_cuda::view::AllocationId;

use crate::{
    backend::cuda::graph_value::CudaGraphValue, data_type::DataType, error::Result, layout::Layout,
    utility::Locked,
};

// The replay buffers an executor's specializations of one program share, since only one of them replays at a time, and submissions order their replays by the allocations they touch.
// A compile takes each buffer at most once, so the values of one graph never overlap, and adds a buffer only when none of those it has not taken fits.
#[derive(Debug, Clone, Default)]
pub struct GraphArena {
    buffers: Arc<Mutex<Vec<CudaGraphValue>>>,
}

impl GraphArena {
    // Returns the smallest buffer outside `taken` that holds `byte_size` bytes as whole elements of `data_type`, as `data_type` and `layout`.
    pub(crate) fn take(
        &self,
        data_type: DataType,
        layout: &Layout,
        byte_size: usize,
        taken: &mut Vec<AllocationId>,
    ) -> Result<Option<CudaGraphValue>> {
        let best = self
            .buffers
            .locked("graph arena")?
            .iter()
            .filter(|buffer| !taken.contains(&buffer.allocation_id()))
            .filter_map(|buffer| {
                let bytes = buffer.pending_allocation_bytes()?;
                (bytes >= byte_size && bytes.is_multiple_of(data_type.size_bytes()))
                    .then_some((bytes, buffer))
            })
            .min_by_key(|&(bytes, _)| bytes)
            .map(|(_, buffer)| buffer.clone());
        let Some(buffer) = best else {
            return Ok(None);
        };
        taken.push(buffer.allocation_id());
        Ok(Some(buffer.reinterpret(data_type, layout.clone())?))
    }

    pub(crate) fn add(&self, buffer: CudaGraphValue) -> Result<()> {
        self.buffers.locked("graph arena")?.push(buffer);
        Ok(())
    }
}
