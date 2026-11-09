use std::collections::HashMap;

use singe_cuda::{error::Error as CudaError, graph::GraphNode};

use crate::{
    backend::cuda::{
        allocator::CudaAllocator,
        compiled::plan::memory::{GraphArena, GraphMemoryPlanner, MemoryPlanKey},
        graph_value::CudaGraphValue,
    },
    data_type::DataType,
    error::{Error, Result},
    graph::ir::{Graph, ValueId},
    layout::Layout,
    utility::checked_add,
};

impl GraphMemoryPlanner {
    /// Obtains storage for the node's primary result, reusing a free buffer with the same data type and layout when possible.
    /// A reused allocation carries the graph dependencies that must precede its next write.
    ///
    /// # Errors
    ///
    /// Returns an error if graph result metadata is absent, the allocator serves another device or returns another CUDA context, span or counter arithmetic overflows, or allocation or region validation fails.
    /// An error after selecting a reusable buffer can leave that buffer removed from the free pool and reuse counters partially updated.
    pub fn allocate(
        &mut self,
        allocator: &dyn CudaAllocator,
        ir: &Graph,
        id: ValueId,
    ) -> Result<GraphAllocation> {
        let value = id;
        let device = ir.required_value(value)?.device();
        if allocator.device() != device {
            return Err(Error::device_mismatch(
                "cuda graph allocator device",
                device,
                allocator.device(),
            ));
        }

        let data_type = ir.required_value_data_type(value)?;
        let layout = ir.required_value_layout(value)?;
        let element_count = layout.storage_span_element_count()?;
        let byte_size = element_count
            .checked_mul(data_type.size_bytes())
            .ok_or_else(|| Error::out_of_range("cuda graph allocation byte size"))?;
        let key = MemoryPlanKey::new(data_type, layout.clone());
        if let Some(buffers) = self.free.get_mut(&key)
            && let Some(buffer) = buffers.pop()
        {
            self.reused_buffers += 1;
            self.reused_buffer_bytes = checked_add(
                self.reused_buffer_bytes,
                byte_size,
                "cuda graph reused buffer bytes",
            )?;
            return Ok(GraphAllocation::new(
                buffer.tensor.clone(),
                buffer.dependencies,
            ));
        }

        if let Some(buffer) = self.take_reinterpretable(data_type, byte_size) {
            self.reused_buffers += 1;
            self.reused_buffer_bytes = checked_add(
                self.reused_buffer_bytes,
                byte_size,
                "cuda graph reused buffer bytes",
            )?;
            return Ok(GraphAllocation::new(
                buffer.tensor.reinterpret(data_type, layout.clone())?,
                buffer.dependencies,
            ));
        }

        let (value, fresh) = self.buffer(allocator, data_type, layout, element_count)?;
        if fresh {
            self.allocated_bytes = checked_add(
                self.allocated_bytes,
                byte_size,
                "cuda graph allocated bytes",
            )?;
        }
        Ok(GraphAllocation::new(value, Vec::new()))
    }

    pub(crate) fn set_arena(&mut self, arena: Option<GraphArena>) {
        self.arena = arena;
    }

    /// Returns a buffer the graph holds for itself, from the arena when one fits, and whether it had to allocate it.
    pub(crate) fn buffer(
        &mut self,
        allocator: &dyn CudaAllocator,
        data_type: DataType,
        layout: &Layout,
        element_count: usize,
    ) -> Result<(CudaGraphValue, bool)> {
        let byte_size = element_count
            .checked_mul(data_type.size_bytes())
            .ok_or_else(|| Error::out_of_range("cuda graph allocation byte size"))?;
        if let Some(arena) = &self.arena
            && let Some(value) = arena.take(data_type, layout, byte_size, &mut self.arena_taken)?
        {
            return Ok((value, false));
        }

        let storage = allocator.allocate_uninit(data_type, element_count)?;
        if storage.context() != allocator.context() {
            return Err(CudaError::MemoryContextMismatch.into());
        }

        let value = CudaGraphValue::create(storage, data_type, layout.clone())?;
        if let Some(arena) = &self.arena {
            self.arena_taken.push(value.allocation_id());
            arena.add(value.clone())?;
        }
        Ok((value, true))
    }

    /// Removes the smallest free pending allocation that holds `byte_size` bytes as whole elements of `data_type`.
    /// Its whole alias group has been released, so every byte of the allocation is free for another data type and layout.
    fn take_reinterpretable(
        &mut self,
        data_type: DataType,
        byte_size: usize,
    ) -> Option<GraphReusableBuffer> {
        let fits = |buffer: &GraphReusableBuffer| {
            buffer
                .tensor
                .pending_allocation_bytes()
                .filter(|&bytes| bytes >= byte_size && bytes.is_multiple_of(data_type.size_bytes()))
        };
        // Ties break by key and position, so compiling one graph twice picks the same buffers.
        let (_, key, index) = self
            .free
            .iter()
            .flat_map(|(key, buffers)| {
                buffers
                    .iter()
                    .enumerate()
                    .filter_map(move |(index, buffer)| Some((fits(buffer)?, key, index)))
            })
            .min()?;
        let key = key.clone();
        let buffers = self.free.get_mut(&key)?;
        let buffer = buffers.swap_remove(index);
        if buffers.is_empty() {
            self.free.remove(&key);
        }
        Some(buffer)
    }
}

/// A pending graph value and the graph nodes that must finish before its storage is written, which are the readers of its previous value when the storage is reused.
#[derive(Debug)]
pub struct GraphAllocation {
    tensor: CudaGraphValue,
    dependencies: Vec<GraphNode>,
}

#[derive(Debug)]
pub struct GraphReusableBuffer {
    pub(crate) tensor: CudaGraphValue,
    pub(crate) dependencies: Vec<GraphNode>,
}

impl GraphAllocation {
    pub(crate) fn new(tensor: CudaGraphValue, dependencies: Vec<GraphNode>) -> Self {
        Self {
            tensor,
            dependencies,
        }
    }

    /// Consumes the allocation descriptor and returns its pending graph value.
    /// The caller must preserve any reuse dependency before scheduling writes.
    #[must_use]
    pub fn into_value(self) -> CudaGraphValue {
        self.tensor
    }

    /// Returns the graph nodes that must complete before the allocation is written.
    #[must_use]
    pub fn dependencies(&self) -> Vec<GraphNode> {
        self.dependencies.clone()
    }
}

impl GraphMemoryPlanner {
    pub(crate) fn release(&mut self, tensor: &CudaGraphValue, dependencies: Vec<GraphNode>) {
        let key = MemoryPlanKey::for_tensor(tensor);
        self.free.entry(key).or_default().push(GraphReusableBuffer {
            tensor: tensor.clone(),
            dependencies,
        });
    }
}

impl GraphMemoryPlanner {
    // The unprotected values whose whole storage group `id` is the last to read at `index`, each group once, for callers that track buffers themselves.
    pub(crate) fn released_by(&mut self, id: ValueId, index: usize) -> Vec<ValueId> {
        let dependencies = self.dependencies.get(&id).cloned().unwrap_or_default();
        dependencies
            .into_iter()
            .filter(|dependency| {
                self.release_use.get(dependency).copied() == Some(index)
                    && self.release_group.get(dependency).is_some_and(|&group| {
                        !self.protected_groups.contains(&group)
                            && self.released_groups.insert(group)
                    })
            })
            .collect()
    }
}

impl GraphMemoryPlanner {
    /// Releases unprotected dependency storage groups whose last use is `index`.
    /// The node `id` supplies the dependency list, and `values` supplies their retained buffers.
    /// Each group enters the free pool at most once, with `barrier` as its reuse dependency.
    pub fn release_dependencies(
        &mut self,
        id: ValueId,
        index: usize,
        values: &HashMap<ValueId, CudaGraphValue>,
        barrier: &GraphNode,
    ) {
        let dependencies = self.dependencies.get(&id).cloned().unwrap_or_default();
        // Readers of one group are not ordered against each other, so reuse waits for all of them, not only the last in execution order.
        for dependency in &dependencies {
            if let Some(&group) = self.release_group.get(dependency) {
                let readers = self.group_readers.entry(group).or_default();
                if !readers.contains(barrier) {
                    readers.push(barrier.clone());
                }
            }
        }
        for dependency in dependencies {
            if self.release_use.get(&dependency).copied() == Some(index)
                && let Some(tensor) = values.get(&dependency)
            {
                let Some(group) = self.release_group.get(&dependency).copied() else {
                    continue;
                };
                if self.protected_groups.contains(&group) {
                    continue;
                }
                if self.released_groups.insert(group) {
                    debug_assert!(
                        !self.protected_groups.contains(&group),
                        "attempted to release protected cuda graph alias group"
                    );
                    let readers = self.group_readers.remove(&group).unwrap_or_default();
                    self.release(tensor, readers);
                }
            }
        }
    }
}
