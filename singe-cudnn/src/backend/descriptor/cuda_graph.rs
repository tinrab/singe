use singe_cuda::graph::Graph;
use singe_cudnn_sys as sys;

use crate::{
    backend::descriptor::{BackendDescriptor, BackendDescriptorType},
    context::Context,
    error::{Error, Result},
    try_ffi,
    utility::validate_context_pair,
};

impl BackendDescriptor {
    /// Directly builds a CUDA graph (not to be confused with a cuDNN graph)
    /// representing the given engine.
    /// Instantiating and executing this CUDA graph executes
    /// the engine configuration plan on the [`VariantPack`](crate::backend::execution::VariantPack)
    /// and the finalized [`ExecutionPlan`](crate::backend::execution::ExecutionPlan) on
    /// the data.
    ///
    /// The resulting CUDA graph captures the data and workspace addresses in the
    /// [`VariantPack`](crate::backend::execution::VariantPack) at the time this method is called, but it can
    /// be run arbitrarily many times with different data at the same addresses.
    /// The graph can also later be modified in place with different
    /// [`VariantPack`](crate::backend::execution::VariantPack) addresses by using
    /// [`BackendDescriptor::update_graph`].
    ///
    /// The initial CUDA graph passed to this method must be empty (having no nodes), and no additional nodes may be appended to the resulting graph.
    /// However, the graph can be embedded as a child node of a larger CUDA graph, for example, by [`Graph::add_child_graph_node`].
    /// This is the typical usage.
    ///
    /// Only a limited number of engines currently support this method (with more to be added in future releases of cuDNN).
    /// Engines that support it have the behavior note [`BackendBehaviorNote::SupportsCudaGraphNativeApi`](crate::backend::behavior::BackendBehaviorNote::SupportsCudaGraphNativeApi).
    ///
    /// # Errors
    ///
    /// Returns an error if required descriptors are not finalized or have the
    /// wrong type, if required data pointers are invalid, if `graph` is not
    /// initially empty, if CUDA graph creation fails, if the engine does not
    /// support the native CUDA graph API, if this cuDNN build uses an
    /// incompatible CUDA runtime, or if cuDNN reports an internal failure.
    /// A known graph context must match the supplied cuDNN context.
    ///
    /// # Safety
    /// All native accesses require live plan code and valid, initialized inputs or writable outputs,
    /// with compatible layouts, permitted aliases and established producer dependencies.
    /// Keep device storage and code alive through completion and every captured replay.
    /// Host argument storage must remain valid while cuDNN consumes it.
    /// Retain previous device bindings and workspace through executable snapshots
    /// and pending launches that still reference them.
    pub unsafe fn populate_graph(
        &self,
        context: &Context,
        varpack: &Self,
        graph: &Graph,
    ) -> Result<()> {
        if self.descriptor_type != BackendDescriptorType::ExecutionPlan {
            return Err(Error::InvalidBackendDescriptorType {
                expected: BackendDescriptorType::ExecutionPlan,
                actual: self.descriptor_type,
            });
        }
        if !self.finalized {
            return Err(Error::DescriptorRequiredFinalized);
        }
        if varpack.descriptor_type() != BackendDescriptorType::VariantPack {
            return Err(Error::InvalidBackendDescriptorType {
                expected: BackendDescriptorType::VariantPack,
                actual: varpack.descriptor_type(),
            });
        }
        if !varpack.is_finalized() {
            return Err(Error::DescriptorRequiredFinalized);
        }

        validate_context_pair(Some(context.cuda_context()), graph.context(), "cuda graph")?;
        context.bind()?;
        // SAFETY: `cudnnBackendPopulateCudaGraph` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cudnnBackendPopulateCudaGraph(
                context.as_raw(),
                self.as_raw(),
                varpack.as_raw(),
                graph.as_raw(),
            ))?;
        }

        Ok(())
    }

    /// Updates an existing CUDA graph previously populated by [`BackendDescriptor::populate_graph`]
    /// (or a clone thereof) with a new `varpack`.
    ///
    /// Only a limited number of engines currently support this method (with more to be added in future releases of cuDNN).
    /// Engines that support it have the behavior note [`BackendBehaviorNote::SupportsCudaGraphNativeApi`](crate::backend::behavior::BackendBehaviorNote::SupportsCudaGraphNativeApi).
    ///
    /// # Errors
    ///
    /// Returns an error if required descriptors are not finalized or have the
    /// wrong type, if required data pointers are invalid, if `graph` was not
    /// populated by [`BackendDescriptor::populate_graph`] for this execution plan
    /// or contains unexpected additional nodes, if CUDA graph update fails, if
    /// the engine does not support the native CUDA graph API, if this cuDNN build
    /// uses an incompatible CUDA runtime, or if cuDNN reports an internal failure.
    /// A known graph context must match the supplied cuDNN context.
    ///
    /// # Safety
    /// All native accesses require live plan code and valid, initialized inputs or writable outputs,
    /// with compatible layouts, permitted aliases and established producer dependencies.
    /// Keep device storage and code alive through completion and every captured replay.
    /// Host argument storage must remain valid while cuDNN consumes it.
    /// Retain previous device bindings and workspace through executable snapshots
    /// and pending launches that still reference them.
    pub unsafe fn update_graph(
        &self,
        context: &Context,
        new_varpack: &Self,
        graph: &Graph,
    ) -> Result<()> {
        if self.descriptor_type != BackendDescriptorType::ExecutionPlan {
            return Err(Error::InvalidBackendDescriptorType {
                expected: BackendDescriptorType::ExecutionPlan,
                actual: self.descriptor_type,
            });
        }
        if !self.finalized {
            return Err(Error::DescriptorRequiredFinalized);
        }
        if new_varpack.descriptor_type != BackendDescriptorType::VariantPack {
            return Err(Error::InvalidBackendDescriptorType {
                expected: BackendDescriptorType::VariantPack,
                actual: new_varpack.descriptor_type,
            });
        }
        if !new_varpack.is_finalized() {
            return Err(Error::DescriptorRequiredFinalized);
        }

        validate_context_pair(Some(context.cuda_context()), graph.context(), "cuda graph")?;
        context.bind()?;
        // SAFETY: `cudnnBackendUpdateCudaGraph` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(sys::cudnnBackendUpdateCudaGraph(
                context.as_raw(),
                self.as_raw(),
                new_varpack.as_raw(),
                graph.as_raw()
            ))?;
        }

        Ok(())
    }
}
