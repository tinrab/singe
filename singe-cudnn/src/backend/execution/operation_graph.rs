use std::sync::Arc;

use singe_cuda::context::Context as CudaContext;

use crate::{
    backend::{
        attribute::BackendAttributeName,
        descriptor::{BackendDescriptor, BackendDescriptorType},
    },
    context::Context,
    error::{Error, Result},
    utility::{bind_context, bind_for_drop, to_usize},
};

/// Independent native shape modes for an operation graph.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OperationGraphConfig {
    /// Requests native dynamic-shape support when finalizing the operation graph.
    pub dynamic_shape_enabled: bool,
    /// Requests native execution-time shape overrides independently of dynamic-shape support.
    pub override_shape_enabled: bool,
}

/// Owns a finalized native operation graph and retains its live CUDA target when present.
#[derive(Debug)]
pub struct OperationGraph {
    descriptor: BackendDescriptor,
    cuda_context: Option<Arc<CudaContext>>,
    config: OperationGraphConfig,
    operation_types: Vec<BackendDescriptorType>,
}

impl OperationGraph {
    /// Creates a live operation graph with explicitly named shape modes.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation list is empty, the CUDA context cannot be bound, or native graph creation, attribute assignment, or finalization fails.
    pub fn create(
        context: &Context,
        operations: &[&BackendDescriptor],
        config: OperationGraphConfig,
    ) -> Result<Self> {
        Self::create_with_context(Some(context), operations, config)
    }

    /// Creates an operation graph without assigning a live CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation list is empty or native graph creation, attribute assignment, or finalization fails.
    pub fn create_deviceless(
        operations: &[&BackendDescriptor],
        config: OperationGraphConfig,
    ) -> Result<Self> {
        Self::create_with_context(None, operations, config)
    }

    fn create_with_context(
        context: Option<&Context>,
        operations: &[&BackendDescriptor],
        config: OperationGraphConfig,
    ) -> Result<Self> {
        if operations.is_empty() {
            return Err(Error::EmptyList {
                name: "operations".into(),
            });
        }

        if let Some(context) = context {
            context.bind()?;
        }

        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::OperationGraph)?;
        if let Some(context) = context {
            // SAFETY: context is live through native graph configuration and finalization.
            unsafe {
                descriptor.set_attribute_handle(
                    BackendAttributeName::OperationGraphHandle,
                    context.as_raw(),
                )
            }?;
        }
        descriptor
            .set_attribute_descriptor_slice(BackendAttributeName::OperationGraphOps, operations)?;
        if config.dynamic_shape_enabled {
            descriptor.set_attribute_bool(
                BackendAttributeName::OperationGraphIsDynamicShapeEnabled,
                true,
            )?;
        }
        if config.override_shape_enabled {
            descriptor.set_attribute_bool(
                BackendAttributeName::OperationGraphIsOverrideShapeEnabled,
                true,
            )?;
        }
        descriptor.finalize()?;

        Ok(Self {
            descriptor,
            cuda_context: context.map(|context| Arc::clone(context.cuda_context())),
            config,
            operation_types: operations
                .iter()
                .map(|operation| operation.descriptor_type())
                .collect(),
        })
    }

    /// Number of native operations accepted during construction.
    /// Expanded frontend compositions may produce more than one native operation.
    /// Reading this metadata does not query cuDNN or bind a CUDA context.
    #[must_use]
    pub fn operation_count(&self) -> usize {
        self.operation_types.len()
    }

    /// Native operation descriptor kinds in construction order, including repeated kinds.
    /// Available for both live and deviceless graphs without querying native descriptors.
    #[must_use]
    pub fn operation_types(&self) -> &[BackendDescriptorType] {
        &self.operation_types
    }

    /// Queries the number of native engines available to this operation graph.
    ///
    /// # Errors
    ///
    /// Returns an error if the retained CUDA context cannot be bound, the native attribute query fails, or the reported count cannot fit `usize`.
    pub fn engine_count(&self) -> Result<usize> {
        bind_context(self.cuda_context())?;
        to_usize(
            self.descriptor
                .attribute_i64(BackendAttributeName::OperationGraphEngineGlobalCount)?,
            "engine_count",
        )
    }

    /// Borrows the finalized native operation-graph descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }

    /// Returns the actual retained CUDA context, or None for deviceless metadata.
    #[must_use]
    pub fn cuda_context(&self) -> Option<&Arc<CudaContext>> {
        self.cuda_context.as_ref()
    }

    /// Shape modes accepted during construction, including disabled native defaults.
    #[must_use]
    pub fn config(&self) -> OperationGraphConfig {
        self.config
    }

    /// Returns whether native dynamic-shape support was requested at construction.
    #[must_use]
    pub fn is_dynamic_shape_enabled(&self) -> bool {
        self.config.dynamic_shape_enabled
    }

    /// Returns whether native execution-time shape overrides were requested at construction.
    #[must_use]
    pub fn is_override_shape_enabled(&self) -> bool {
        self.config.override_shape_enabled
    }
}

impl Drop for OperationGraph {
    fn drop(&mut self) {
        bind_for_drop(self.cuda_context(), "cudnn operation graph");
    }
}
