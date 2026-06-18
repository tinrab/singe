use std::sync::Arc;

use singe_cuda::context::Context as CudaContext;

use crate::{
    attribute::BackendAttributeName,
    context::Context,
    descriptor::{BackendDescriptor, BackendDescriptorType},
    error::{Error, Result},
    utility::to_usize,
};

#[derive(Debug)]
pub struct OperationGraph {
    descriptor: BackendDescriptor,
    cuda_ctx: Option<Arc<CudaContext>>,
}

impl OperationGraph {
    pub fn create(ctx: &Context, operations: &[&BackendDescriptor]) -> Result<Self> {
        Self::create_with_context(Some(ctx), operations, false, false)
    }

    pub fn create_with_config(
        ctx: &Context,
        operations: &[&BackendDescriptor],
        dynamic_shape_enabled: bool,
    ) -> Result<Self> {
        Self::create_with_context(Some(ctx), operations, dynamic_shape_enabled, false)
    }

    pub fn create_with_shape_config(
        ctx: &Context,
        operations: &[&BackendDescriptor],
        dynamic_shape_enabled: bool,
        override_shape_enabled: bool,
    ) -> Result<Self> {
        Self::create_with_context(
            Some(ctx),
            operations,
            dynamic_shape_enabled,
            override_shape_enabled,
        )
    }

    pub fn create_deviceless(
        operations: &[&BackendDescriptor],
        dynamic_shape_enabled: bool,
    ) -> Result<Self> {
        Self::create_with_context(None, operations, dynamic_shape_enabled, false)
    }

    pub fn create_deviceless_with_shape_config(
        operations: &[&BackendDescriptor],
        dynamic_shape_enabled: bool,
        override_shape_enabled: bool,
    ) -> Result<Self> {
        Self::create_with_context(
            None,
            operations,
            dynamic_shape_enabled,
            override_shape_enabled,
        )
    }

    fn create_with_context(
        ctx: Option<&Context>,
        operations: &[&BackendDescriptor],
        dynamic_shape_enabled: bool,
        override_shape_enabled: bool,
    ) -> Result<Self> {
        if operations.is_empty() {
            return Err(Error::EmptyList {
                name: "operations".into(),
            });
        }

        if let Some(ctx) = ctx {
            ctx.bind()?;
        }

        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::OperationGraph)?;
        if let Some(ctx) = ctx {
            descriptor
                .set_attribute_handle(BackendAttributeName::OperationGraphHandle, ctx.as_raw())?;
        }
        descriptor
            .set_attribute_descriptor_slice(BackendAttributeName::OperationGraphOps, operations)?;
        if dynamic_shape_enabled {
            descriptor.set_attribute_bool(
                BackendAttributeName::OperationGraphIsDynamicShapeEnabled,
                true,
            )?;
        }
        if override_shape_enabled {
            descriptor.set_attribute_bool(
                BackendAttributeName::OperationGraphIsOverrideShapeEnabled,
                true,
            )?;
        }
        descriptor.finalize()?;

        Ok(Self {
            descriptor,
            cuda_ctx: ctx.map(|ctx| Arc::clone(ctx.cuda_context())),
        })
    }

    pub fn engine_count(&self) -> Result<usize> {
        to_usize(
            self.descriptor
                .attribute_i64(BackendAttributeName::OperationGraphEngineGlobalCount)?,
            "engine_count",
        )
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }

    pub(crate) fn cuda_context(&self) -> Option<&Arc<CudaContext>> {
        self.cuda_ctx.as_ref()
    }

    pub fn is_dynamic_shape_enabled(&self) -> Result<bool> {
        self.descriptor
            .attribute_bool(BackendAttributeName::OperationGraphIsDynamicShapeEnabled)
    }

    pub fn is_override_shape_enabled(&self) -> Result<bool> {
        self.descriptor
            .attribute_bool(BackendAttributeName::OperationGraphIsOverrideShapeEnabled)
    }
}
