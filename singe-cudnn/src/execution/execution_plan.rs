use std::{rc::Rc, sync::Arc};

use singe_cuda::context::Context as CudaContext;
use singe_cuda::graph::Graph;

use crate::{
    attribute::BackendAttributeName,
    context::Context,
    descriptor::{BackendDescriptor, BackendDescriptorType, backend_execute},
    device::DeviceProperties,
    error::{Error, Result},
    execution::{EngineConfig, KernelCache, VariantPack, validate_context_pair},
    utility::to_usize,
};

#[derive(Debug)]
pub struct ExecutionPlan {
    descriptor: BackendDescriptor,
    cuda_ctx: Option<Arc<CudaContext>>,
}

impl ExecutionPlan {
    pub fn create(ctx: &Context, engine_config: &EngineConfig) -> Result<Self> {
        Self::create_with_context(Some(ctx), engine_config, None, None)
    }

    pub fn create_with_config(
        ctx: &Context,
        engine_config: &EngineConfig,
        kernel_cache: Option<&KernelCache>,
        device_properties: Option<&DeviceProperties>,
    ) -> Result<Self> {
        Self::create_with_context(Some(ctx), engine_config, kernel_cache, device_properties)
    }

    pub fn create_deviceless(
        engine_config: &EngineConfig,
        kernel_cache: Option<&KernelCache>,
        device_properties: Option<&DeviceProperties>,
    ) -> Result<Self> {
        Self::create_with_context(None, engine_config, kernel_cache, device_properties)
    }

    fn create_with_context(
        ctx: Option<&Context>,
        engine_config: &EngineConfig,
        kernel_cache: Option<&KernelCache>,
        device_properties: Option<&DeviceProperties>,
    ) -> Result<Self> {
        validate_context_pair(
            ctx.map(|ctx| ctx.cuda_context()),
            engine_config.cuda_context(),
            "engine config",
        )?;

        if let Some(ctx) = ctx {
            ctx.bind()?;
        }
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::ExecutionPlan)?;
        if let Some(ctx) = ctx {
            descriptor
                .set_attribute_handle(BackendAttributeName::ExecutionPlanHandle, ctx.as_raw())?;
        }
        descriptor.set_attribute_descriptor(
            BackendAttributeName::ExecutionPlanEngineConfig,
            engine_config.descriptor(),
        )?;
        if let Some(kernel_cache) = kernel_cache {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::ExecutionPlanKernelCache,
                kernel_cache.descriptor(),
            )?;
        }
        if let Some(device_properties) = device_properties {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::ExecutionPlanDeviceProp,
                device_properties.descriptor(),
            )?;
        }
        descriptor.finalize()?;

        Ok(Self {
            descriptor,
            cuda_ctx: ctx
                .map(|ctx| Arc::clone(ctx.cuda_context()))
                .or_else(|| engine_config.cuda_context().map(Arc::clone)),
        })
    }

    pub fn from_json(ctx: &Context, json_plan: &str) -> Result<Self> {
        ctx.bind()?;

        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::ExecutionPlan)?;
        descriptor.set_attribute_char_string(
            BackendAttributeName::ExecutionPlanJsonRepresentation,
            json_plan,
        )?;
        descriptor.set_attribute_handle(BackendAttributeName::ExecutionPlanHandle, ctx.as_raw())?;
        descriptor.finalize()?;

        Ok(Self {
            descriptor,
            cuda_ctx: Some(Arc::clone(ctx.cuda_context())),
        })
    }

    pub fn workspace_size(&self) -> Result<usize> {
        to_usize(
            self.descriptor
                .attribute_i64(BackendAttributeName::ExecutionPlanWorkspaceSize)?,
            "workspace_size",
        )
    }

    pub fn computed_intermediate_ids(&self) -> Result<Vec<i64>> {
        self.descriptor
            .attribute_i64_slice(BackendAttributeName::ExecutionPlanComputedIntermediateIds)
    }

    pub fn run_only_intermediate_ids(&self) -> Result<Vec<i64>> {
        self.descriptor
            .attribute_i64_slice(BackendAttributeName::ExecutionPlanRunOnlyIntermediateIds)
    }

    pub fn json_representation(&self) -> Result<String> {
        self.descriptor
            .attribute_char_string(BackendAttributeName::ExecutionPlanJsonRepresentation)
    }

    pub fn engine_config(&self) -> Result<EngineConfig> {
        let descriptor = self.descriptor.attribute_descriptor_of_type(
            BackendAttributeName::ExecutionPlanEngineConfig,
            BackendDescriptorType::EngineCfg,
        )?;
        let keepalive = Rc::new(BackendDescriptor::from_raw_finalized_borrowed(
            self.descriptor.as_raw(),
            BackendDescriptorType::ExecutionPlan,
        )?);
        Ok(EngineConfig::from_descriptor_keepalive(
            descriptor,
            self.cuda_ctx.as_ref().map(Arc::clone),
            keepalive,
        ))
    }

    pub fn tag(&self) -> Result<String> {
        let engine_config = self.engine_config()?;
        let mut tag = format!("eng{}", engine_config.engine_index()?);
        for choice in engine_config.knob_choices()? {
            tag.push_str(&format!("_k{}={}", choice.knob_type as i32, choice.value));
        }
        Ok(tag)
    }

    pub fn kernel_cache_json(&self) -> Result<Option<String>> {
        let descriptor = match self.descriptor.attribute_descriptor_of_type(
            BackendAttributeName::ExecutionPlanKernelCache,
            BackendDescriptorType::KernelCache,
        ) {
            Ok(descriptor) => descriptor,
            Err(Error::DescriptorAttributeNotFound(_)) => return Ok(None),
            Err(error) => return Err(error),
        };
        Ok(Some(descriptor.attribute_char_string(
            BackendAttributeName::KernelCacheJsonRepresentation,
        )?))
    }

    pub fn execute(&self, ctx: &Context, variant_pack: &VariantPack) -> Result<()> {
        self.ensure_context(ctx, "execution plan")?;
        backend_execute(ctx, self.descriptor(), variant_pack.descriptor())
    }

    pub fn populate_graph(
        &self,
        ctx: &Context,
        variant_pack: &VariantPack,
        graph: &Graph,
    ) -> Result<()> {
        self.ensure_context(ctx, "execution plan")?;
        self.descriptor
            .populate_graph(ctx, variant_pack.descriptor(), graph)
    }

    pub fn update_graph(
        &self,
        ctx: &Context,
        variant_pack: &VariantPack,
        graph: &Graph,
    ) -> Result<()> {
        self.ensure_context(ctx, "execution plan")?;
        self.descriptor
            .update_graph(ctx, variant_pack.descriptor(), graph)
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }

    pub(crate) fn cuda_context(&self) -> Option<&Arc<CudaContext>> {
        self.cuda_ctx.as_ref()
    }

    fn ensure_context(&self, ctx: &Context, name: &str) -> Result<()> {
        validate_context_pair(Some(ctx.cuda_context()), self.cuda_context(), name)
    }
}
