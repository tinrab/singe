use crate::{
    attribute::BackendAttributeName,
    descriptor::{BackendDescriptor, BackendDescriptorType},
    error::Result,
    execution::OperationGraph,
};

#[derive(Debug)]
pub struct KernelCache {
    descriptor: BackendDescriptor,
}

impl KernelCache {
    pub fn create() -> Result<Self> {
        Ok(Self {
            descriptor: BackendDescriptor::create(BackendDescriptorType::KernelCache)?,
        })
    }

    pub fn from_json(json_cache: &str) -> Result<Self> {
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::KernelCache)?;
        descriptor.set_attribute_char_string(
            BackendAttributeName::KernelCacheJsonRepresentation,
            json_cache,
        )?;
        Ok(Self { descriptor })
    }

    pub fn is_engine_config_cached(&self) -> Result<bool> {
        self.descriptor
            .attribute_bool(BackendAttributeName::KernelCacheIsEngineCfgKernelCached)
    }

    pub fn json_representation(&self) -> Result<String> {
        self.descriptor
            .attribute_char_string(BackendAttributeName::KernelCacheJsonRepresentation)
    }

    pub fn build(&mut self, operation_graph: &OperationGraph) -> Result<()> {
        if !self.descriptor.is_null() && self.descriptor.is_finalized() {
            return Ok(());
        }

        self.descriptor.set_attribute_descriptor(
            BackendAttributeName::KernelCacheOperationGraph,
            operation_graph.descriptor(),
        )?;
        self.descriptor.finalize()
    }

    pub fn is_finalized(&self) -> bool {
        self.descriptor.is_finalized()
    }

    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}
