use std::{fmt, sync::Arc};

use singe_cuda::{
    submission::{AccessMode, Resources},
    types::DevicePtr,
    view::{DeviceRegionProvider, OwnedDeviceRegion},
};

use crate::{
    backend::tensor::TensorId,
    error::Result,
    frontend::{
        bindings::{BindingAccess, Bindings, binding_memory_info},
        error::Error as FrontendError,
        operation::TensorAccess,
        plan::RequiredTensor,
    },
};

pub struct RetainedRegion {
    memory: Box<dyn DeviceRegionProvider + Send + Sync>,
}

impl fmt::Debug for RetainedRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RetainedRegion")
            .field("region", &self.memory.region())
            .finish()
    }
}

impl Bindings<'_> {
    /// Binds a retained byte range in the tensor's native storage encoding.
    /// Supports packed encodings, subranges and stream-ordered allocations.
    /// Owned execution derives access declarations from the graph
    /// and retain this range independently of the returned output and graph replay handles.
    /// Cloning bindings shares the retained owner; rebinding releases only this binding's reference.
    /// `access` limits native uses, including captures and storage views resolved to this binding.
    ///
    /// This establishes ownership and capacity, not initialization, valid addressing metadata
    /// or permission for concurrent access. Execution retains its unsafe access contract.
    /// Effective layout capacity and alignment are checked when preparing execution.
    ///
    /// # Errors
    /// Returns an error for bindings without graph metadata, an unknown tensor or non-device placement.
    pub fn set_owned_region<O: DeviceRegionProvider + Send + Sync + 'static>(
        &mut self,
        id: TensorId,
        memory: OwnedDeviceRegion<O>,
        access: BindingAccess,
    ) -> Result<&mut Self> {
        let tensors = self
            .tensors
            .ok_or(FrontendError::GraphAwareBindingsRequired)?;
        let record = tensors.get(&id).ok_or(FrontendError::TensorNotFound(id))?;
        self.validate_device_placement(id)?;
        let owner = Arc::new(RetainedRegion {
            memory: Box::new(memory),
        });
        let region = owner.memory.region();
        let pointer = region.as_ptr();
        let mut info = binding_memory_info(region, record.tensor.data_type, access);
        info.owner = Some(owner);
        // SAFETY: the retained provider keeps this pointer live, and execution validates its effective layout.
        // Initialization, encoding and asynchronous access remain execution's explicit contract.
        unsafe { self.set_id(id, DevicePtr::from_raw(pointer.cast())) };
        self.memory_info.insert(id, info);
        Ok(self)
    }

    pub(crate) fn retain_resources(self, required: &[RequiredTensor], resources: &mut Resources) {
        for binding in required {
            let mode = match binding.access {
                TensorAccess::Read => AccessMode::Read,
                TensorAccess::Write => AccessMode::Write,
                TensorAccess::ReadWrite => AccessMode::ReadWrite,
            };
            if let Some(access) = self.binding_access(binding.id, mode) {
                resources.declare_access(access);
            }
            if let Some(owner) = self
                .memory_info
                .get(&binding.id)
                .and_then(|info| info.owner.as_ref())
            {
                resources.retain(Arc::clone(owner));
            }
        }
        resources.retain(Arc::new(self.scalar_storage));
    }
}
