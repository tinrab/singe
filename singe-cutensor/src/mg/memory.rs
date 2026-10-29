use std::{mem::MaybeUninit, ptr, sync::Arc};

use singe_cuda::{
    context::Context as CudaContext,
    error::{Error as CudaError, Result as CudaResult},
    memory::{DeviceMemory, HostAllocationFlags, PinnedHostMemory},
    view::{DeviceRepr, ZeroableDeviceRepr},
};

use crate::{
    error::{Error, Result},
    mg::{
        context::{Context, ContextRef},
        tensor::TensorDevice,
        workspace::Workspace,
    },
};

/// Owns one device allocation for each tensor pointer, in descriptor order.
#[derive(Debug)]
pub struct DistributedDeviceMemory<T> {
    buffers: Vec<DeviceMemory<T>>,
    devices: Vec<TensorDevice>,
    len_per_pointer: usize,
}

/// Owns per-context-device pending byte buffers and optional pinned host workspace.
#[derive(Debug)]
pub struct WorkspaceMemory {
    device: Vec<DeviceMemory<MaybeUninit<u8>>>,
    host: HostWorkspace,
}

#[derive(Debug)]
struct HostWorkspace {
    memory: Option<PinnedHostMemory<MaybeUninit<u8>>>,
}

impl<T> DistributedDeviceMemory<T> {
    /// Allocates initialized zero-filled partitions on the selected devices.
    ///
    /// `len_per_pointer` counts elements of `T`, not bytes. Each entry in `devices` receives a separate allocation in its retained CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if a host storage location is requested, a device does not participate in the context, a byte-size calculation overflows, or CUDA allocation or zero initialization fails.
    pub fn zeroed(
        context: &Context,
        devices: &[TensorDevice],
        len_per_pointer: usize,
    ) -> Result<Self>
    where
        T: ZeroableDeviceRepr,
    {
        let mut buffers = Vec::with_capacity(devices.len());
        for device in devices {
            let TensorDevice::Device(device_id) = *device else {
                return Err(Error::MgHostTensorMemoryUnsupported {
                    name: "distributed device memory".into(),
                });
            };
            buffers.push(DeviceMemory::<T>::zeroed(
                context.cuda_context(device_id)?,
                len_per_pointer,
            )?);
        }

        Ok(Self {
            buffers,
            devices: devices.to_vec(),
            len_per_pointer,
        })
    }

    /// Allocates uninitialized partitions on the selected devices.
    ///
    /// `len_per_pointer` counts elements of `T`, not bytes. Each entry in `devices` receives a separate allocation in its retained CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if a host storage location is requested, a device does not participate in the context, a byte-size calculation overflows, or CUDA allocation fails.
    pub fn create_uninit(
        context: &Context,
        devices: &[TensorDevice],
        len_per_pointer: usize,
    ) -> Result<DistributedDeviceMemory<MaybeUninit<T>>>
    where
        T: DeviceRepr,
    {
        let buffers = devices
            .iter()
            .map(|device| {
                let TensorDevice::Device(device_id) = *device else {
                    return Err(Error::MgHostTensorMemoryUnsupported {
                        name: "distributed device memory".into(),
                    });
                };
                Ok(DeviceMemory::<T>::create_uninit(
                    context.cuda_context(device_id)?,
                    len_per_pointer,
                )?)
            })
            .collect::<Result<_>>()?;
        Ok(DistributedDeviceMemory {
            buffers,
            devices: devices.to_vec(),
            len_per_pointer,
        })
    }

    /// Returns the device partitions in descriptor order.
    #[must_use]
    pub fn buffers(&self) -> &[DeviceMemory<T>] {
        &self.buffers
    }

    /// Returns mutable device partitions in descriptor order.
    pub fn buffers_mut(&mut self) -> &mut [DeviceMemory<T>] {
        &mut self.buffers
    }

    pub(crate) fn validate_context(&self, context: &ContextRef) -> Result<()> {
        for (buffer, device) in self.buffers.iter().zip(&self.devices) {
            let TensorDevice::Device(device_id) = *device else {
                return Err(Error::MgHostTensorMemoryUnsupported {
                    name: "distributed device memory".into(),
                });
            };
            if buffer.context().as_ref() != context.cuda_context(device_id)?.as_ref() {
                return Err(CudaError::MemoryContextMismatch.into());
            }
        }
        Ok(())
    }

    /// Returns the number of device allocation pointers.
    #[must_use]
    pub fn pointer_count(&self) -> usize {
        self.buffers.len()
    }

    /// Returns the element count requested for each allocation.
    #[must_use]
    pub fn len_per_pointer(&self) -> usize {
        self.len_per_pointer
    }

    /// Returns tensor storage locations in allocation and descriptor order.
    #[must_use]
    pub fn devices(&self) -> &[TensorDevice] {
        &self.devices
    }

    pub(crate) fn const_ptrs(&self) -> Vec<*const ()> {
        self.buffers
            .iter()
            .map(|buffer| buffer.as_ptr().cast())
            .collect()
    }

    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "writable native pointers require an exclusive workspace borrow"
    )]
    pub(crate) fn mut_ptrs(&mut self) -> Vec<*mut ()> {
        self.buffers
            .iter()
            .map(|buffer| buffer.as_mut_ptr().cast())
            .collect()
    }
}

impl<T> DistributedDeviceMemory<MaybeUninit<T>> {
    /// Marks every element of every device partition as initialized.
    ///
    /// # Safety
    ///
    /// Each element must contain a valid initialized `T`, with its producer
    /// ordered before every consumer on every participating device.
    #[must_use]
    pub unsafe fn assume_init(self) -> DistributedDeviceMemory<T> {
        DistributedDeviceMemory {
            buffers: self
                .buffers
                .into_iter()
                // SAFETY: the caller upholds this function's safety contract.
                .map(|buffer| unsafe { buffer.assume_init() })
                .collect(),
            devices: self.devices,
            len_per_pointer: self.len_per_pointer,
        }
    }
}

impl WorkspaceMemory {
    /// Allocates pending device byte buffers and pinned host storage for workspace requirements.
    ///
    /// Device buffers follow context device order. Pinned host storage uses the first participating device's CUDA context; zero host requirements use no allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if device requirements do not match the context device count, the context has no devices, a byte count does not fit `usize`, or CUDA device or pinned host allocation fails.
    pub fn create(context: &ContextRef, workspace: &Workspace) -> Result<Self> {
        workspace.validate_for_context(context)?;

        let device_sizes = workspace.device_sizes_bytes()?;
        let mut device = Vec::with_capacity(context.device_count());
        for (&device_id, &size) in context.devices().iter().zip(&device_sizes) {
            device.push(DeviceMemory::<u8>::create_uninit(
                context.cuda_context(device_id)?,
                size,
            )?);
        }

        Ok(Self {
            device,
            host: HostWorkspace::create(
                context.cuda_context(*context.devices().first().ok_or(CudaError::InvalidValue)?)?,
                workspace.host_size_bytes()?,
            )?,
        })
    }

    pub(crate) fn validate(&self, context: &ContextRef, required: &Workspace) -> Result<()> {
        required.validate_for_context(context)?;
        if self.device.len() != context.device_count() {
            return Err(Error::LengthMismatch {
                name: "workspace devices".into(),
                expected: context.device_count(),
                actual: self.device.len(),
            });
        }
        for ((buffer, &device), bytes) in self
            .device
            .iter()
            .zip(context.devices())
            .zip(required.device_sizes_bytes()?)
        {
            if buffer.context().as_ref() != context.cuda_context(device)?.as_ref() {
                return Err(CudaError::MemoryContextMismatch.into());
            }
            if buffer.len() < bytes {
                return Err(Error::InsufficientWorkspaceSize {
                    required: bytes as u64,
                    actual: buffer.len() as u64,
                });
            }
        }
        let bytes = required.host_size_bytes()?;
        let actual = self.host.memory.as_ref().map_or(0, PinnedHostMemory::len);
        if actual < bytes {
            return Err(Error::InsufficientWorkspaceSize {
                required: bytes as u64,
                actual: actual as u64,
            });
        }
        Ok(())
    }

    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "writable native pointers require an exclusive workspace borrow"
    )]
    pub(crate) fn device_mut_ptrs(&mut self) -> Vec<*mut ()> {
        self.device
            .iter()
            .map(|buffer| buffer.as_mut_ptr().cast())
            .collect()
    }

    pub(crate) fn device_pointer_count(&self) -> usize {
        self.device.len()
    }

    pub(crate) fn host_mut_ptr(&self) -> *mut () {
        self.host.as_mut_ptr()
    }
}

impl HostWorkspace {
    fn create(context: &Arc<CudaContext>, size: usize) -> CudaResult<Self> {
        let memory = if size == 0 {
            None
        } else {
            Some(PinnedHostMemory::<u8>::create_uninit(
                context,
                size,
                HostAllocationFlags::DEFAULT,
            )?)
        };
        Ok(Self { memory })
    }

    fn as_mut_ptr(&self) -> *mut () {
        self.memory
            .as_ref()
            .map_or_else(ptr::null_mut, |memory| memory.as_ptr().cast_mut().cast())
    }
}
