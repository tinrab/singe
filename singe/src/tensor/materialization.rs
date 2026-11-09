use std::sync::Arc;

#[cfg(feature = "cuda")]
use crate::{
    backend::cuda::{CudaStream, CudaTransfer, pinned_tensor},
    data_type::DataType,
    error::Error,
    shape::Shape,
};
use crate::{
    backend::registry::{Operation, OperationAttributes},
    device::Device,
    dispatch::call_one,
    error::Result,
    storage::{Storage, cpu::CpuStorage, meta::MetaStorage},
    tensor::Tensor,
};

impl Tensor {
    /// Copies logical values into an independent contiguous allocation.
    /// Clone, views, and same-device `to_device` calls may share storage.
    ///
    /// # Errors
    ///
    /// Returns an error if graph capture rejects the copy, backend storage support is unavailable, or value readback, allocation, or data transfer fails.
    pub fn copy(&self) -> Result<Self> {
        call_one(Operation::Copy, OperationAttributes::None, &[self])
    }

    /// Copies or aliases the tensor onto `device`.
    ///
    /// This records a graph operation when called inside graph capture.
    ///
    /// # Errors
    ///
    /// Returns an error if the source or destination backend is unsupported or the device copy fails.
    pub fn to_device(&self, device: Device) -> Result<Self> {
        call_one(
            Operation::CopyTo,
            OperationAttributes::CopyTo(device),
            &[self],
        )
    }

    // Returns a metadata-only tensor with this tensor's data type and layout, reading none of its values.
    pub fn to_meta(&self) -> Result<Self> {
        Self::create_from_storage(
            Arc::new(Storage::Meta(MetaStorage::new(self.storage().byte_size()))),
            self.data_type(),
            self.layout().clone(),
        )
    }

    /// Returns a contiguous, zero-offset tensor.
    ///
    /// This returns `self.clone()` when the tensor is already contiguous.
    ///
    /// # Errors
    ///
    /// Returns an error if materializing a contiguous copy fails.
    pub fn contiguous(&self) -> Result<Self> {
        let byte_size = self.shape().byte_size(self.data_type().size_bytes())?;
        if self.is_contiguous()
            && self.layout().storage_offset() == 0
            && self.inner.storage.byte_size() == byte_size
        {
            return Ok(self.clone());
        }
        call_one(Operation::Contiguous, OperationAttributes::None, &[self])
    }
}

impl Tensor {
    /// Returns whether the tensor's storage is page-locked host memory from `Tensor::pin_memory` or `Tensor::zeros_pinned`, which CUDA builds provide.
    #[must_use]
    pub fn is_pinned(&self) -> bool {
        self.storage().as_cpu().is_some_and(CpuStorage::is_pinned)
    }

    /// Copies a CPU tensor's logical values into page-locked host memory, or returns `self.clone()` when it is already pinned.
    ///
    /// A pinned tensor is an ordinary CPU tensor to every operation; CUDA copies read and write it without staging, and non-blocking transfers require it.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is not on the CPU, its values cannot be read, or CUDA cannot allocate page-locked memory.
    #[cfg(feature = "cuda")]
    pub fn pin_memory(&self) -> Result<Self> {
        if self.device() != Device::Cpu {
            return Err(Error::device_mismatch(
                "pin_memory input",
                Device::Cpu,
                self.device(),
            ));
        }

        if self.is_pinned() {
            return Ok(self.clone());
        }
        pinned_tensor(self.data_type(), self.shape(), &self.to_bytes()?)
    }

    /// Allocates a zero-filled, page-locked CPU tensor, such as the destination of a non-blocking device-to-host copy.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte size overflows or CUDA cannot allocate page-locked memory.
    #[cfg(feature = "cuda")]
    pub fn zeros_pinned(data_type: DataType, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        let bytes = vec![0_u8; shape.byte_size(data_type.size_bytes())?];
        pinned_tensor(data_type, &shape, &bytes)
    }
}

#[cfg(feature = "cuda")]
impl Tensor {
    /// Starts copying this tensor to `device` on `stream` and returns while the copy runs.
    ///
    /// A pinned CPU tensor copies to a new tensor on `stream`'s device, a CUDA tensor copies to a new pinned CPU tensor, and a CUDA tensor copies to a new tensor on `stream`'s device when that is its own device or a peer.
    /// The tensor is taken by value: a pinned source must own its storage, so host code cannot write it during the copy, and [`CudaTransfer::wait_with_source`] returns it for reuse.
    ///
    /// # Errors
    ///
    /// Returns an error if the devices are neither pinned host memory and CUDA nor CUDA devices with peer access, a pinned source is shared or strided, `stream` belongs to another device, the stream is capturing, or allocation or the copy submission fails.
    pub fn to_device_non_blocking(
        self,
        device: Device,
        stream: &CudaStream,
    ) -> Result<CudaTransfer> {
        CudaTransfer::start(self, device, stream)
    }

    /// Starts copying this CUDA tensor into `destination`, a pinned CPU tensor of the same shape and data type, on `stream`.
    ///
    /// Reusing one destination across steps avoids allocating page-locked memory for each copy; [`CudaTransfer::wait`] returns it.
    ///
    /// # Errors
    ///
    /// Returns an error if `destination` is not a contiguous pinned tensor that owns its storage, the shapes or data types differ, `stream` belongs to another device, the stream is capturing, or the copy submission fails.
    pub fn copy_to_non_blocking(
        &self,
        destination: Self,
        stream: &CudaStream,
    ) -> Result<CudaTransfer> {
        CudaTransfer::start_into(self, destination, stream)
    }
}
