//! Non-blocking copies between page-locked host tensors and CUDA tensors.

use std::sync::Arc;

use singe_core::ErrorChain;
use singe_cuda::{
    memory::{DeviceMemory, MemoryCopyKind},
    stream::StreamCaptureStatus,
};

use crate::{
    backend::cuda::{
        self, EventTiming,
        eager::storage::peer_copy,
        event::CudaEvent,
        stream::CudaStream,
        tensor_region::{CudaTensorSource, PendingCudaTensor},
    },
    device::Device,
    error::{Error, Result},
    tensor::Tensor,
};

/// A copy between a pinned host tensor and a CUDA tensor that runs on a stream while the host continues.
///
/// The transfer owns both tensors until the copy completes, so neither can be read, written, or freed while the copy runs.
/// [`CudaTransfer::wait`] hands out the destination, and dropping an unfinished transfer waits for it.
#[derive(Debug)]
pub struct CudaTransfer {
    event: CudaEvent,
    pending: Option<Pending>,
}

#[derive(Debug)]
struct Pending {
    source: Tensor,
    destination: Destination,
}

#[derive(Debug)]
enum Destination {
    Device(PendingCudaTensor),
    Host(Tensor),
}

impl CudaTransfer {
    /// Returns the event recorded after the copy, which other streams can wait for.
    #[must_use]
    pub fn event(&self) -> &CudaEvent {
        &self.event
    }

    /// Returns whether the copy has completed.
    ///
    /// # Errors
    ///
    /// Returns an error if the copy failed.
    pub fn is_complete(&self) -> Result<bool> {
        self.event.query()
    }

    /// Waits for the copy and returns the destination tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if the copy failed.
    pub fn wait(self) -> Result<Tensor> {
        self.wait_with_source().map(|(destination, _)| destination)
    }

    /// Waits for the copy and returns the destination and source tensors, so a pinned buffer can be filled again.
    ///
    /// # Errors
    ///
    /// Returns an error if the copy failed.
    pub fn wait_with_source(mut self) -> Result<(Tensor, Tensor)> {
        self.event.synchronize()?;
        let Pending {
            source,
            destination,
        } = self
            .pending
            .take()
            .ok_or_else(|| Error::missing_input("wait with source"))?;
        let destination = match destination {
            // SAFETY: the completed copy wrote every byte of the destination from a tensor of the same data type.
            Destination::Device(pending) => unsafe { pending.assume_init() },
            Destination::Host(tensor) => tensor,
        };
        Ok((destination, source))
    }

    /// Starts copying a pinned CPU `source` to a new tensor on `stream`'s device, a CUDA `source` to a new pinned CPU tensor, or a CUDA `source` to a new tensor on the same device or a peer device.
    pub(crate) fn start(source: Tensor, device: Device, stream: &CudaStream) -> Result<Self> {
        match (source.device(), device) {
            (Device::Cpu, Device::Cuda(_)) => {
                ensure_stream_device(stream, device)?;
                ensure_exclusive_pinned(&source, "non-blocking copy source")?;
                let context = cuda::execution_context(device)?;
                let destination = context.allocate_output(source.data_type(), source.shape())?;
                Self::enqueue(source, Destination::Device(destination), stream)
            }
            (Device::Cuda(_), Device::Cpu) => {
                let destination = Tensor::zeros_pinned(source.data_type(), source.shape().clone())?;
                Self::start_into(&source, destination, stream)
            }
            (Device::Cuda(_), Device::Cuda(_)) => {
                ensure_stream_device(stream, device)?;
                if source.device() != device && peer_copy(source.device(), device)?.is_none() {
                    return Err(Error::unsupported_operation_with_details(
                        "cuda non-blocking copy",
                        format!("{} cannot access {device} directly", source.device()),
                    ));
                }

                let context = cuda::execution_context(device)?;
                let destination = context.allocate_output(source.data_type(), source.shape())?;
                Self::enqueue(
                    source.contiguous()?,
                    Destination::Device(destination),
                    stream,
                )
            }
            (from, to) => Err(Error::unsupported_operation_with_details(
                "cuda non-blocking copy",
                format!("copies run between pinned host memory and cuda, not {from} to {to}"),
            )),
        }
    }

    /// Starts copying a CUDA `source` into `destination`, a pinned CPU tensor of the same shape and data type that owns its storage.
    pub(crate) fn start_into(
        source: &Tensor,
        destination: Tensor,
        stream: &CudaStream,
    ) -> Result<Self> {
        ensure_stream_device(stream, source.device())?;
        ensure_exclusive_pinned(&destination, "non-blocking copy destination")?;
        if source.data_type() != destination.data_type() {
            return Err(Error::data_type_mismatch(
                "non-blocking copy destination",
                source.data_type(),
                destination.data_type(),
            ));
        }

        if source.shape() != destination.shape() {
            return Err(Error::shape_mismatch(
                "non-blocking copy destination",
                source.shape().dimensions().to_vec(),
                destination.shape().dimensions().to_vec(),
            ));
        }
        Self::enqueue(source.contiguous()?, Destination::Host(destination), stream)
    }

    fn enqueue(source: Tensor, mut destination: Destination, stream: &CudaStream) -> Result<Self> {
        if stream.raw().capture_status()? != StreamCaptureStatus::None {
            return Err(Error::unsupported_operation(
                "cuda non-blocking copy during stream capture",
            ));
        }

        let bytes = source.shape().byte_size(source.data_type().size_bytes())?;
        let (target, origin, kind) = match &mut destination {
            Destination::Device(pending) if source.device().is_cuda() => (
                pending.region().region().as_ptr(),
                source.tensor_region()?.region().as_ptr().cast_const(),
                MemoryCopyKind::DeviceToDevice,
            ),
            Destination::Device(pending) => (
                pending.region().region().as_ptr(),
                host_pointer(&source)?,
                MemoryCopyKind::HostToDevice,
            ),
            Destination::Host(tensor) => (
                host_pointer(tensor)?.cast_mut(),
                source.tensor_region()?.region().as_ptr().cast_const(),
                MemoryCopyKind::DeviceToHost,
            ),
        };
        // The destination's allocation is ordered on the context's stream, so the copy waits for that stream first.
        let context = cuda::execution_context(stream.device())?;
        if let Some(allocation_stream) = context.stream() {
            let ready = CudaEvent::create(stream.device(), EventTiming::Disabled)?;
            ready.record(allocation_stream)?;
            ready.wait(stream)?;
        }

        // SAFETY: both ranges hold `bytes` contiguous bytes of exclusively owned tensors, which the transfer retains until the copy completes; `Drop` waits for an unfinished copy.
        unsafe { DeviceMemory::<u8>::copy_async(target, origin, bytes, kind, stream.raw())? };
        let event = CudaEvent::create(stream.device(), EventTiming::Disabled)?;
        event.record(stream)?;
        Ok(Self {
            event,
            pending: Some(Pending {
                source,
                destination,
            }),
        })
    }
}

impl Drop for CudaTransfer {
    fn drop(&mut self) {
        if self.pending.is_some()
            && let Err(error) = self.event.synchronize()
        {
            eprintln!(
                "failed to wait for a dropped cuda transfer: {}",
                ErrorChain(&error)
            );
        }
    }
}

fn ensure_stream_device(stream: &CudaStream, device: Device) -> Result<()> {
    if stream.device() == device {
        Ok(())
    } else {
        Err(Error::device_mismatch(
            "cuda non-blocking copy stream",
            device,
            stream.device(),
        ))
    }
}

/// Checks that `tensor` is a contiguous pinned CPU tensor that no other tensor shares, so host code cannot touch its bytes during the copy.
fn ensure_exclusive_pinned(tensor: &Tensor, name: &'static str) -> Result<()> {
    if !tensor.is_pinned() || !tensor.is_contiguous() {
        return Err(Error::invalid_argument(format!(
            "`{name}` pinned contiguous cpu tensor"
        )));
    }

    if Arc::strong_count(&tensor.inner) != 1 || Arc::strong_count(tensor.storage()) != 1 {
        return Err(Error::invalid_argument(format!(
            "`{name}` storage ownership"
        )));
    }
    Ok(())
}

/// Returns the address of a contiguous CPU tensor's first logical byte.
fn host_pointer(tensor: &Tensor) -> Result<*const u8> {
    let storage = tensor
        .storage()
        .as_cpu()
        .ok_or_else(|| Error::unsupported_operation("cuda non-blocking copy host storage"))?;
    let offset = tensor.storage_offset() * tensor.data_type().size_bytes();
    Ok(storage.as_ptr()?.wrapping_add(offset))
}
