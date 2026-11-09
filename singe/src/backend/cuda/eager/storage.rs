use std::sync::Mutex;

use singe_cuda::memory::{HostAllocationFlags, PinnedHostMemory, peer::PeerCopy, transfer};

use crate::{
    backend::{
        cuda::{
            self, CudaStream,
            context::CudaExecutionContext,
            device_views::{tensor_output_view_with_len, tensor_view_with_len},
            stream::ensure_not_stream_capturing,
        },
        from_bytes_on,
    },
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    shape::Shape,
    storage::cpu::CpuStorage,
    tensor::Tensor,
    utility::{Locked, checked_mul},
};

pub fn cuda_zeroed_tensor(device: Device, data_type: DataType, shape: &Shape) -> Result<Tensor> {
    ensure_cuda_storage_device("cuda zeros device", device)?;
    ensure_not_stream_capturing(device, "cuda tensor allocation during stream capture")?;
    let context = cuda::execution_context(device)?;
    context.zeroed_tensor(data_type, shape)
}

pub fn tensor_from_bytes(
    device: Device,
    data_type: DataType,
    shape: &Shape,
    data: &[u8],
) -> Result<Tensor> {
    ensure_cuda_storage_device("cuda from_bytes device", device)?;
    let expected = checked_mul(
        shape.element_count(),
        data_type.size_bytes(),
        "cuda from_bytes expected byte size",
    )?;
    if data.len() != expected {
        return Err(Error::length_mismatch(
            "cuda from_bytes data",
            expected,
            data.len(),
        ));
    }

    ensure_not_stream_capturing(device, "cuda host copy during stream capture")?;
    let context = cuda::execution_context(device)?;
    let storage = context
        .allocator()
        .allocate_from_bytes_with_shape(data_type, shape, data)?;
    CudaExecutionContext::tensor_from_storage(storage, data_type, shape)
}

pub fn copy_tensor_to(src: &Tensor, device: Device) -> Result<Tensor> {
    ensure_not_stream_capturing(src.device(), "cuda device copy during stream capture")?;
    if device == src.device() {
        return copy_contiguous_tensor_same_device(&src.contiguous()?);
    }

    if src.device().is_cuda() && device.is_cuda() {
        let src = src.contiguous()?;
        return peer_copy(src.device(), device)?.map_or_else(
            || staged_copy(&src, device),
            |peer| copy_through_peer(&peer, &src, device),
        );
    }

    let data = src.to_bytes()?;
    from_bytes_on(device, src.data_type(), src.shape(), &data)
}

/// A source and destination device with their direct copy path, or `None` for a pair without peer access.
type PeerPath = ((Device, Device), Option<PeerCopy>);

/// The peer copy decision for each device pair, made once per pair.
static PEER_COPIES: Mutex<Vec<PeerPath>> = Mutex::new(Vec::new());

/// Returns the peer copy path from `source` to `destination`, enabling peer access the first time a pair supports it.
pub fn peer_copy(source: Device, destination: Device) -> Result<Option<PeerCopy>> {
    if let Some((_, path)) = PEER_COPIES
        .locked("peer copies")?
        .iter()
        .find(|(pair, _)| *pair == (source, destination))
    {
        return Ok(path.clone());
    }

    let source_context = cuda::execution_context(source)?;
    let destination_context = cuda::execution_context(destination)?;
    let (from, to) = (
        source_context.runtime().context(),
        destination_context.runtime().context(),
    );
    let path = if from.device().can_access_peer(to.device())? {
        // A failure to enable the path may pass, so this copy goes through the host and a later one tries again.
        let Ok(copy) = PeerCopy::create(from, to) else {
            return Ok(None);
        };
        Some(copy)
    } else {
        None
    };
    PEER_COPIES
        .locked("peer copies")?
        .push(((source, destination), path.clone()));
    Ok(path)
}

/// Copies a contiguous tensor to another device directly over peer access.
fn copy_through_peer(peer: &PeerCopy, src: &Tensor, device: Device) -> Result<Tensor> {
    let context = cuda::execution_context(device)?;
    let output = context.allocate_output(src.data_type(), src.shape())?;
    let byte_size = src.byte_size()?;
    if byte_size == 0 {
        // SAFETY: the empty output has no elements to initialize.
        return Ok(unsafe { output.assume_init() });
    }

    let src_memory =
        tensor_view_with_len::<u8>(src, src.data_type(), byte_size, "cuda peer copy input")?;
    let mut output_memory = tensor_output_view_with_len::<u8>(
        output.region(),
        src.data_type(),
        byte_size,
        "cuda peer copy output",
    )?;
    if let Some(stream) = context.stream() {
        peer.copy(&mut output_memory, &src_memory, stream.raw())?;
    } else {
        let stream = CudaStream::create(device)?;
        peer.copy(&mut output_memory, &src_memory, stream.raw())?;
    }
    // SAFETY: the completed copy writes every byte with the source's valid representation.
    Ok(unsafe { output.assume_init() })
}

/// Copies a contiguous tensor to another device through a page-locked host buffer, for device pairs without peer access.
pub fn staged_copy(src: &Tensor, device: Device) -> Result<Tensor> {
    let stream = CudaStream::create(src.device())?;
    let host = src
        .copy_to_non_blocking(
            Tensor::zeros_pinned(src.data_type(), src.shape().clone())?,
            &stream,
        )?
        .wait()?;
    host.to_device(device)
}

fn copy_contiguous_tensor_same_device(src: &Tensor) -> Result<Tensor> {
    let context = cuda::execution_context(src.device())?;
    context.bind()?;
    let output = context.allocate_output(src.data_type(), src.shape())?;
    let byte_size = src.byte_size()?;
    if byte_size == 0 {
        // SAFETY: the empty output has no elements to initialize.
        return Ok(unsafe { output.assume_init() });
    }

    // The tensors own these CUDA allocations while the non-owning views are used only to submit this device-to-device copy.
    let src_memory =
        tensor_view_with_len::<u8>(src, src.data_type(), byte_size, "cuda copy input")?;
    let mut output_memory = tensor_output_view_with_len::<u8>(
        output.region(),
        src.data_type(),
        byte_size,
        "cuda copy output",
    )?;

    context.sync_scope(|| {
        if let Some(stream) = context.stream() {
            // SAFETY: the source is initialized and the draining scope retains both borrows.
            unsafe {
                transfer::copy_device_async_unchecked(
                    &mut output_memory,
                    &src_memory,
                    stream.raw(),
                )?;
            }
        } else {
            transfer::copy_device(&mut output_memory, &src_memory)?;
        }
        Ok(())
    })?;
    // SAFETY: the completed copy writes every byte with the source's valid representation.
    Ok(unsafe { output.assume_init() })
}

fn ensure_cuda_storage_device(context: &'static str, device: Device) -> Result<()> {
    if device.is_cuda() {
        return Ok(());
    }
    Err(Error::device_mismatch(context, Device::Cuda(0), device))
}

/// Copies `data`, the native-endian bytes of `shape` elements of `data_type`, into a new page-locked CPU tensor.
///
/// The allocation is portable, so every CUDA context reads and writes it without staging.
pub fn pinned_tensor(data_type: DataType, shape: &Shape, data: &[u8]) -> Result<Tensor> {
    let expected = checked_mul(
        shape.element_count(),
        data_type.size_bytes(),
        "cuda pinned tensor byte size",
    )?;
    if data.len() != expected {
        return Err(Error::length_mismatch(
            "cuda pinned tensor data",
            expected,
            data.len(),
        ));
    }

    let context = cuda::execution_context(Device::Cuda(0))?;
    let memory = PinnedHostMemory::from_slice_with_flags(
        context.runtime().context(),
        data,
        HostAllocationFlags::PORTABLE,
    )?;
    let storage = CpuStorage::create_pinned(data_type, memory)?;
    // SAFETY: the storage holds exactly the validated bytes of this shape and data type.
    Ok(unsafe { Tensor::from_parts_unchecked(storage, data_type, shape.clone()) })
}
