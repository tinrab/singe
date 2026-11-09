use crate::{
    backend::from_bytes_on,
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    shape::Shape,
    storage::cpu::CpuStorage,
    tensor::Tensor,
    utility::{checked_add, checked_add_i64, checked_mul, to_usize},
};

/// Allocates a contiguous zero-filled tensor on the CPU.
///
/// # Errors
///
/// Returns an error if `device` is not a CPU device or the shape byte count overflows.
pub fn zeros(device: Device, data_type: DataType, shape: &Shape) -> Result<Tensor> {
    if !device.is_cpu() {
        return Err(Error::device_mismatch("device", Device::Cpu, device));
    }

    shape.byte_size(data_type.size_bytes())?;
    let storage = CpuStorage::create_zeroed(device, data_type, shape.element_count())?;
    // SAFETY: the storage was built from values computed for exactly this shape and data type.
    Ok(unsafe { Tensor::from_parts_unchecked(storage, data_type, shape.clone()) })
}

/// Copies native-endian host bytes into a contiguous CPU tensor.
/// Boolean bytes must be zero or one.
///
/// # Errors
///
/// Returns an error if `device` is not a CPU device, the shape byte count overflows, the byte length differs from that count, or a boolean byte is invalid.
pub fn create_from_bytes(
    device: Device,
    data_type: DataType,
    shape: &Shape,
    data: &[u8],
) -> Result<Tensor> {
    if !device.is_cpu() {
        return Err(Error::device_mismatch("device", Device::Cpu, device));
    }

    let expected = shape.byte_size(data_type.size_bytes())?;
    if data.len() != expected {
        return Err(Error::length_mismatch("data", expected, data.len()));
    }

    let storage = CpuStorage::create_from_bytes(device, data_type, data)?;
    // SAFETY: the storage was built from values computed for exactly this shape and data type.
    Ok(unsafe { Tensor::from_parts_unchecked(storage, data_type, shape.clone()) })
}

/// Copies logical tensor values into storage on `device`, reading contiguous storage in place.
///
/// # Errors
///
/// Returns an error if exporting source values fails, the destination backend has no storage implementation, or destination allocation or byte import fails.
pub fn copy_to(src: &Tensor, device: Device) -> Result<Tensor> {
    // Contiguous storage is copied from in place, so a pinned source reaches CUDA without staging.
    if src.is_contiguous()
        && let Some(storage) = src.storage().as_cpu()
    {
        let size = src.data_type().size_bytes();
        let start = src.storage_offset() * size;
        let length = src.shape().byte_size(size)?;
        let bytes = storage.as_slice()?;
        return from_bytes_on(
            device,
            src.data_type(),
            src.shape(),
            &bytes[start..start + length],
        );
    }
    from_bytes_on(device, src.data_type(), src.shape(), &src.to_bytes()?)
}

pub fn contiguous(input: &Tensor) -> Result<Tensor> {
    let element_count = input.shape().element_count();
    let data_type_size = input.data_type().size_bytes();
    let byte_size = checked_mul(element_count, data_type_size, "cpu contiguous byte size")?;
    let mut new_bytes = vec![0u8; byte_size];

    let src_bytes = input.storage().byte_size();
    let mut src = vec![0u8; src_bytes];
    input.storage().copy_to_host(&mut src)?;

    let rank = to_usize(input.shape().rank(), "cpu contiguous rank")?;
    let dims = input.shape().dimensions();
    let strides = input.layout().strides();
    let offset = input.storage_offset();

    let mut indices = vec![0i64; rank];
    for logical_index in 0..element_count {
        let source_element = source_element_offset(strides, offset, &indices)?;
        copy_element_bytes(
            input.data_type().size_bytes(),
            logical_index,
            source_element,
            &src,
            &mut new_bytes,
        )?;
        advance_indices(&mut indices, dims)?;
    }

    let new_shape = Shape::create(input.shape().dimensions().to_vec())?;
    create_from_bytes(input.device(), input.data_type(), &new_shape, &new_bytes)
}

fn source_element_offset(strides: &[i64], base_offset: usize, indices: &[i64]) -> Result<usize> {
    let mut offset = base_offset;
    for (axis, index) in indices.iter().enumerate() {
        let index = to_usize(*index, &format!("cpu contiguous index axis {axis}"))?;
        let stride = usize::try_from(strides[axis]).map_err(|_| Error::InvalidStride)?;
        let axis_offset = checked_mul(index, stride, "cpu contiguous source offset")?;
        offset = checked_add(offset, axis_offset, "cpu contiguous source offset")?;
    }
    Ok(offset)
}

fn copy_element_bytes(
    data_type_size: usize,
    logical_index: usize,
    source_element: usize,
    source: &[u8],
    destination: &mut [u8],
) -> Result<()> {
    let destination_start = checked_mul(
        logical_index,
        data_type_size,
        "cpu contiguous destination byte offset",
    )?;
    let source_start = checked_mul(
        source_element,
        data_type_size,
        "cpu contiguous source byte offset",
    )?;
    let source_end = checked_add(
        source_start,
        data_type_size,
        "cpu contiguous source byte offset",
    )?;
    destination[destination_start..destination_start + data_type_size]
        .copy_from_slice(&source[source_start..source_end]);
    Ok(())
}

fn advance_indices(indices: &mut [i64], dimensions: &[i64]) -> Result<()> {
    for axis in (0..indices.len()).rev() {
        indices[axis] = checked_add_i64(indices[axis], 1, "cpu contiguous index")?;
        if indices[axis] < dimensions[axis] {
            break;
        }

        indices[axis] = 0;
    }
    Ok(())
}

/// Copies `source` into the elements `destination` addresses.
///
/// The caller has checked that both tensors share shape and data type, that `destination` has no internal overlap, and that `source` does not share its storage.
pub fn assign(destination: &Tensor, source: &Tensor) -> Result<()> {
    let element_count = destination.shape().element_count();
    if element_count == 0 {
        return Ok(());
    }

    let size = destination.data_type().size_bytes();
    let source_storage = cpu_storage(source)?;
    let destination_storage = cpu_storage(destination)?;
    let source_bytes = source_storage.as_slice()?;
    let mut destination_bytes = destination_storage.write_bytes()?;
    for linear in 0..element_count {
        let from = source.layout().linear_offset(linear) * size;
        let to = destination.layout().linear_offset(linear) * size;
        destination_bytes[to..to + size].copy_from_slice(&source_bytes[from..from + size]);
    }

    drop(destination_bytes);
    drop(source_bytes);
    Ok(())
}

fn cpu_storage(tensor: &Tensor) -> Result<&CpuStorage> {
    tensor
        .storage()
        .as_cpu()
        .ok_or_else(|| Error::unsupported_operation("cpu assign storage"))
}
