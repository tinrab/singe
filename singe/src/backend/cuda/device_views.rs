use std::mem::size_of;

use singe_cuda::{
    error::Error as CudaError,
    stream::{Stream, StreamCaptureStatus},
    view::{
        AnyBitPatternDeviceRepr, DeviceInputView, DeviceOutputView, DeviceRegion, DeviceRepr,
        DeviceView,
    },
};

use crate::{
    backend::cuda::{
        context::CudaExecutionContext,
        stream::CudaStream,
        tensor_region::{CudaTensorRegion, CudaTensorSource},
        utility::cuda_storage,
    },
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    tensor::Tensor,
    utility::{checked_add, checked_mul},
};

pub fn f32_storage_view(
    tensor: &(impl CudaTensorSource + ?Sized),
) -> Result<DeviceInputView<'_, f32>> {
    tensor_input_storage_view::<f32>(tensor, DataType::F32, "cuda kernel f32 storage")
}

pub fn tensor_input_view<'a, T: DeviceRepr>(
    tensor: &'a (impl CudaTensorSource + ?Sized),
    data_type: DataType,
    name: &'static str,
) -> Result<DeviceInputView<'a, T>> {
    tensor_input_view_with_len(tensor, data_type, tensor.element_count(), name)
}

pub fn tensor_input_view_with_len<'a, T: DeviceRepr>(
    tensor: &'a (impl CudaTensorSource + ?Sized),
    data_type: DataType,
    length: usize,
    name: &'static str,
) -> Result<DeviceInputView<'a, T>> {
    if tensor.data_type() != data_type {
        return Err(Error::data_type_mismatch(
            name,
            data_type,
            tensor.data_type(),
        ));
    }

    if !tensor.is_contiguous() {
        return Err(Error::ContiguousRequired);
    }

    let bytes = checked_mul(length, size_of::<T>(), "cuda input view bytes")?;
    Ok(DeviceInputView::create(
        tensor.tensor_region()?.region().slice(..bytes)?,
    )?)
}

pub fn tensor_input_storage_view<'a, T: DeviceRepr>(
    tensor: &'a (impl CudaTensorSource + ?Sized),
    data_type: DataType,
    name: &'static str,
) -> Result<DeviceInputView<'a, T>> {
    if tensor.data_type() != data_type {
        return Err(Error::data_type_mismatch(
            name,
            data_type,
            tensor.data_type(),
        ));
    }
    Ok(DeviceInputView::create(tensor.tensor_region()?.region())?)
}

pub fn tensor_view_with_len<'a, T: AnyBitPatternDeviceRepr>(
    tensor: &'a Tensor,
    data_type: DataType,
    view_element_count: usize,
    name: &'static str,
) -> Result<DeviceView<'a, T>> {
    let region = tensor_region::<T>(tensor, data_type, view_element_count, name)?;
    // SAFETY: CudaStorage owns initialized bytes, and every pattern is valid for T.
    // The checked constructor establishes the requested alignment and byte span.
    Ok(unsafe { DeviceView::from_raw_parts(region, view_element_count)? })
}

fn tensor_region<'a, T: DeviceRepr>(
    tensor: &'a Tensor,
    data_type: DataType,
    view_element_count: usize,
    name: &'static str,
) -> Result<DeviceRegion<'a>> {
    if !tensor.device().is_cuda() {
        return Err(Error::device_mismatch(
            name,
            Device::Cuda(0),
            tensor.device(),
        ));
    }

    if tensor.data_type() != data_type {
        return Err(Error::data_type_mismatch(
            name,
            data_type,
            tensor.data_type(),
        ));
    }

    if !tensor.is_contiguous() {
        return Err(Error::unsupported_operation_with_details(
            name,
            "non-contiguous tensor is not supported",
        ));
    }

    let storage = cuda_storage(tensor)?;
    let tensor_byte_size = tensor.byte_size()?;
    let view_byte_size = checked_mul(
        view_element_count,
        size_of::<T>(),
        "cuda kernel view byte size",
    )?;
    let expected_byte_size = checked_mul(
        tensor.element_count(),
        data_type.size_bytes(),
        "cuda kernel expected byte size",
    )?;
    if tensor_byte_size != expected_byte_size {
        return Err(Error::byte_length_mismatch(
            name,
            data_type.size_bytes(),
            tensor_byte_size,
        ));
    }

    let byte_offset = checked_mul(
        tensor.storage_offset(),
        data_type.size_bytes(),
        "cuda kernel tensor storage byte offset",
    )?;
    let byte_end = checked_add(
        byte_offset,
        view_byte_size,
        "cuda kernel tensor storage byte span",
    )?;
    if storage.byte_size() < byte_end {
        return Err(Error::length_mismatch(name, byte_end, storage.byte_size()));
    }

    storage.region(byte_offset..byte_end)
}

pub fn tensor_output_view<'a, T: DeviceRepr>(
    tensor: CudaTensorRegion<'a>,
    data_type: DataType,
    name: &'static str,
) -> Result<DeviceOutputView<'a, T>> {
    tensor_output_view_with_len(tensor, data_type, tensor.element_count(), name)
}

pub fn tensor_output_view_with_len<'a, T: DeviceRepr>(
    tensor: CudaTensorRegion<'a>,
    data_type: DataType,
    length: usize,
    name: &'static str,
) -> Result<DeviceOutputView<'a, T>> {
    if tensor.data_type() != data_type {
        return Err(Error::data_type_mismatch(
            name,
            data_type,
            tensor.data_type(),
        ));
    }

    if !tensor.is_contiguous() {
        return Err(Error::ContiguousRequired);
    }

    let bytes = checked_mul(length, size_of::<T>(), "cuda output view bytes")?;
    let region = tensor.region().slice(..bytes)?;
    // SAFETY: backend execution grants this operation exclusive access to its output range.
    Ok(unsafe { DeviceOutputView::from_raw_parts(region, length)? })
}

pub enum CudaKernelStream<'a> {
    Borrowed(&'a Stream),
    Owned(CudaStream),
}

impl CudaKernelStream<'_> {
    pub(crate) fn raw(&self) -> &Stream {
        match self {
            Self::Borrowed(stream) => stream,
            Self::Owned(stream) => stream.raw(),
        }
    }

    /// Completes borrowed eager work or records it into the backend's active graph.
    ///
    /// # Safety
    ///
    /// The launch closure must honor its kernel's layout and index contracts.
    /// During capture the caller must retain every input and output allocation through graph replay, order accesses, and exclude conflicting output access.
    pub(crate) unsafe fn complete_or_record<'a, S: CudaTensorSource + ?Sized + 'a>(
        &self,
        inputs: impl IntoIterator<Item = &'a S>,
        outputs: &[CudaTensorRegion<'_>],
        launch: impl FnOnce() -> Result<()>,
    ) -> Result<()> {
        let stream = self.raw();
        if outputs
            .iter()
            .any(|output| output.region().context() != stream.context())
        {
            return Err(CudaError::MemoryContextMismatch.into());
        }

        let capturing = stream.capture_status()? != StreamCaptureStatus::None;
        for input in inputs {
            let input = input.tensor_region()?.region();
            if input.context() != stream.context() {
                return Err(CudaError::MemoryContextMismatch.into());
            }

            if outputs.iter().any(|output| input.overlaps(output.region())) {
                return Err(Error::invalid_argument(
                    "cuda kernel input and output overlap",
                ));
            }
            if !capturing
                && input
                    .stream()
                    .is_some_and(|ready| ready.as_raw() != stream.as_raw())
            {
                input.order_on(stream)?;
            }
        }
        if capturing {
            launch()
        } else {
            // The caller's tensor owners remain borrowed while the core scope drains success, errors, and unwinding.
            stream.sync_scope(|_| Ok(launch()))?
        }
    }
}

pub fn kernel_stream_for_context(
    context: &CudaExecutionContext,
    device: Device,
) -> Result<CudaKernelStream<'_>> {
    if context.device() != device {
        return Err(Error::device_mismatch(
            "cuda kernel context device",
            context.device(),
            device,
        ));
    }
    match context.stream() {
        Some(stream) => Ok(CudaKernelStream::Borrowed(stream.raw())),
        None => Ok(CudaKernelStream::Owned(CudaStream::create(device)?)),
    }
}
