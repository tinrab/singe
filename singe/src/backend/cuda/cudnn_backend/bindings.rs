use std::sync::Arc;

use singe_cuda::{
    context::Context as CudaContext, error::Error as CudaError, types::DevicePtr,
    view::DeviceRegion,
};
use singe_cudnn::{
    backend::tensor::BackendTensorUid as CudnnTensorUid, error::Error as CudnnError,
};

use crate::{
    backend::cuda::{
        cudnn_backend::{data_type::storage_type, types::CudnnBackendGraphTemplate},
        utility::device_ptr,
    },
    data_type::DataType,
    error::Result,
};

/// Validates native layout requirements before registering an eager or replay address.
pub fn cudnn_backend_tensor_binding(
    template: &CudnnBackendGraphTemplate,
    context: &Arc<CudaContext>,
    id: CudnnTensorUid,
    data_type: DataType,
    region: DeviceRegion<'_>,
) -> Result<(CudnnTensorUid, DevicePtr)> {
    if region.context() != context {
        return Err(CudaError::MemoryContextMismatch.into());
    }

    let native = template
        .binding_tensor(id)
        .ok_or_else(|| CudnnError::DescriptorMismatch {
            name: format!("native graph tensor binding `{id}`").into(),
        })?;
    let actual_type = storage_type(data_type)?;
    if actual_type != native.data_type() {
        return Err(CudnnError::BackendTensorBindingTypeMismatch {
            tensor_id: id,
            expected: native.data_type(),
            actual: actual_type,
        }
        .into());
    }

    let required = native.storage_byte_len(None)?;
    if required > region.byte_len() {
        return Err(CudnnError::BackendTensorBindingCapacityMismatch {
            tensor_id: id,
            required,
            actual: region.byte_len(),
        }
        .into());
    }

    let alignment =
        usize::try_from(native.byte_alignment()).map_err(|_| CudnnError::OutOfRange {
            name: "native tensor binding alignment".into(),
        })?;
    if alignment > 1 && !region.as_ptr().addr().is_multiple_of(alignment) {
        return Err(CudnnError::BackendTensorBindingAlignmentMismatch {
            tensor_id: id,
            required: alignment,
        }
        .into());
    }
    // SAFETY: the validated pointer belongs to this region; address registration grants no access.
    Ok((id, unsafe { device_ptr(region.as_ptr()) }))
}
