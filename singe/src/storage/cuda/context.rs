use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

use singe_cuda::{
    context::Context as CudaContext,
    device::Device as CudaDevice,
    error::{Error as CudaError, Status as CudaStatus},
};

use crate::{
    device::Device,
    error::{Error, Result},
    utility::Locked,
};

pub fn cuda_device_id(device: Device) -> Result<i32> {
    let Some(device_id) = device.cuda_id() else {
        return Err(Error::device_mismatch(
            "cuda device id",
            Device::Cuda(0),
            device,
        ));
    };
    validate_cuda_device_id(device_id)?;
    Ok(device_id)
}

fn validate_cuda_device_id(device_id: i32) -> Result<()> {
    if device_id < 0 {
        return Err(Error::negative_cuda_device_id(device_id));
    }

    let count = CudaDevice::count().map_err(map_cuda_device_error)?;
    if count == 0 {
        return Err(Error::no_cuda_devices());
    }

    if device_id >= count {
        return Err(Error::cuda_device_out_of_range(device_id, count));
    }
    Ok(())
}

fn map_cuda_device_error(error: CudaError) -> Error {
    match error {
        CudaError::Cuda { code, .. }
            if code == CudaStatus::NoDevice || code == CudaStatus::InvalidDevice =>
        {
            Error::no_cuda_devices()
        }
        error => Error::from(error),
    }
}

pub fn cuda_context_for_device(device_id: i32) -> Result<Arc<CudaContext>> {
    static CONTEXTS: OnceLock<Mutex<HashMap<i32, Arc<CudaContext>>>> = OnceLock::new();

    validate_cuda_device_id(device_id)?;

    let contexts = CONTEXTS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut contexts = contexts.locked("cuda runtime cache")?;

    if let Some(context) = contexts.get(&device_id) {
        return Ok(Arc::clone(context));
    }

    let device = CudaDevice::new(device_id);
    let context_ref =
        CudaContext::retain_primary_for_device(device).map_err(map_cuda_device_error)?;
    contexts.insert(device_id, Arc::clone(&context_ref));
    drop(contexts);

    Ok(context_ref)
}
