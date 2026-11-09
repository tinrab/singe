use std::{mem::MaybeUninit, sync::Arc};

use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory};

use crate::{
    backend::cuda::{tensor_region::CudaTensorSource, utility::ensure_same_device},
    error::{Error, Result},
};

pub fn validate_cublaslt_matmul_tensors(
    a: &(impl CudaTensorSource + ?Sized),
    b: &(impl CudaTensorSource + ?Sized),
    output: &(impl CudaTensorSource + ?Sized),
) -> Result<()> {
    ensure_same_device(a, b)?;
    ensure_same_device(a, output)?;
    if a.data_type() != output.data_type() {
        return Err(Error::data_type_mismatch(
            "cuBLASLt matmul left-hand input",
            output.data_type(),
            a.data_type(),
        ));
    }

    if b.data_type() != output.data_type() {
        return Err(Error::data_type_mismatch(
            "cuBLASLt matmul right-hand input",
            output.data_type(),
            b.data_type(),
        ));
    }

    a.tensor_region()?;
    b.tensor_region()?;
    output.tensor_region()?;
    Ok(())
}

pub fn cublaslt_workspace(
    context: &Arc<CudaContext>,
    workspace_bytes: usize,
) -> Result<Option<DeviceMemory<MaybeUninit<u8>>>> {
    if workspace_bytes == 0 {
        Ok(None)
    } else {
        Ok(Some(DeviceMemory::<u8>::create_uninit(
            context,
            workspace_bytes,
        )?))
    }
}
