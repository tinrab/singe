use std::{mem::MaybeUninit, sync::Arc};

use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory};

use crate::error::Result;

pub fn cutensor_workspace(
    context: &Arc<CudaContext>,
    workspace_size: usize,
) -> Result<Option<DeviceMemory<MaybeUninit<u8>>>> {
    if workspace_size == 0 {
        Ok(None)
    } else {
        Ok(Some(DeviceMemory::<u8>::create_uninit(
            context,
            workspace_size,
        )?))
    }
}
