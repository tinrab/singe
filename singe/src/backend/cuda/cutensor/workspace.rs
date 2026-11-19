use std::{mem::MaybeUninit, sync::Arc};

use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory};

use crate::{backend::cuda::allocator::allocate_scratch, error::Result};

pub fn cutensor_workspace(
    context: &Arc<CudaContext>,
    workspace_size: usize,
) -> Result<Option<DeviceMemory<MaybeUninit<u8>>>> {
    if workspace_size == 0 {
        Ok(None)
    } else {
        Ok(Some(allocate_scratch(context, workspace_size)?))
    }
}
