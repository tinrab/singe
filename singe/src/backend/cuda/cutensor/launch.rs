use std::mem::{MaybeUninit, size_of};

use singe_cuda::{memory::DeviceMemory, view::DeviceOutputView};

use singe_cutensor::plan::Plan as CutensorPlan;

use crate::{
    backend::cuda::{
        device_views::f32_storage_view,
        stream::CudaStream,
        tensor_region::{CudaTensorRegion, CudaTensorSource},
    },
    error::Result,
};

/// Enqueues a contraction over the supplied tensors using an existing cuTENSOR plan.
///
/// # Safety
///
/// The plan must describe these tensors' datatypes, shapes, strides, and operators.
/// All tensors and workspace must remain live and correctly ordered until the stream completes.
/// Output and workspace accesses must not conflict with other queued work.
pub unsafe fn launch_cutensor_contraction_f32(
    plan: &CutensorPlan,
    stream: &CudaStream,
    a: &dyn CudaTensorSource,
    b: &dyn CudaTensorSource,
    c: &dyn CudaTensorSource,
    output: CudaTensorRegion<'_>,
    alpha: f32,
    beta: f32,
    workspace: Option<&mut DeviceMemory<MaybeUninit<u8>>>,
) -> Result<()> {
    let a_memory = f32_storage_view(a)?;
    let b_memory = f32_storage_view(b)?;
    let c_memory = f32_storage_view(c)?;
    let region = output.region();
    // SAFETY: the caller retains exclusive native writes over this checked output span.
    let mut output_memory = unsafe {
        DeviceOutputView::<f32>::from_raw_parts(region, region.byte_len() / size_of::<f32>())?
    };
    // SAFETY: the caller establishes the plan, storage lifetime, and stream access contract.
    unsafe {
        plan.contract(
            &alpha,
            &a_memory,
            &b_memory,
            &beta,
            &c_memory,
            &mut output_memory,
            workspace,
            stream.raw(),
        )?;
    };
    Ok(())
}
