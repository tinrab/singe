mod cache;
mod key;
mod plan;
mod runtime;
mod validation;

use std::mem::MaybeUninit;

use singe_cuda::memory::DeviceMemory;
use singe_cutensor::{context::Context as CutensorContext, plan::Plan as CutensorPlan};

pub use self::{
    cache::CachedCutensorContractionPlan, key::CutensorContractionPlanKey,
    runtime::cached_cutensor_contraction_plan,
};
use crate::{
    backend::cuda::{
        cutensor::{
            contraction::{
                plan::create_cutensor_contraction_plan_handle,
                validation::validate_cutensor_contraction_inputs,
            },
            cutensor_workspace,
        },
        tensor_region::{CudaTensorRegion, CudaTensorSource},
    },
    error::Result,
};

pub fn create_cutensor_contraction_plan(
    cutensor: &CutensorContext,
    a: &dyn CudaTensorSource,
    a_mode: &[i64],
    b: &dyn CudaTensorSource,
    b_mode: &[i64],
    c: &dyn CudaTensorSource,
    c_mode: &[i64],
    d_mode: &[i64],
    output: CudaTensorRegion<'_>,
) -> Result<(CutensorPlan, Option<DeviceMemory<MaybeUninit<u8>>>)> {
    validate_cutensor_contraction_inputs(a, a_mode, b, b_mode, c, c_mode, d_mode, output)?;
    let plan = create_cutensor_contraction_plan_handle(
        cutensor, a, a_mode, b, b_mode, c, c_mode, d_mode, output,
    )?;
    let workspace_size = plan.required_workspace_size_bytes()?;
    Ok((
        plan,
        cutensor_workspace(cutensor.cuda_context(), workspace_size)?,
    ))
}
