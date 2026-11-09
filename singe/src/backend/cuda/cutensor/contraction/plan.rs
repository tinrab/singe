use singe_cutensor::{
    context::Context as CutensorContext,
    operation::{
        ComputeDescriptor as CutensorComputeDescriptor,
        OperationDescriptor as CutensorOperationDescriptor, TensorOperand as CutensorTensorOperand,
    },
    plan::{Plan as CutensorPlan, PlanPreference as CutensorPlanPreference},
    types::WorkspacePreference as CutensorWorkspacePreference,
};

use crate::{
    backend::cuda::{
        cutensor::{cutensor_modes, cutensor_tensor_descriptor},
        tensor_region::{CudaTensorRegion, CudaTensorSource},
    },
    error::{Error, Result},
    graph::{Determinism, NumericalPolicy},
};

pub fn create_cutensor_contraction_plan_handle(
    cutensor: &CutensorContext,
    a: &dyn CudaTensorSource,
    a_mode: &[i64],
    b: &dyn CudaTensorSource,
    b_mode: &[i64],
    c: &dyn CudaTensorSource,
    c_mode: &[i64],
    d_mode: &[i64],
    output: CudaTensorRegion<'_>,
) -> Result<CutensorPlan> {
    if NumericalPolicy::current().determinism == Determinism::Require {
        return Err(Error::unsupported_operation(
            "cuTENSOR contraction determinism is not established",
        ));
    }

    let a_descriptor = cutensor_tensor_descriptor(cutensor, a.tensor_region()?)?;
    let b_descriptor = cutensor_tensor_descriptor(cutensor, b.tensor_region()?)?;
    let c_descriptor = cutensor_tensor_descriptor(cutensor, c.tensor_region()?)?;
    let output_descriptor = cutensor_tensor_descriptor(cutensor, output)?;
    let a_modes = cutensor_modes(a_mode)?;
    let b_modes = cutensor_modes(b_mode)?;
    let c_modes = cutensor_modes(c_mode)?;
    let d_modes = cutensor_modes(d_mode)?;

    let operation = CutensorOperationDescriptor::contraction(
        cutensor,
        CutensorTensorOperand::identity(&a_descriptor, &a_modes),
        CutensorTensorOperand::identity(&b_descriptor, &b_modes),
        CutensorTensorOperand::identity(&c_descriptor, &c_modes),
        CutensorTensorOperand::identity(&output_descriptor, &d_modes),
        CutensorComputeDescriptor::f32(),
    )?;
    let preference = CutensorPlanPreference::create_default(cutensor)?;
    let workspace_estimate = CutensorPlan::estimate_workspace_size(
        cutensor,
        &operation,
        &preference,
        CutensorWorkspacePreference::Default,
    )?;
    Ok(CutensorPlan::create(
        cutensor,
        &operation,
        &preference,
        workspace_estimate,
    )?)
}
