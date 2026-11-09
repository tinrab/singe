use std::sync::Arc;

use crate::{
    backend::cuda::{
        cutensor::contraction::{
            cache::CachedCutensorContractionPlan, key::cutensor_contraction_plan_key,
            plan::create_cutensor_contraction_plan_handle,
            validation::validate_cutensor_contraction_inputs,
        },
        runtime::CudaRuntime,
        tensor_region::{CudaTensorRegion, CudaTensorSource},
    },
    error::Result,
    utility::Locked,
};

pub fn cached_cutensor_contraction_plan(
    runtime: &Arc<CudaRuntime>,
    a: &dyn CudaTensorSource,
    a_mode: &[i64],
    b: &dyn CudaTensorSource,
    b_mode: &[i64],
    c: &dyn CudaTensorSource,
    c_mode: &[i64],
    d_mode: &[i64],
    output: CudaTensorRegion<'_>,
) -> Result<CachedCutensorContractionPlan> {
    validate_cutensor_contraction_inputs(a, a_mode, b, b_mode, c, c_mode, d_mode, output)?;
    let key = cutensor_contraction_plan_key(a, a_mode, b, b_mode, c, c_mode, d_mode, output)?;
    {
        let cache = runtime
            .cutensor_contraction_plans()
            .locked("cuda runtime cache")?;
        if let Some(plan) = cache.get(&key) {
            return Ok(plan.clone());
        }
    }

    let plan = {
        let cutensor = runtime.cutensor().locked("cuda runtime cache")?;
        create_cutensor_contraction_plan_handle(
            &cutensor, a, a_mode, b, b_mode, c, c_mode, d_mode, output,
        )?
    };
    let workspace_size = plan.required_workspace_size_bytes()?;
    let cached = CachedCutensorContractionPlan::new(plan, workspace_size);

    let mut cache = runtime
        .cutensor_contraction_plans()
        .locked("cuda runtime cache")?;
    if let Some(existing) = cache.get(&key) {
        return Ok(existing.clone());
    }

    cache.insert(key, cached.clone());
    drop(cache);
    Ok(cached)
}
