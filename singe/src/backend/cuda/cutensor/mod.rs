mod contraction;
mod descriptor;
mod launch;
mod validation;
mod workspace;

pub use self::{
    contraction::{
        CachedCutensorContractionPlan, CutensorContractionPlanKey,
        cached_cutensor_contraction_plan, create_cutensor_contraction_plan,
    },
    descriptor::{cutensor_modes, cutensor_tensor_descriptor},
    launch::launch_cutensor_contraction_f32,
    validation::validate_cutensor_tensor_compatibility,
    workspace::cutensor_workspace,
};

#[cfg(all(test, feature = "testing"))]
mod tests;
