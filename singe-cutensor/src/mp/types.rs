use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cutensor_sys as sys;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Algorithm {
    Default = sys::cutensorMpAlgo_t::CUTENSORMP_ALGO_DEFAULT as _,
}

impl_enum_conversion!(sys::cutensorMpAlgo_t, Algorithm);

impl_enum_display!(Algorithm, {
    Algorithm::Default => "CUTENSORMP_ALGO_DEFAULT",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum PlanAttribute {
    RequiredWorkspaceDevice =
        sys::cutensorMpPlanAttribute_t::CUTENSORMP_PLAN_REQUIRED_WORKSPACE_DEVICE as _,
    RequiredWorkspaceHost =
        sys::cutensorMpPlanAttribute_t::CUTENSORMP_PLAN_REQUIRED_WORKSPACE_HOST as _,
}

impl_enum_conversion!(sys::cutensorMpPlanAttribute_t, PlanAttribute);

impl_enum_display!(PlanAttribute, {
    PlanAttribute::RequiredWorkspaceDevice => "CUTENSORMP_PLAN_REQUIRED_WORKSPACE_DEVICE",
    PlanAttribute::RequiredWorkspaceHost => "CUTENSORMP_PLAN_REQUIRED_WORKSPACE_HOST",
});
