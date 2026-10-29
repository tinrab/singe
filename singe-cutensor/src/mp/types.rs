use num_enum::{IntoPrimitive, TryFromPrimitive};

use singe_core::{impl_enum_conversion, impl_enum_display};

use singe_cutensor_sys as sys;

/// Algorithm selector for distributed multi-process tensor operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Algorithm {
    /// Default algorithm.
    Default = sys::cutensorMpAlgo_t::CUTENSORMP_ALGO_DEFAULT as _,
}

impl_enum_conversion!(sys::cutensorMpAlgo_t, Algorithm);

impl_enum_display!(Algorithm, {
    Algorithm::Default => "CUTENSORMP_ALGO_DEFAULT",
});

/// Plan attributes reporting required device or host workspace bytes as `uint64_t` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum PlanAttribute {
    /// `uint64_t`: exact required workspace in bytes that is needed to execute the plan.
    RequiredWorkspaceDevice =
        sys::cutensorMpPlanAttribute_t::CUTENSORMP_PLAN_REQUIRED_WORKSPACE_DEVICE as _,
    /// `uint64_t`: exact required workspace in bytes that is needed to execute the plan.
    RequiredWorkspaceHost =
        sys::cutensorMpPlanAttribute_t::CUTENSORMP_PLAN_REQUIRED_WORKSPACE_HOST as _,
}

impl_enum_conversion!(sys::cutensorMpPlanAttribute_t, PlanAttribute);

impl_enum_display!(PlanAttribute, {
    PlanAttribute::RequiredWorkspaceDevice => "CUTENSORMP_PLAN_REQUIRED_WORKSPACE_DEVICE",
    PlanAttribute::RequiredWorkspaceHost => "CUTENSORMP_PLAN_REQUIRED_WORKSPACE_HOST",
});
