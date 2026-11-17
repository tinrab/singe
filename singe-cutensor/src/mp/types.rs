use num_enum::{IntoPrimitive, TryFromPrimitive};

use singe_core::impl_enum_display;

use singe_cutensor_sys as sys;

singe_core::native_enum! {
    /// Algorithm selector for distributed multi-process tensor operations.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum Algorithm: sys::cutensorMpAlgo_t {
        /// Default algorithm.
        Default = CUTENSORMP_ALGO_DEFAULT,
    }
}

impl_enum_display!(Algorithm, {
    Algorithm::Default => "CUTENSORMP_ALGO_DEFAULT",
});

singe_core::native_enum! {
    /// Plan attributes reporting required device or host workspace bytes as `u64` values.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum PlanAttribute: sys::cutensorMpPlanAttribute_t {
        /// Exact device workspace in bytes the plan needs on this rank.
        RequiredWorkspaceDevice = CUTENSORMP_PLAN_REQUIRED_WORKSPACE_DEVICE,
        /// Exact host workspace in bytes the plan needs on this rank.
        RequiredWorkspaceHost = CUTENSORMP_PLAN_REQUIRED_WORKSPACE_HOST,
    }
}

impl_enum_display!(PlanAttribute, {
    PlanAttribute::RequiredWorkspaceDevice => "CUTENSORMP_PLAN_REQUIRED_WORKSPACE_DEVICE",
    PlanAttribute::RequiredWorkspaceHost => "CUTENSORMP_PLAN_REQUIRED_WORKSPACE_HOST",
});
