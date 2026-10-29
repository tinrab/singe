use num_enum::{IntoPrimitive, TryFromPrimitive};

use singe_core::{impl_enum_conversion, impl_enum_display};

use singe_cutensor_sys as sys;

/// Represents the selected algorithm when planning for a contraction operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(i32)]
#[non_exhaustive]
pub enum Algorithm {
    /// Lets the internal heuristic choose.
    Default = sys::cutensorMgAlgo_t::CUTENSORMG_ALGO_DEFAULT as _,
}

impl_enum_conversion!(i32, sys::cutensorMgAlgo_t, Algorithm);

impl_enum_display!(Algorithm, {
    Algorithm::Default => "CUTENSORMG_ALGO_DEFAULT",
});
