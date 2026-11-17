use num_enum::{IntoPrimitive, TryFromPrimitive};

use singe_core::impl_enum_display;

use singe_cutensor_sys as sys;

singe_core::native_enum! {
    /// Represents the selected algorithm when planning for a contraction operation.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(i32)]
    #[non_exhaustive]
    pub enum Algorithm: sys::cutensorMgAlgo_t {
        /// Lets the internal heuristic choose.
        Default = CUTENSORMG_ALGO_DEFAULT,
    }
}

impl_enum_display!(Algorithm, {
    Algorithm::Default => "CUTENSORMG_ALGO_DEFAULT",
});
