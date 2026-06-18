use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::impl_enum_conversion;
use singe_cutensor_sys as sys;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(i32)]
#[non_exhaustive]
pub enum Algorithm {
    Default = sys::cutensorMgAlgo_t::CUTENSORMG_ALGO_DEFAULT as _,
}

impl_enum_conversion!(i32, sys::cutensorMgAlgo_t, Algorithm);
