//! Raw FFI bindings for CUPTI.
//!
//! Prefer the safe `singe-cupti` crate unless direct NVIDIA ABI access is required.

#![allow(deprecated, warnings, unused_qualifications, clippy::all)]

use num_enum::{IntoPrimitive, TryFromPrimitive};

#[cfg(feature = "cupti_13_3")]
include!("cupti_sys_130300.rs");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(CUPTI_API_VERSION, 130300);
    }
}
