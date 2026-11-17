//! Shared macros and FFI helpers for the Singe wrapper crates.
//! The wrapper crates re-export the types a user sees, such as [`CudaVersion`], so a program does not depend on this crate directly.
//!
//! This crate provides:
//!
//! - **Status and error macros:** [`status_enum!`] and [`wrapper_status!`] declare a library's status enum and the conversions from a raw status into the crate's `Error`, including the load error for a missing library.
//! - **Enum and integer macros:** [`native_enum!`] declares a typed enum over a bindgen enum with checked conversions both ways, and [`checked_conversions!`] and [`checked_arithmetic!`] declare integer conversions and size arithmetic that report overflow as the crate's out-of-range error.
//! - **Native loggers:** [`logger::CallbackSlot`], which holds the closure a crate's process-wide logger calls.
//! - **Versions:** [`CudaVersion`] and [`LibraryVersion`], with [`Display`](std::fmt::Display) and ordering.
//! - **Errors and strings:** [`ErrorChain`], which prints an error with each of its sources, and helpers that copy between Rust strings and C character buffers.
//!
//! # Features
//!
//! - `serde`: `Serialize` and `Deserialize` for [`CudaVersion`] and [`LibraryVersion`].

mod assert;
mod cleanup;
mod conversion;
mod enums;
pub mod logger;
mod report;
mod status;
mod string;
mod version;

pub use crate::{
    cleanup::{abort_after_cleanup_failure, report_cleanup_failure},
    enums::{UnknownEnumValue, try_enum_from_raw},
    report::ErrorChain,
    string::{
        copy_string_to_c_chars, path_to_cstring, static_str_from_c_ptr, string_from_c_chars,
        string_from_c_ptr,
    },
    version::{CudaVersion, LibraryVersion, PackedVersion},
};

#[cfg(test)]
#[expect(
    unreachable_pub,
    reason = "the status macros declare public items, as they do in a wrapper's error module"
)]
mod tests {
    use std::{error::Error, ffi::c_char, fmt, io, ptr};

    use super::*;

    // A chain prints each cause once, after the error it caused.
    #[test]
    fn test_an_error_chain_prints_every_cause_once() {
        #[derive(Debug)]
        struct Outer(io::Error);

        impl fmt::Display for Outer {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("reading the plan failed")
            }
        }

        impl Error for Outer {
            fn source(&self) -> Option<&(dyn Error + 'static)> {
                Some(&self.0)
            }
        }

        let error = Outer(io::Error::other("disk gone"));
        assert_eq!(error.to_string(), "reading the plan failed");
        assert_eq!(
            ErrorChain(&error).to_string(),
            "reading the plan failed: disk gone"
        );
    }

    #[test]
    fn test_library_version_decodes_packed_schemes() {
        let decode = |value, scheme| LibraryVersion::from_packed(value, scheme).unwrap();
        assert_eq!(
            decode(130_402_u32, PackedVersion::Major10000Minor100),
            LibraryVersion::new(13, 4, 2)
        );
        assert_eq!(
            decode(12_304_u32, PackedVersion::Major1000Minor100),
            LibraryVersion::new(12, 3, 4)
        );
        assert_eq!(
            decode(1190_u32, PackedVersion::Major1000Minor10),
            LibraryVersion::new(1, 19, 0)
        );
        assert_eq!(
            LibraryVersion::from_packed(-1_i32, PackedVersion::Major1000Minor10),
            None
        );
        assert_eq!(LibraryVersion::new(13, 4, 2).to_string(), "13.4.2");
    }

    #[test]
    fn test_cuda_version_rejects_negative_values() {
        let version = CudaVersion::from_raw(13_040_i32).unwrap();
        assert_eq!((version.major(), version.minor()), (13, 4));
        assert_eq!(version.to_string(), "13.4");
        assert_eq!(CudaVersion::from_raw(-1_i32), None);
    }

    #[test]
    fn test_c_chars_round_trip() {
        let mut buffer: [c_char; 8] = [0; 8];
        copy_string_to_c_chars(&mut buffer, "singe-core");
        assert_eq!(string_from_c_chars(&buffer), "singe-c");
        let mut buffer: [c_char; 8] = [1; 8];
        copy_string_to_c_chars(&mut buffer, "ok");
        assert_eq!(string_from_c_chars(&buffer), "ok");
    }

    #[test]
    fn test_null_c_pointer_is_empty() {
        // SAFETY: a null pointer is explicitly accepted.
        assert_eq!(unsafe { string_from_c_ptr(ptr::null()) }, "");
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct RawStatus(u32);

    impl RawStatus {
        const SUCCESS: Self = Self(0);
        const INVALID: Self = Self(3);
    }

    #[derive(Debug, Clone, Copy)]
    struct Absence {
        reason: &'static str,
    }

    impl Absence {
        const fn into_error(self) -> &'static str {
            self.reason
        }
    }

    const fn take_absence() -> Option<Absence> {
        None
    }

    #[derive(Debug)]
    enum TestError {
        Load(&'static str),
        Test { code: Status },
    }

    wrapper_status! {
        library = "test",
        error = TestError::Test,
        raw = RawStatus,
        success = SUCCESS,
        absence = take_absence,
        unknown = "UNKNOWN_TEST_STATUS",
        pub enum Status: u32 {
            Success = SUCCESS => "success",
            Invalid = INVALID => "invalid value",
        }
    }

    // One table for what `status_enum!` and `wrapper_status!` generate, which every wrapper's status relies on.
    #[test]
    fn test_wrapper_status_converts_and_describes() {
        let cases = [
            (3, Status::Invalid, "INVALID", "invalid value"),
            (0, Status::Success, "SUCCESS", "success"),
            (
                9999,
                Status::Unknown(9999),
                "UNKNOWN_TEST_STATUS(9999)",
                "unknown test error",
            ),
        ];
        for (code, status, display, description) in cases {
            assert_eq!(Status::from(RawStatus(code)), status, "code {code}");
            assert_eq!(status.raw(), code);
            assert_eq!(status.to_string(), display);
            assert_eq!(status.description(), description);
        }

        assert_eq!(RawStatus::try_from(Status::Invalid), Ok(RawStatus::INVALID));
        assert_eq!(
            RawStatus::try_from(Status::Unknown(7)),
            Err(Status::Unknown(7))
        );
        assert!(TestError::check(RawStatus::SUCCESS).is_ok());
        assert!(matches!(
            TestError::check(RawStatus::INVALID),
            Err(TestError::Test {
                code: Status::Invalid
            })
        ));
        match TestError::Load(Absence { reason: "missing" }.into_error()) {
            TestError::Load(reason) => assert_eq!(reason, "missing"),
            error @ TestError::Test { .. } => panic!("unexpected error: {error:?}"),
        }
    }

    #[test]
    fn test_enum_conversion_error_names_value() {
        let error = try_enum_from_raw::<u8, u32>("flag", 300).unwrap_err();
        assert_eq!(error.to_string(), "unknown `flag` value 300");
    }
}
