//! Shared macros and small FFI utility helpers used by Singe wrapper crates.
//!
//! This crate shouldn't be used directly.

mod assert;
mod conversion;
mod enums;
mod status;
mod string;
mod version;

pub use crate::{
    enums::{EnumConversionError, try_enum_from_raw},
    string::{
        copy_string_to_c_chars, path_to_cstring, static_str_from_c_ptr, string_from_c_chars,
        string_from_c_ptr,
    },
    version::{CudaRuntimeVersion, LibraryVersion},
};

#[cfg(test)]
mod tests {
    use std::ptr;

    use super::*;

    #[test]
    fn test_library_version_encodes_components() {
        let version = LibraryVersion::new(13, 4, 2);
        assert_eq!(version.raw(), 130_402);
        assert_eq!(version, 130_402_u64);
        assert!(version > 130_401_u32);
        assert!(version > -1_i32);
        assert_eq!(version.to_string(), "130402");
    }

    #[test]
    fn test_cuda_runtime_version_compares_with_raw_integers() {
        let version = CudaRuntimeVersion::from(13_040_u32);
        assert_eq!(version, 13_040_u64);
        assert!(version < 13_050_i32);
        assert_eq!(version.to_string(), "13040");
    }

    #[test]
    fn test_c_chars_round_trip() {
        let mut buffer = [0_i8; 8];
        copy_string_to_c_chars(&mut buffer, "singe-core");
        assert_eq!(string_from_c_chars(&buffer), "singe-c");
        assert_eq!(string_from_c_chars(&[b'o' as i8, b'k' as i8]), "ok");
    }

    #[test]
    fn test_null_c_pointer_is_empty() {
        // SAFETY: a null pointer is explicitly accepted.
        assert_eq!(unsafe { string_from_c_ptr(ptr::null()) }, "");
    }

    #[test]
    fn test_enum_conversion_error_names_value() {
        let error = try_enum_from_raw::<u8, u32>("flag", 300).unwrap_err();
        assert_eq!(error.to_string(), "unknown flag value 300");
    }
}
