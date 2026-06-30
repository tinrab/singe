//! Shared macros and small FFI utility helpers used by Singe wrapper crates.
//!
//! This crate shouldn't be used directly.

use std::{
    ffi::{CStr, CString, NulError},
    fmt::{self, Display, Formatter},
    path::Path,
};

#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EnumConversionError<T> {
    pub name: &'static str,
    pub value: T,
}

impl<T: Display> Display for EnumConversionError<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "unknown {} value {}", self.name, self.value)
    }
}

impl<T: Display + fmt::Debug> std::error::Error for EnumConversionError<T> {}

pub fn try_enum_from_raw<E, T>(name: &'static str, value: T) -> Result<E, EnumConversionError<T>>
where
    E: TryFrom<T>,
    T: Copy,
{
    E::try_from(value).map_err(|_| EnumConversionError { name, value })
}

/// Converts an integer with caller-provided error construction.
///
/// Wrapper crates use this to share the checked conversion policy while keeping
/// their crate-specific error types and messages.
pub fn checked_int<T, U, E>(value: T, name: &str, error: impl FnOnce(String) -> E) -> Result<U, E>
where
    T: TryInto<U>,
{
    value.try_into().map_err(|_| error(name.into()))
}

/// Implements bidirectional conversions between a raw FFI enum and a typed Rust enum.
///
/// The typed enum must implement `TryFrom<$ty>` and `Into<$ty>`, where `$ty`
/// matches the raw enum representation. Converting from the raw enum panics if
/// the raw discriminant is not represented by the typed enum. Converting back to
/// the raw enum transmutes the typed discriminant, so the raw enum must use the
/// same integer representation.
#[macro_export]
macro_rules! impl_enum_conversion {
    ($ty:ty, $from:ty, $into:ty $(,)?) => {
        const _: () = {
            impl From<$from> for $into {
                fn from(value: $from) -> Self {
                    let value = value as $ty;
                    <$into>::try_from(value).unwrap()
                }
            }

            impl From<$into> for $from {
                fn from(value: $into) -> Self {
                    let value = value.into();
                    unsafe { core::mem::transmute::<$ty, $from>(value) }
                }
            }
        };
    };

    ($from:ty, $into:ty $(,)?) => {
        impl_enum_conversion!(u32, $from, $into);
    };
}

/// Implements [`Display`](std::fmt::Display) for an enum by mapping variants to
/// fixed string names.
#[macro_export]
macro_rules! impl_enum_display {
    ($enum:ty, { $($variant:path => $name:expr),+ $(,)? } $(,)?) => {
        impl std::fmt::Display for $enum {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    $($variant => f.write_str($name),)+
                }
            }
        }
    };
}

/// Asserts that two numeric values or numeric slices are equal within a tolerance.
///
/// Slice forms compare lengths first and then compare each element using an
/// absolute difference. Scalar forms accept a label that is included in the panic
/// message. The default tolerance is `1.0e-5`.
#[macro_export]
macro_rules! assert_close {
    ($actual:expr, $expected:expr $(,)?) => {
        $crate::assert_close!($actual, $expected, 1.0e-5);
    };

    ($actual:expr, $expected:expr, $tolerance:expr $(,)?) => {{
        let actual = $actual;
        let expected = $expected;
        let tolerance = $tolerance as f64;

        assert_eq!(
            actual.len(),
            expected.len(),
            "assert_close length mismatch: actual len {}, expected len {}",
            actual.len(),
            expected.len(),
        );

        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            let actual = *actual as f64;
            let expected = *expected as f64;
            let difference = (actual - expected).abs();

            assert!(
                difference <= tolerance,
                "assert_close failed at index {index}: actual {actual}, expected {expected}, difference {difference}, tolerance {tolerance}",
            );
        }
    }};

    ($actual:expr, $expected:expr, $tolerance:expr, $name:expr $(,)?) => {{
        let actual = $actual as f64;
        let expected = $expected as f64;
        let tolerance = $tolerance as f64;
        let difference = (actual - expected).abs();

        assert!(
            difference <= tolerance,
            "assert_close failed for {}: actual {actual}, expected {expected}, difference {difference}, tolerance {tolerance}",
            $name,
        );
    }};
}

/// Asserts that two complex numeric values are equal within a tolerance.
///
/// The compared values must expose `re` and `im` numeric fields. This keeps the
/// helper independent of any particular complex-number crate while covering the
/// CUDA complex wrapper types used in examples and tests. The default tolerance
/// is `1.0e-5`.
#[macro_export]
macro_rules! assert_complex_close {
    ($actual:expr, $expected:expr $(,)?) => {
        $crate::assert_complex_close!($actual, $expected, 1.0e-5);
    };

    ($actual:expr, $expected:expr, $tolerance:expr $(,)?) => {{
        let actual = $actual;
        let expected = $expected;
        let tolerance = $tolerance as f64;
        let real_difference = (actual.re as f64 - expected.re as f64).abs();
        let imaginary_difference = (actual.im as f64 - expected.im as f64).abs();

        assert!(
            real_difference <= tolerance && imaginary_difference <= tolerance,
            "assert_complex_close failed: actual ({}, {}i), expected ({}, {}i), real difference {}, imaginary difference {}, tolerance {}",
            actual.re,
            actual.im,
            expected.re,
            expected.im,
            real_difference,
            imaginary_difference,
            tolerance,
        );
    }};
}

/// Converts a filesystem path to a C string.
///
/// On Unix, paths are converted from their raw OS bytes. On other platforms,
/// paths are converted as UTF-8. Interior NUL bytes are reported as
/// [`NulError`].
///
/// # Errors
///
/// Returns an error if the converted path contains an interior NUL byte.
pub fn path_to_cstring(path: &Path) -> Result<CString, NulError> {
    #[cfg(unix)]
    {
        CString::new(path.as_os_str().as_bytes())
    }

    #[cfg(not(unix))]
    {
        CString::new(path.as_os_str().to_string_lossy().as_bytes())
    }
}

/// Converts a fixed C character buffer to a Rust string.
///
/// Bytes after the first NUL are ignored and invalid UTF-8 is replaced with the
/// Unicode replacement character. If the buffer has no NUL terminator, the whole
/// bounded buffer is converted.
pub fn string_from_c_chars(buffer: &[i8]) -> String {
    let bytes = unsafe { std::slice::from_raw_parts(buffer.as_ptr().cast::<u8>(), buffer.len()) };
    let end = bytes
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

/// Converts a C string pointer to a Rust string.
///
/// A null pointer is treated as an empty string. Otherwise `ptr` must point to a
/// valid NUL-terminated C string for the duration of the call.
///
/// # Safety
///
/// `ptr` must be either null or a valid pointer accepted by [`CStr::from_ptr`].
pub unsafe fn string_from_c_ptr(ptr: *const i8) -> String {
    if ptr.is_null() {
        return String::new();
    }

    unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
}

/// Copies a Rust string into a fixed C character buffer.
///
/// At most `N - 1` bytes are copied and the remaining buffer contents are left
/// unchanged. To guarantee a trailing NUL, zero-initialize the destination before
/// calling this function.
pub fn copy_string_to_c_chars<const N: usize>(buffer: &mut [i8; N], value: &str) {
    let bytes = value.as_bytes();
    let len = bytes.len().min(N.saturating_sub(1));
    for (slot, byte) in buffer.iter_mut().zip(bytes.iter().copied()).take(len) {
        *slot = byte as i8;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct LibraryVersion {
    major: u64,
    minor: u64,
    patch: u64,
}

impl LibraryVersion {
    pub const fn new(major: u64, minor: u64, patch: u64) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    pub const fn major(self) -> u64 {
        self.major
    }

    pub const fn minor(self) -> u64 {
        self.minor
    }

    pub const fn patch(self) -> u64 {
        self.patch
    }

    pub const fn raw(self) -> u64 {
        self.major * 10000 + self.minor * 100 + self.patch
    }
}

impl PartialEq<u64> for LibraryVersion {
    fn eq(&self, other: &u64) -> bool {
        self.raw() == *other
    }
}

impl PartialOrd<u64> for LibraryVersion {
    fn partial_cmp(&self, other: &u64) -> Option<std::cmp::Ordering> {
        self.raw().partial_cmp(other)
    }
}

impl PartialEq<u32> for LibraryVersion {
    fn eq(&self, other: &u32) -> bool {
        self.raw() == u64::from(*other)
    }
}

impl PartialOrd<u32> for LibraryVersion {
    fn partial_cmp(&self, other: &u32) -> Option<std::cmp::Ordering> {
        self.raw().partial_cmp(&u64::from(*other))
    }
}

impl PartialEq<i32> for LibraryVersion {
    fn eq(&self, other: &i32) -> bool {
        *other >= 0 && self.raw() == *other as u64
    }
}

impl PartialOrd<i32> for LibraryVersion {
    fn partial_cmp(&self, other: &i32) -> Option<std::cmp::Ordering> {
        if *other < 0 {
            Some(std::cmp::Ordering::Greater)
        } else {
            self.raw().partial_cmp(&(*other as u64))
        }
    }
}

impl Display for LibraryVersion {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.raw(), f)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CudaRuntimeVersion {
    pub raw: u64,
}

impl CudaRuntimeVersion {
    pub const fn from_raw(raw: u64) -> Self {
        Self { raw }
    }

    pub const fn raw(self) -> u64 {
        self.raw
    }
}

impl From<u64> for CudaRuntimeVersion {
    fn from(raw: u64) -> Self {
        Self::from_raw(raw)
    }
}

impl From<u32> for CudaRuntimeVersion {
    fn from(raw: u32) -> Self {
        Self::from_raw(raw as u64)
    }
}

impl From<usize> for CudaRuntimeVersion {
    fn from(raw: usize) -> Self {
        Self::from_raw(raw as u64)
    }
}

impl From<i32> for CudaRuntimeVersion {
    fn from(raw: i32) -> Self {
        Self::from_raw(raw as u64)
    }
}

impl PartialEq<u64> for CudaRuntimeVersion {
    fn eq(&self, other: &u64) -> bool {
        self.raw == *other
    }
}

impl PartialOrd<u64> for CudaRuntimeVersion {
    fn partial_cmp(&self, other: &u64) -> Option<std::cmp::Ordering> {
        self.raw.partial_cmp(other)
    }
}

impl PartialEq<u32> for CudaRuntimeVersion {
    fn eq(&self, other: &u32) -> bool {
        self.raw == u64::from(*other)
    }
}

impl PartialOrd<u32> for CudaRuntimeVersion {
    fn partial_cmp(&self, other: &u32) -> Option<std::cmp::Ordering> {
        self.raw.partial_cmp(&u64::from(*other))
    }
}

impl PartialEq<i32> for CudaRuntimeVersion {
    fn eq(&self, other: &i32) -> bool {
        *other >= 0 && self.raw == *other as u64
    }
}

impl PartialOrd<i32> for CudaRuntimeVersion {
    fn partial_cmp(&self, other: &i32) -> Option<std::cmp::Ordering> {
        if *other < 0 {
            Some(std::cmp::Ordering::Greater)
        } else {
            self.raw.partial_cmp(&(*other as u64))
        }
    }
}

impl Display for CudaRuntimeVersion {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.raw, f)
    }
}
