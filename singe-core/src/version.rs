use std::{
    cmp,
    fmt::{self, Display, Formatter},
};

use serde::{Deserialize, Serialize};

/// NVIDIA library version encoded as `major * 10000 + minor * 100 + patch`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct LibraryVersion {
    major: u64,
    minor: u64,
    patch: u64,
}

impl LibraryVersion {
    /// Stores the major, minor, and patch components without validating their ranges.
    #[must_use]
    pub const fn new(major: u64, minor: u64, patch: u64) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Returns the major version component.
    #[must_use]
    pub const fn major(self) -> u64 {
        self.major
    }

    /// Returns the minor version component.
    #[must_use]
    pub const fn minor(self) -> u64 {
        self.minor
    }

    /// Returns the patch version component.
    #[must_use]
    pub const fn patch(self) -> u64 {
        self.patch
    }

    /// Encodes the components as `major * 10000 + minor * 100 + patch`.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if the encoded value overflows `u64`.
    #[must_use]
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
    fn partial_cmp(&self, other: &u64) -> Option<cmp::Ordering> {
        self.raw().partial_cmp(other)
    }
}

impl PartialEq<u32> for LibraryVersion {
    fn eq(&self, other: &u32) -> bool {
        self.raw() == u64::from(*other)
    }
}

impl PartialOrd<u32> for LibraryVersion {
    fn partial_cmp(&self, other: &u32) -> Option<cmp::Ordering> {
        self.raw().partial_cmp(&u64::from(*other))
    }
}

impl PartialEq<i32> for LibraryVersion {
    fn eq(&self, other: &i32) -> bool {
        *other >= 0 && self.raw() == *other as u64
    }
}

impl PartialOrd<i32> for LibraryVersion {
    fn partial_cmp(&self, other: &i32) -> Option<cmp::Ordering> {
        if *other < 0 {
            Some(cmp::Ordering::Greater)
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

/// CUDA toolkit version integer in the native `CUDA_VERSION` encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CudaRuntimeVersion {
    /// Encoded version, conventionally `major * 1000 + minor * 10`.
    pub raw: u64,
}

impl CudaRuntimeVersion {
    /// Stores a native CUDA version integer without validating its encoding.
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self { raw }
    }

    /// Returns the stored native CUDA version integer.
    #[must_use]
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
        Self::from_raw(u64::from(raw))
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
    fn partial_cmp(&self, other: &u64) -> Option<cmp::Ordering> {
        self.raw.partial_cmp(other)
    }
}

impl PartialEq<u32> for CudaRuntimeVersion {
    fn eq(&self, other: &u32) -> bool {
        self.raw == u64::from(*other)
    }
}

impl PartialOrd<u32> for CudaRuntimeVersion {
    fn partial_cmp(&self, other: &u32) -> Option<cmp::Ordering> {
        self.raw.partial_cmp(&u64::from(*other))
    }
}

impl PartialEq<i32> for CudaRuntimeVersion {
    fn eq(&self, other: &i32) -> bool {
        *other >= 0 && self.raw == *other as u64
    }
}

impl PartialOrd<i32> for CudaRuntimeVersion {
    fn partial_cmp(&self, other: &i32) -> Option<cmp::Ordering> {
        if *other < 0 {
            Some(cmp::Ordering::Greater)
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
