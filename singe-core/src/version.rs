use std::fmt::{self, Display, Formatter};

/// Version of an NVIDIA library as major, minor, and patch components, ordered by component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LibraryVersion {
    major: u32,
    minor: u32,
    patch: u32,
}

/// Scheme a library uses to pack its version into one integer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PackedVersion {
    /// `major * 10000 + minor * 100 + patch`, as cuBLAS, cuDNN 9, cuTENSOR, cuDSS, and NCCL report it.
    Major10000Minor100,
    /// `major * 1000 + minor * 100 + patch`, as cuRAND, cuSOLVER, cuSPARSE, cuFFT, and nvImageCodec report it.
    ///
    /// A minor version of 10 or more cannot be represented.
    Major1000Minor100,
    /// `major * 1000 + minor * 10`, as cuFile and the CUDA runtime report it; the patch is 0.
    Major1000Minor10,
}

impl LibraryVersion {
    /// Creates a version from its components.
    #[must_use]
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Creates a version from components a library reports separately, such as through its `GetProperty` call.
    ///
    /// Returns `None` if a component is negative or does not fit in `u32`.
    #[must_use]
    pub fn from_properties(
        major: impl TryInto<u32>,
        minor: impl TryInto<u32>,
        patch: impl TryInto<u32>,
    ) -> Option<Self> {
        Some(Self::new(
            major.try_into().ok()?,
            minor.try_into().ok()?,
            patch.try_into().ok()?,
        ))
    }

    /// Decodes a version a library packs into one integer with `scheme`.
    ///
    /// Returns `None` if `value` is negative or does not fit in `u32`.
    #[must_use]
    pub fn from_packed(value: impl TryInto<u32>, scheme: PackedVersion) -> Option<Self> {
        let value: u32 = value.try_into().ok()?;
        Some(match scheme {
            PackedVersion::Major10000Minor100 => {
                Self::new(value / 10_000, value % 10_000 / 100, value % 100)
            }
            PackedVersion::Major1000Minor100 => {
                Self::new(value / 1000, value % 1000 / 100, value % 100)
            }
            PackedVersion::Major1000Minor10 => Self::new(value / 1000, value % 1000 / 10, 0),
        })
    }

    /// Returns the major version component.
    #[must_use]
    pub const fn major(self) -> u32 {
        self.major
    }

    /// Returns the minor version component.
    #[must_use]
    pub const fn minor(self) -> u32 {
        self.minor
    }

    /// Returns the patch version component.
    #[must_use]
    pub const fn patch(self) -> u32 {
        self.patch
    }
}

impl Display for LibraryVersion {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// CUDA runtime or driver version in the native `major * 1000 + minor * 10` encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CudaVersion {
    raw: u32,
}

impl CudaVersion {
    /// Creates a version from the native integer, such as `13040` for CUDA 13.4.
    ///
    /// Returns `None` if `raw` is negative or does not fit in `u32`.
    #[must_use]
    pub fn from_raw(raw: impl TryInto<u32>) -> Option<Self> {
        Some(Self {
            raw: raw.try_into().ok()?,
        })
    }

    /// Returns the major version.
    #[must_use]
    pub const fn major(self) -> u32 {
        self.raw / 1000
    }

    /// Returns the minor version.
    #[must_use]
    pub const fn minor(self) -> u32 {
        self.raw % 1000 / 10
    }
}

impl Display for CudaVersion {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major(), self.minor())
    }
}
