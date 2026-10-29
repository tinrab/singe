//! Architectural limits and PTX language versions of the emulated `sm_80` target.

use std::fmt::{self, Display, Formatter};

use crate::{
    ast::{Module, Span},
    emulator::error::{CompileError, CompileErrorKind, Result},
};

/// Lanes per warp.
pub const WARP_SIZE: usize = 32;
/// Upper bound reported by `%nwarpid`.
pub const WARP_IDENTIFIER_COUNT: u32 = 64;
/// Maximum CTA extent along x, y, and z.
pub const MAX_BLOCK_DIMENSIONS: [u32; 3] = [1024, 1024, 64];
/// Maximum threads in one CTA.
pub const MAX_BLOCK_THREADS: u32 = 1024;
/// Maximum grid extent along x, y, and z.
pub const MAX_GRID_DIMENSIONS: [u32; 3] = [i32::MAX.cast_unsigned(), 65_535, 65_535];
/// Warps in the largest CTA.
pub const WARPS_PER_CTA: usize = MAX_BLOCK_THREADS as usize / WARP_SIZE;
/// Named CTA barriers available to `bar` and `barrier`.
pub const NAMED_BARRIERS: usize = 16;
/// Register slots one function may declare; a frontend limit, not an architectural one.
pub const MAX_FUNCTION_REGISTERS: usize = 65_536;

/// PTX language version, ordered by major and then minor version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PtxVersion {
    major: u32,
    minor: u32,
}

impl PtxVersion {
    pub(crate) fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }

    pub(crate) fn of(module: &Module) -> Self {
        Self::new(module.version.major, module.version.minor)
    }

    /// Returns whether this is a released PTX version from 7.0 through 9.4.
    pub(crate) fn is_supported(self) -> bool {
        matches!((self.major, self.minor), (7 | 8, 0..=8) | (9, 0..=4))
    }
}

impl Display for PtxVersion {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

pub fn validate_module(module: &Module) -> Result<()> {
    let version = PtxVersion::of(module);
    if !version.is_supported() {
        return Err(CompileError::unsupported(
            module.version.span,
            format!("PTX version {version}"),
        ));
    }
    if module.target.specifiers != ["sm_80"] {
        return Err(CompileError::unsupported(
            module.target.span,
            format!("target {}", module.target.specifiers.join(", ")),
        ));
    }
    Ok(())
}

/// Returns whether CTA dimensions are nonzero and fit the architectural limits.
pub fn block_fits(dimensions: [u32; 3]) -> bool {
    dimensions
        .into_iter()
        .zip(MAX_BLOCK_DIMENSIONS)
        .all(|(value, maximum)| value > 0 && value <= maximum)
        && dimensions
            .into_iter()
            .try_fold(1u32, u32::checked_mul)
            .is_some_and(|count| count <= MAX_BLOCK_THREADS)
}

/// Returns whether grid dimensions are nonzero and fit the architectural limits.
pub fn grid_fits(dimensions: [u32; 3]) -> bool {
    dimensions
        .into_iter()
        .zip(MAX_GRID_DIMENSIONS)
        .all(|(value, maximum)| value > 0 && value <= maximum)
}

pub fn validate_block(dimensions: [u32; 3], span: Span) -> Result<()> {
    if block_fits(dimensions) {
        Ok(())
    } else {
        Err(CompileError::new(
            span,
            CompileErrorKind::InvalidLaunch("block dimensions exceed sm_80 limits".into()),
        ))
    }
}
