//! Binding documentation injection and cuTile kernel generation for workspace commands.

/// Generates cuTile kernels for the supported scalar dtypes.
pub mod cutile_dtype_kernels;
/// Generates fused cuTile kernels from the configured expressions.
pub mod cutile_fused;
/// Adds scraped NVIDIA API documentation to generated FFI declarations.
pub mod document;
/// Locates the workspace and writes generated files only when their contents change.
pub mod utility;
