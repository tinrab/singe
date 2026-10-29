//! Low-level BLAS operation mirrors.

/// Vector operations, reductions, and rotations.
pub mod level1;
/// Matrix-vector products, rank updates, and triangular solves.
pub mod level2;
/// Matrix-matrix products, rank updates, and triangular solves.
pub mod level3;
