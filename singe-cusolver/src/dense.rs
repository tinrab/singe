//! Dense cuSOLVER operation wrappers.
//!
//! Typed operation families such as [`Potrf`], [`Getrf`], and [`Geqrf`] complete
//! borrowed execution before returning. Workspace queries use [`MatrixLayout`](crate::layout::MatrixLayout)
//! and require no device allocations. Direct native entry points remain public.
//!
//! # Raw solver contracts
//!
//! Raw execution is unsafe and queues work on the context's installed stream.
//! Retain the context, parameter objects, input and output allocations, and both
//! workspaces through completion on every path and throughout captured replay.
//! Order initialized inputs before use and exclude overlapping writes and reads.
//! Pointer-table callers must also retain every pointee and establish its extent,
//! actual CUDA context, initialization, and exclusive access where it is written.
//!
//! Successful enqueue does not establish initialized outputs or a valid result.
//! After completion, inspect the produced status before using numerical outputs.
//! Scalar status writes produce one element; batched POTRF produces one per
//! matrix, while batched POTRS produces a single status. LU pivots produce
//! `min(rows, columns)` elements; symmetric factorization pivots produce `n`.
//! QR coefficients produce `min(rows, columns)` elements. Tridiagonal reduction
//! produces `n` diagonal values and `n - 1` off-diagonal values and coefficients.
//! Bidiagonal reduction requires `min(rows, columns)` capacity for each output,
//! but only `min(rows, columns) - 1` off-diagonal values are numerical results.
//! Scratch never establishes initialized result values. Leading-dimension
//! padding and allocation tails are outside every logical output contract.
//! Single-matrix triangular factorization and inversion preserve their initialized
//! destination outside the selected triangle. Batched POTRF and POTRS use the opposite
//! triangle as scratch. LARFT also requires an initialized destination
//! because its numerical output contract specifies only the selected triangle.

mod generic;
/// Scalar-specific dense solver entry points and legacy interfaces.
pub mod legacy;
mod operations;
mod scalar;
pub(crate) mod validation;

pub use generic::{
    xgetrf, xgetrf_buffer_size, xgetrs, xhetrf, xhetrf_buffer_size, xhetrs, xhetrs_buffer_size,
    xlarft, xlarft_buffer_size, xpotrf, xpotrf_buffer_size, xpotrs, xsytrf, xsytrf_buffer_size,
    xsytrs, xsytrs_buffer_size, xtrtri, xtrtri_buffer_size,
};
pub use operations::{
    BatchedPotrf, BatchedPotrfArgs, BatchedPotrs, BatchedPotrsArgs, Gebrd, GebrdArgs, Geqrf,
    GeqrfArgs, Getrf, GetrfArgs, Getrs, GetrsArgs, Larft, LarftArgs, Orgbr, OrgbrArgs, Orgqr,
    OrgqrArgs, Orgtr, OrgtrArgs, Ormqr, OrmqrArgs, Ormtr, OrmtrArgs, Potrf, PotrfArgs, Potri,
    PotriArgs, Potrs, PotrsArgs, Sytrd, SytrdArgs, Sytrf, SytrfArgs, Sytrs, SytrsArgs, Trtri,
    TrtriArgs,
};
pub use scalar::{
    GebrdScalar, OrgbrScalar, OrgqrScalar, OrgtrScalar, OrmqrScalar, OrmtrScalar,
    PotrfBatchedScalar, PotriScalar, PotrsBatchedScalar, SytrdScalar, SytrfScalar,
};
