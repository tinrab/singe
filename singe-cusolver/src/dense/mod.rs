//! Dense cuSOLVER operation wrappers.
//!
//! Typed operation families such as [`Potrf`], [`Getrf`], and [`Geqrf`] have a waiting `execute` and an unsafe `execute_async` that follows the crate's [raw execution contract](crate#raw-execution).
//! Workspace queries use [`MatrixLayout`](crate::layout::MatrixLayout) and require no device allocations.
//! The generic native 64-bit entry points stay public as `x*_async` functions; the per-precision entry points are reached through [`SolverElement`](crate::data_type::SolverElement).
//!
//! Successful completion does not establish a valid result: inspect the produced status before using numerical outputs.
//! Scalar status writes produce one element; batched POTRF produces one per matrix, while batched POTRS produces a single status.
//! LU pivots produce `min(rows, columns)` elements; symmetric factorization pivots produce `n`.
//! QR coefficients produce `min(rows, columns)` elements.
//! Tridiagonal reduction produces `n` diagonal values and `n - 1` off-diagonal values and coefficients.
//! Bidiagonal reduction requires `min(rows, columns)` capacity for each output, but only `min(rows, columns) - 1` off-diagonal values are numerical results.
//! Scratch never establishes initialized result values.
//! Leading-dimension padding and allocation tails are outside every logical output contract.
//! Single-matrix triangular factorization and inversion preserve their initialized destination outside the selected triangle.
//! Batched POTRF and POTRS use the opposite triangle as scratch.
//! LARFT also requires an initialized destination because its numerical output contract specifies only the selected triangle.

mod generic;

mod legacy;
mod operations;
pub(crate) mod scalar;
pub(crate) mod validation;

pub use self::{
    generic::{
        xgetrf_async, xgetrf_workspace_size, xgetrs_async, xhetrf_async, xhetrf_workspace_size,
        xhetrs_async, xhetrs_workspace_size, xlarft_async, xlarft_workspace_size, xpotrf_async,
        xpotrf_workspace_size, xpotrs_async, xsytrf_async, xsytrf_workspace_size, xsytrs_async,
        xsytrs_workspace_size, xtrtri_async, xtrtri_workspace_size,
    },
    legacy::qr::{xgeqrf_async, xgeqrf_workspace_size},
    operations::{
        BatchedPotrf, BatchedPotrfBindings, BatchedPotrs, BatchedPotrsBindings, Gebrd,
        GebrdBindings, Geqrf, GeqrfBindings, Getrf, GetrfBindings, Getrs, GetrsBindings, Larft,
        LarftBindings, Orgbr, OrgbrBindings, Orgqr, OrgqrBindings, Orgtr, OrgtrBindings, Ormqr,
        OrmqrBindings, Ormtr, OrmtrBindings, Potrf, PotrfBindings, Potri, PotriBindings, Potrs,
        PotrsBindings, Sytrd, SytrdBindings, Sytrf, SytrfBindings, Sytrs, SytrsBindings, Trtri,
        TrtriBindings,
    },
};
