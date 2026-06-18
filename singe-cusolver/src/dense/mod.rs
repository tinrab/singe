//! Dense cuSOLVER operation wrappers.
//!
//! Public APIs should prefer typed operation families such as [`Potrf`],
//! [`Getrf`], and [`Geqrf`]. Upstream `S`/`D`/`C`/`Z` entry-point wrappers are
//! implementation details unless an operation has not yet been lifted into a
//! typed family.

pub(crate) mod validation;

// TODO: remove?
pub mod legacy;

mod generic;
pub use generic::*;

mod scalar;
pub use scalar::*;

mod operations;
pub use operations::*;
