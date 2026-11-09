//! Emitters: from a kernel and a schedule to code a target compiles.
//!
//! Every emitter declares the parameters [`abi::parameters`](crate::abi::parameters) lists and divides work as the schedule says, so launching code is the same for all of them.

pub mod cuda_c;
mod source;
#[cfg(feature = "tile")]
pub mod tile;

use thiserror::Error;

pub use self::source::{EntryLinkage, EntryPoint, KernelSource, SourceLanguage, SourceParameter};

use crate::{
    ir::{Kernel, VerifyError},
    schedule::{ScheduleError, size_values, space::operand_lengths},
};

/// Increments whenever an emitter's output for some kernel and schedule changes, so recorded timings stop matching.
pub const EMITTER_VERSION: u32 = 8;

/// A reason an emitter cannot write a kernel.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum EmitError {
    #[error("kernel is invalid")]
    Verify(#[from] VerifyError),
    #[error("schedule is invalid")]
    Schedule(#[from] ScheduleError),
    #[error("the emitter does not support {reason}")]
    Unsupported { reason: &'static str },
}

/// The width of the unsigned integers a kernel computes its indices and offsets in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndexWidth {
    U32,
    U64,
}

impl IndexWidth {
    /// Returns the narrowest width that holds every index and offset of `kernel` launched with argument sizes `arguments`.
    ///
    /// That is 32 bits when every operand's span and the product of every axis's extent stay below 2^31, and the kernel addresses no operand through index tensors, whose values only the kernel checks.
    #[must_use]
    pub fn of(kernel: &Kernel, arguments: &[u64]) -> Self {
        const LIMIT: u64 = 1 << 31;
        let Ok(sizes) = size_values(kernel, arguments) else {
            return Self::U64;
        };
        let domain = kernel.axes.iter().try_fold(1_u64, |product, axis| {
            product.checked_mul(sizes[axis.extent.0])
        });
        let fits = domain.is_some_and(|domain| domain < LIMIT)
            && sizes.iter().all(|&size| size < LIMIT)
            && operand_lengths(kernel, arguments)
                .is_some_and(|lengths| lengths.iter().all(|&length| length < LIMIT));
        if fits { Self::U32 } else { Self::U64 }
    }
}
