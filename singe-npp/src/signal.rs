pub(crate) mod arithmetic;
pub(crate) mod conversion;
pub(crate) mod filtering;
pub(crate) mod initialization;
/// Owned native NPP signal allocations and explicit initialization state.
pub mod memory;
pub(crate) mod statistics;
mod statistics_dispatch;
pub mod view;

pub use memory::{Signal, SupportedSignal};
pub use view::{SignalOutputView, SignalView, SignalViewMut};

/// Public raw NPP signal operations with native enqueue semantics.
///
/// # Execution contract
///
/// Inputs contain initialized values. Pure destinations use `SignalOutputView`;
/// accumulators and in-place operands use initialized `SignalViewMut`. Preserve
/// native datatype, length, and aliasing requirements. Order producer writes
/// before reads, and keep the context, stream, allocations, descriptors, and
/// caller-provided scratch live through completion on errors and graph replay.
/// Success posts native work; it does not establish initialized output values.
///
/// Scratch-owning convenience calls complete before releasing temporary storage
/// and reject capture. Use the public forms with caller-owned scratch for capture.
/// Completed fluent pipelines retain their arguments through stream scopes.
pub mod raw {
    /// Raw vector arithmetic and elementwise transforms.
    pub mod arithmetic {
        pub use crate::signal::arithmetic::*;
    }

    /// Raw signal type conversions and constant-level thresholding.
    pub mod conversion {
        pub use crate::signal::conversion::*;
    }

    /// Raw cumulative sums and their scratch-buffer requirements.
    pub mod filtering {
        pub use crate::signal::filtering::*;
    }

    /// Raw signal initialization and element-setting operations.
    pub mod initialization {
        pub use crate::signal::initialization::*;
    }

    /// Raw reductions, extrema, norms, error metrics, and range counts.
    pub mod statistics {
        pub use crate::signal::statistics::*;
    }
}
