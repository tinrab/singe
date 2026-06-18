pub(crate) mod arithmetic;
pub(crate) mod conversion;
pub(crate) mod filtering;
pub(crate) mod initialization;
pub mod memory;
pub(crate) mod statistics;
mod statistics_dispatch;
pub mod view;

pub use memory::{Signal, SupportedSignal, create};
pub use view::{SignalView, SignalViewMut};

/// Low-level signal wrappers that keep upstream NPP operation names and shapes.
///
/// Prefer `Signal`, `SignalView`, and `crate::pipeline::SignalPipeline` for
/// the typed Rust API. This module is for porting NPP examples and for
/// operations that do not have a typed pipeline surface yet.
pub mod raw {
    pub mod arithmetic {
        pub use super::super::arithmetic::*;
    }

    pub mod conversion {
        pub use super::super::conversion::*;
    }

    pub mod filtering {
        pub use super::super::filtering::*;
    }

    pub mod initialization {
        pub use super::super::initialization::*;
    }

    pub mod statistics {
        pub use super::super::statistics::*;
    }
}
