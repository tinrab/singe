use super::*;

#[path = "statistics_scalar_sum.rs"]
mod sum;
pub use sum::*;

#[path = "statistics_scalar_min_max.rs"]
mod min_max;
pub use min_max::*;

#[path = "statistics_scalar_min_max_pair.rs"]
mod min_max_pair;
pub use min_max_pair::*;

#[path = "statistics_scalar_dispatch.rs"]
mod dispatch;
pub use dispatch::*;
