use super::*;

#[path = "statistics_error_metrics_maximum.rs"]
mod maximum;
pub use maximum::*;

#[path = "statistics_error_metrics_average.rs"]
mod average;
pub use average::*;

#[path = "statistics_error_metrics_maximum_relative.rs"]
mod maximum_relative;
pub use maximum_relative::*;

#[path = "statistics_error_metrics_quality.rs"]
mod quality;
pub use quality::*;
