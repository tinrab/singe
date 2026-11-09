//! Native image operations on an explicitly selected stream.
//!
//! # Safety
//! Retain each allocation until completion or its last captured replay and establish access ordering.
//! Input and read-write storage must contain valid initialized values for the complete native footprint.
//! Non-border spatial operations require initialized source halo beyond the output ROI.
//! Outputs promise writable capacity only; promotion requires all logical samples to be initialized.
//! Helpers owning temporary scratch or device descriptors complete their native use before releasing it.
//! Such helpers reject capture before allocating those temporary resources.

mod profiles;
#[macro_use]
mod direct_macros;
#[macro_use]
mod generic_macros;
mod dot_product;
mod error_metrics;
mod every;
mod indexed;
mod mean;
mod norms;
mod scalar;

pub use self::{
    dot_product::*, error_metrics::*, every::*, indexed::*, mean::*, norms::*, profiles::*,
    scalar::*,
};
pub use crate::image::{
    statistics_batch_metrics::*, statistics_histograms::*, statistics_integral::*,
    statistics_pair_metrics::*, statistics_template::*,
};
