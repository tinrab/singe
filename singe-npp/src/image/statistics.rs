//! Native image operations on an explicitly selected stream.
//!
//! # Safety
//! Retain each allocation until completion or its last captured replay and establish access ordering.
//! Input and read-write storage must contain valid initialized values for the complete native footprint.
//! Non-border spatial operations require initialized source halo beyond the output ROI.
//! Outputs promise writable capacity only; promotion requires all logical samples to be initialized.
//! Helpers owning temporary scratch or device descriptors complete their native use before releasing it.
//! Such helpers reject capture before allocating those temporary resources.

pub use crate::image::{
    statistics_batch_metrics::*, statistics_histograms::*, statistics_integral::*,
    statistics_pair_metrics::*, statistics_template::*,
};

mod profiles;

pub use profiles::*;

#[macro_use]
mod direct_macros;

#[macro_use]
mod generic_macros;

mod mean;

pub use mean::*;

mod scalar;

pub use scalar::*;

mod indexed;

pub use indexed::*;

mod dot_product;

pub use dot_product::*;

mod every;

pub use every::*;

mod error_metrics;

pub use error_metrics::*;

mod norms;

pub use norms::*;
