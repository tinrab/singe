//! Native image operations on an explicitly selected stream.
//!
//! # Safety
//! Retain each allocation until completion or its last captured replay and establish access ordering.
//! Input and read-write storage must contain valid initialized values for the complete native footprint.
//! Non-border spatial operations require initialized source halo beyond the output ROI.
//! Outputs promise writable capacity only; promotion requires all logical samples to be initialized.
//! Helpers owning temporary scratch or device descriptors complete their native use before releasing it.
//! Such helpers reject capture before allocating those temporary resources.

#[macro_use]
mod mask_macros;
#[macro_use]
mod advanced_macros;
#[macro_use]
mod directional_macros;
#[macro_use]
mod kernel_macros;
#[macro_use]
mod median_macros;
mod adaptive_threshold;
mod basic;
mod directional;
mod features;
mod gaussian;
mod gradient_vector;
mod high_pass;
mod low_pass;
mod median;
mod segmentation;
mod sharpen;
mod wiener;

pub use self::{
    adaptive_threshold::*, basic::*, directional::*, features::*, gaussian::*, gradient_vector::*,
    high_pass::*, low_pass::*, median::*, segmentation::*, sharpen::*, wiener::*,
};
pub use crate::image::{
    filtering_distance::*, filtering_flood_fill::*, filtering_hog::*, filtering_labels::*,
    filtering_separable::*,
};
