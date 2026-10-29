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
    filtering_distance::*, filtering_flood_fill::*, filtering_hog::*, filtering_labels::*,
    filtering_separable::*,
};

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

pub use adaptive_threshold::*;

mod basic;

pub use basic::*;

mod median;

pub use median::*;

mod high_pass;

pub use high_pass::*;

mod sharpen;

pub use sharpen::*;

mod low_pass;

pub use low_pass::*;

mod gaussian;

pub use gaussian::*;

mod directional;

pub use directional::*;

mod features;

pub use features::*;

mod gradient_vector;

pub use gradient_vector::*;

mod wiener;

pub use wiener::*;

mod segmentation;

pub use segmentation::*;
