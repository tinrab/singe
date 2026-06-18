use singe_cuda::memory::DeviceMemory;
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{AC4, C1, C3, C4, ImageView},
    try_ffi,
    types::{DataTypeLike, Size},
    utility::to_usize,
};

use super::statistics_validation::*;

#[path = "statistics_histograms_count.rs"]
mod count;
pub use count::*;

#[path = "statistics_histograms_even.rs"]
mod even;
pub use even::*;

#[path = "statistics_histograms_range.rs"]
mod range;
pub use range::*;
