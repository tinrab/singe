use singe_cuda::memory::DeviceMemory;
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, C3, ImageViewMut},
    try_ffi,
    types::{ConnectedRegion, DataTypeLike, ImageNormalization, Point, Size},
    utility::to_usize,
};

use super::filtering_validation::*;

#[path = "filtering_flood_fill_direct.rs"]
mod direct;
pub use direct::*;

#[path = "filtering_flood_fill_standard.rs"]
mod standard;
pub use standard::*;

#[path = "filtering_flood_fill_gradient.rs"]
mod gradient;
pub use gradient::*;
