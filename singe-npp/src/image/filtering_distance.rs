use singe_cuda::memory::DeviceMemory;
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, C2, ImageView, ImageViewMut},
    try_ffi,
    types::{DataTypeLike, Size},
    utility::to_usize,
};

use super::filtering_validation::*;

#[path = "filtering_distance_raw.rs"]
mod raw;
pub use raw::*;

#[path = "filtering_distance_unsigned.rs"]
mod unsigned;
pub use unsigned::*;

#[path = "filtering_distance_signed.rs"]
mod signed;
pub use signed::*;
