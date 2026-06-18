use singe_cuda::{memory::DeviceMemory, types::f16};
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{
        AC4, C1, C2, C3, C4, ChannelLayout, ImageView, ImageViewMut, PlanarImageView,
        PlanarImageViewMut,
    },
    try_ffi,
    types::{
        BayerGridPosition, ColorSpace, DataTypeLike, ImageNormalization, InterpolationMode, Point,
        Rectangle,
    },
};

use super::color_descriptors::*;

pub type ColorTwistMatrix = [[f32; 4]; 3];
pub type ColorTwistMatrix4 = [[f32; 4]; 4];
pub type ColorTwistConstants4 = [f32; 4];
pub type ColorTwistBatchConstantsMatrix4 = [[f32; 5]; 4];

#[path = "color_lookup.rs"]
mod lookup;
pub use lookup::*;

#[macro_use]
#[path = "color_twist_macros.rs"]
mod twist_macros;

#[path = "color_twist.rs"]
mod twist;
pub use twist::*;

#[macro_use]
#[path = "color_grayscale_macros.rs"]
mod grayscale_macros;

#[path = "color_grayscale.rs"]
mod grayscale;
pub use grayscale::*;

#[macro_use]
#[path = "color_conversion_uyvp_macros.rs"]
mod conversion_uyvp_macros;

#[macro_use]
#[path = "color_conversion_generic_macros.rs"]
mod conversion_generic_macros;

#[macro_use]
#[path = "color_conversion_jpeg_macros.rs"]
mod conversion_jpeg_macros;

#[macro_use]
#[path = "color_conversion_twist_macros.rs"]
mod conversion_twist_macros;

#[macro_use]
#[path = "color_conversion_nv12_macros.rs"]
mod conversion_nv12_macros;

#[macro_use]
#[path = "color_conversion_subsampled_macros.rs"]
mod conversion_subsampled_macros;

#[path = "color_uyvp.rs"]
mod uyvp;
pub use uyvp::*;
#[path = "color_yuv.rs"]
mod yuv;
pub use yuv::*;
#[path = "color_ycbcr.rs"]
mod ycbcr;
pub use ycbcr::*;
#[path = "color_subsampled.rs"]
mod subsampled;
pub use subsampled::*;
