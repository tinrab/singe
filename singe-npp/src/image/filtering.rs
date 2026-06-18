use singe_cuda::memory::DeviceMemory;
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::view::{AC4, C1, C3, C4, ChannelLayout, ImageView, ImageViewMut},
    try_ffi,
    types::{
        BorderType, CompressedMarkerLabelsInfo, ContourPixelDirectionInfo, ContourTotalsInfo,
        DataTypeLike, DifferentialKernel, ImageNormalization, MaskSize, Point, PointPolar, Size,
        WatershedSegmentBoundaryType,
    },
    utility::to_usize,
};

use super::filtering_validation::*;

pub use super::filtering_distance::*;
pub use super::filtering_flood_fill::*;
pub use super::filtering_hog::*;
pub use super::filtering_labels::*;
pub use super::filtering_separable::*;

#[macro_use]
#[path = "filtering_mask_macros.rs"]
mod mask_macros;

#[macro_use]
#[path = "filtering_advanced_macros.rs"]
mod advanced_macros;

#[macro_use]
#[path = "filtering_directional_macros.rs"]
mod directional_macros;

#[macro_use]
#[path = "filtering_kernel_macros.rs"]
mod kernel_macros;

#[macro_use]
#[path = "filtering_median_macros.rs"]
mod median_macros;

#[path = "filtering_adaptive_threshold.rs"]
mod adaptive_threshold;
pub use adaptive_threshold::*;

#[path = "filtering_basic.rs"]
mod basic;
pub use basic::*;

#[path = "filtering_median.rs"]
mod median;
pub use median::*;
#[path = "filtering_high_pass.rs"]
mod high_pass;
pub use high_pass::*;
#[path = "filtering_sharpen.rs"]
mod sharpen;
pub use sharpen::*;
#[path = "filtering_low_pass.rs"]
mod low_pass;
pub use low_pass::*;
#[path = "filtering_gaussian.rs"]
mod gaussian;
pub use gaussian::*;
#[path = "filtering_directional.rs"]
mod directional;
pub use directional::*;

#[path = "filtering_features.rs"]
mod features;
pub use features::*;

#[path = "filtering_gradient_vector.rs"]
mod gradient_vector;
pub use gradient_vector::*;

#[path = "filtering_wiener.rs"]
mod wiener;
pub use wiener::*;

#[path = "filtering_segmentation.rs"]
mod segmentation;
pub use segmentation::*;
