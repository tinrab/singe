use singe_cuda::{memory::DeviceMemory, types::Complex32};
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::view::{AC4, C1, C2, C3, C4, ImageView, ImageViewMut},
    try_ffi,
    types::{ComplexI16, ComplexI32, DataTypeLike, Point, Point32f, ProfileData, Size},
    utility::to_usize,
};

use super::statistics_validation::*;

pub use super::statistics_batch_metrics::*;
pub use super::statistics_histograms::*;
pub use super::statistics_integral::*;
pub use super::statistics_pair_metrics::*;
pub use super::statistics_template::*;

#[path = "statistics_profiles.rs"]
mod profiles;
pub use profiles::*;

#[macro_use]
#[path = "statistics_direct_macros.rs"]
mod direct_macros;

#[macro_use]
#[path = "statistics_generic_macros.rs"]
mod generic_macros;

#[path = "statistics_mean.rs"]
mod mean;
pub use mean::*;

#[path = "statistics_scalar.rs"]
mod scalar;
pub use scalar::*;

#[path = "statistics_indexed.rs"]
mod indexed;
pub use indexed::*;

#[path = "statistics_dot_product.rs"]
mod dot_product;
pub use dot_product::*;

#[path = "statistics_every.rs"]
mod every;
pub use every::*;

#[path = "statistics_error_metrics.rs"]
mod error_metrics;
pub use error_metrics::*;

#[path = "statistics_norms.rs"]
mod norms;
pub use norms::*;
