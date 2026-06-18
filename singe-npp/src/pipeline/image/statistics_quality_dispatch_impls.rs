use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        statistics,
        view::{C1, C3, C4, ImageView},
    },
};

use super::super::ImagePipeline;
use super::*;

#[path = "statistics_error_metric_dispatch_impls.rs"]
mod error_metric_impls;
#[path = "statistics_quality_norm_dispatch_impls.rs"]
mod norm_impls;

impl_quality_metric_image!(u8, C1, 1, statistics::mse_u8_c1, statistics::psnr_u8_c1);
impl_quality_metric_image!(u8, C3, 3, statistics::mse_u8_c3, statistics::psnr_u8_c3);

impl_structural_similarity_image!(u8, C1, 1, statistics::ssim_u8_c1);
impl_structural_similarity_image!(u8, C3, 3, statistics::ssim_u8_c3);

impl_multiscale_structural_similarity_image!(u8, C1, 1, statistics::msssim_u8_c1);

impl_weighted_multiscale_structural_similarity_image!(u8, C1, 1, statistics::wmsssim_u8_c1);
impl_weighted_multiscale_structural_similarity_image!(u8, C3, 3, statistics::wmsssim_u8_c3);
impl_weighted_multiscale_structural_similarity_image!(u8, C4, 4, statistics::wmsssim_u8_c4);
