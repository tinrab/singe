use singe_cuda::view::DeviceOutputView;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        statistics,
        view::{C1, C3, C4, ImageView},
    },
    pipeline::image::{
        ImagePipeline,
        statistics_dispatch::{
            ErrorMetricImage, MaskedChannelNormDiffImage, MaskedChannelNormRelativeImage,
            MaskedNormDiffImage, MaskedNormRelativeImage, MultiscaleStructuralSimilarityImage,
            QualityMetricImage, StructuralSimilarityImage,
            WeightedMultiscaleStructuralSimilarityImage,
        },
    },
};

mod error_metric_impls;
mod norm_impls;

impl_quality_metric_image!(u8, C1, 1, statistics::mse_u8_c1, statistics::psnr_u8_c1);

impl_quality_metric_image!(u8, C3, 3, statistics::mse_u8_c3, statistics::psnr_u8_c3);

impl_structural_similarity_image!(u8, C1, 1, statistics::ssim_u8_c1);

impl_structural_similarity_image!(u8, C3, 3, statistics::ssim_u8_c3);

impl_multiscale_structural_similarity_image!(u8, C1, 1, statistics::msssim_u8_c1);

impl_weighted_multiscale_structural_similarity_image!(u8, C1, 1, statistics::wmsssim_u8_c1);

impl_weighted_multiscale_structural_similarity_image!(u8, C3, 3, statistics::wmsssim_u8_c3);

impl_weighted_multiscale_structural_similarity_image!(u8, C4, 4, statistics::wmsssim_u8_c4);
