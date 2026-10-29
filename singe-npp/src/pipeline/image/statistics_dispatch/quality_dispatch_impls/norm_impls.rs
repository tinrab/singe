use singe_cuda::view::DeviceOutputView;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        statistics,
        view::{C1, C3, ImageView, MaskView},
    },
    pipeline::image::{
        ImagePipeline,
        statistics_dispatch::quality_dispatch_impls::{
            MaskedChannelNormDiffImage, MaskedChannelNormRelativeImage, MaskedNormDiffImage,
            MaskedNormRelativeImage,
        },
    },
};

impl_masked_norm_diff_image!(
    u8,
    statistics::norm_diff_inf_u8_c1_masked,
    statistics::norm_diff_l1_u8_c1_masked,
    statistics::norm_diff_l2_u8_c1_masked
);

impl_masked_norm_diff_image!(
    i8,
    statistics::norm_diff_inf_i8_c1_masked,
    statistics::norm_diff_l1_i8_c1_masked,
    statistics::norm_diff_l2_i8_c1_masked
);

impl_masked_norm_diff_image!(
    u16,
    statistics::norm_diff_inf_u16_c1_masked,
    statistics::norm_diff_l1_u16_c1_masked,
    statistics::norm_diff_l2_u16_c1_masked
);

impl_masked_norm_diff_image!(
    f32,
    statistics::norm_diff_inf_f32_c1_masked,
    statistics::norm_diff_l1_f32_c1_masked,
    statistics::norm_diff_l2_f32_c1_masked
);

impl_masked_channel_norm_diff_image!(
    u8,
    statistics::norm_diff_inf_u8_c3_masked,
    statistics::norm_diff_l1_u8_c3_masked,
    statistics::norm_diff_l2_u8_c3_masked
);

impl_masked_channel_norm_diff_image!(
    i8,
    statistics::norm_diff_inf_i8_c3_masked,
    statistics::norm_diff_l1_i8_c3_masked,
    statistics::norm_diff_l2_i8_c3_masked
);

impl_masked_channel_norm_diff_image!(
    u16,
    statistics::norm_diff_inf_u16_c3_masked,
    statistics::norm_diff_l1_u16_c3_masked,
    statistics::norm_diff_l2_u16_c3_masked
);

impl_masked_channel_norm_diff_image!(
    f32,
    statistics::norm_diff_inf_f32_c3_masked,
    statistics::norm_diff_l1_f32_c3_masked,
    statistics::norm_diff_l2_f32_c3_masked
);

impl_masked_norm_relative_image!(
    u8,
    statistics::norm_rel_inf_u8_c1_masked,
    statistics::norm_rel_l1_u8_c1_masked,
    statistics::norm_rel_l2_u8_c1_masked
);

impl_masked_norm_relative_image!(
    i8,
    statistics::norm_rel_inf_i8_c1_masked,
    statistics::norm_rel_l1_i8_c1_masked,
    statistics::norm_rel_l2_i8_c1_masked
);

impl_masked_norm_relative_image!(
    u16,
    statistics::norm_rel_inf_u16_c1_masked,
    statistics::norm_rel_l1_u16_c1_masked,
    statistics::norm_rel_l2_u16_c1_masked
);

impl_masked_norm_relative_image!(
    f32,
    statistics::norm_rel_inf_f32_c1_masked,
    statistics::norm_rel_l1_f32_c1_masked,
    statistics::norm_rel_l2_f32_c1_masked
);

impl_masked_channel_norm_relative_image!(
    u8,
    statistics::norm_rel_inf_u8_c3_masked,
    statistics::norm_rel_l1_u8_c3_masked,
    statistics::norm_rel_l2_u8_c3_masked
);

impl_masked_channel_norm_relative_image!(
    i8,
    statistics::norm_rel_inf_i8_c3_masked,
    statistics::norm_rel_l1_i8_c3_masked,
    statistics::norm_rel_l2_i8_c3_masked
);

impl_masked_channel_norm_relative_image!(
    u16,
    statistics::norm_rel_inf_u16_c3_masked,
    statistics::norm_rel_l1_u16_c3_masked,
    statistics::norm_rel_l2_u16_c3_masked
);

impl_masked_channel_norm_relative_image!(
    f32,
    statistics::norm_rel_inf_f32_c3_masked,
    statistics::norm_rel_l1_f32_c3_masked,
    statistics::norm_rel_l2_f32_c3_masked
);
