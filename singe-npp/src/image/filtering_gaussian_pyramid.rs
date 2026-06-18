use super::*;

pub fn filter_gauss_pyramid_layer_down_border_dst_roi(source_roi: Size, rate: f32) -> Result<Size> {
    validate_positive_size(source_roi)?;
    validate_gauss_pyramid_rate(rate)?;
    let mut destination = sys::NppiSize {
        width: 0,
        height: 0,
    };
    unsafe {
        try_ffi!(sys::nppiGetFilterGaussPyramidLayerDownBorderDstROI(
            source_roi.width,
            source_roi.height,
            &raw mut destination,
            rate,
        ))?;
    }
    Ok(destination.into())
}

pub fn filter_gauss_pyramid_layer_up_border_dst_roi(
    source_roi: Size,
    rate: f32,
) -> Result<(Size, Size)> {
    validate_positive_size(source_roi)?;
    validate_gauss_pyramid_rate(rate)?;
    let mut destination_min = sys::NppiSize {
        width: 0,
        height: 0,
    };
    let mut destination_max = sys::NppiSize {
        width: 0,
        height: 0,
    };
    unsafe {
        try_ffi!(sys::nppiGetFilterGaussPyramidLayerUpBorderDstROI(
            source_roi.width,
            source_roi.height,
            &raw mut destination_min,
            &raw mut destination_max,
            rate,
        ))?;
    }
    Ok((destination_min.into(), destination_max.into()))
}

impl_filter_gauss_pyramid_border!(
    filter_gauss_pyramid_layer_down_border_u8_c1,
    u8,
    C1,
    nppiFilterGaussPyramidLayerDownBorder_8u_C1R_Ctx
);
impl_filter_gauss_pyramid_border!(
    filter_gauss_pyramid_layer_down_border_u8_c3,
    u8,
    C3,
    nppiFilterGaussPyramidLayerDownBorder_8u_C3R_Ctx
);
impl_filter_gauss_pyramid_border!(
    filter_gauss_pyramid_layer_down_border_u16_c1,
    u16,
    C1,
    nppiFilterGaussPyramidLayerDownBorder_16u_C1R_Ctx
);
impl_filter_gauss_pyramid_border!(
    filter_gauss_pyramid_layer_down_border_u16_c3,
    u16,
    C3,
    nppiFilterGaussPyramidLayerDownBorder_16u_C3R_Ctx
);
impl_filter_gauss_pyramid_border!(
    filter_gauss_pyramid_layer_down_border_f32_c1,
    f32,
    C1,
    nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx
);
impl_filter_gauss_pyramid_border!(
    filter_gauss_pyramid_layer_down_border_f32_c3,
    f32,
    C3,
    nppiFilterGaussPyramidLayerDownBorder_32f_C3R_Ctx
);

impl_filter_gauss_pyramid_border!(
    filter_gauss_pyramid_layer_up_border_u8_c1,
    u8,
    C1,
    nppiFilterGaussPyramidLayerUpBorder_8u_C1R_Ctx
);
impl_filter_gauss_pyramid_border!(
    filter_gauss_pyramid_layer_up_border_u8_c3,
    u8,
    C3,
    nppiFilterGaussPyramidLayerUpBorder_8u_C3R_Ctx
);
impl_filter_gauss_pyramid_border!(
    filter_gauss_pyramid_layer_up_border_u16_c1,
    u16,
    C1,
    nppiFilterGaussPyramidLayerUpBorder_16u_C1R_Ctx
);
impl_filter_gauss_pyramid_border!(
    filter_gauss_pyramid_layer_up_border_u16_c3,
    u16,
    C3,
    nppiFilterGaussPyramidLayerUpBorder_16u_C3R_Ctx
);
impl_filter_gauss_pyramid_border!(
    filter_gauss_pyramid_layer_up_border_f32_c1,
    f32,
    C1,
    nppiFilterGaussPyramidLayerUpBorder_32f_C1R_Ctx
);
impl_filter_gauss_pyramid_border!(
    filter_gauss_pyramid_layer_up_border_f32_c3,
    f32,
    C3,
    nppiFilterGaussPyramidLayerUpBorder_32f_C3R_Ctx
);

pub trait FilterGaussPyramidLayerDownBorder<L: ChannelLayout>: DataTypeLike {
    fn filter_gauss_pyramid_layer_down_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, Self, L>,
        rate: f32,
        kernel: &[f32],
        border_type: BorderType,
    ) -> Result<()>;
}

macro_rules! impl_filter_gauss_pyramid_layer_down_border_dispatch {
    ($pixel_ty:ty, $layout:ty, $name:ident) => {
        impl FilterGaussPyramidLayerDownBorder<$layout> for $pixel_ty {
            fn filter_gauss_pyramid_layer_down_border(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, Self, $layout>,
                rate: f32,
                kernel: &[f32],
                border_type: BorderType,
            ) -> Result<()> {
                $name(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    rate,
                    kernel,
                    border_type,
                )
            }
        }
    };
}

impl_filter_gauss_pyramid_layer_down_border_dispatch!(
    u8,
    C1,
    filter_gauss_pyramid_layer_down_border_u8_c1
);
impl_filter_gauss_pyramid_layer_down_border_dispatch!(
    u8,
    C3,
    filter_gauss_pyramid_layer_down_border_u8_c3
);
impl_filter_gauss_pyramid_layer_down_border_dispatch!(
    u16,
    C1,
    filter_gauss_pyramid_layer_down_border_u16_c1
);
impl_filter_gauss_pyramid_layer_down_border_dispatch!(
    u16,
    C3,
    filter_gauss_pyramid_layer_down_border_u16_c3
);
impl_filter_gauss_pyramid_layer_down_border_dispatch!(
    f32,
    C1,
    filter_gauss_pyramid_layer_down_border_f32_c1
);
impl_filter_gauss_pyramid_layer_down_border_dispatch!(
    f32,
    C3,
    filter_gauss_pyramid_layer_down_border_f32_c3
);

pub fn filter_gauss_pyramid_layer_down_border<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, T, L>,
    rate: f32,
    kernel: &[f32],
    border_type: BorderType,
) -> Result<()>
where
    T: FilterGaussPyramidLayerDownBorder<L>,
    L: ChannelLayout,
{
    T::filter_gauss_pyramid_layer_down_border(
        stream_context,
        source,
        source_offset,
        destination,
        rate,
        kernel,
        border_type,
    )
}

pub trait FilterGaussPyramidLayerUpBorder<L: ChannelLayout>: DataTypeLike {
    fn filter_gauss_pyramid_layer_up_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, Self, L>,
        rate: f32,
        kernel: &[f32],
        border_type: BorderType,
    ) -> Result<()>;
}

macro_rules! impl_filter_gauss_pyramid_layer_up_border_dispatch {
    ($pixel_ty:ty, $layout:ty, $name:ident) => {
        impl FilterGaussPyramidLayerUpBorder<$layout> for $pixel_ty {
            fn filter_gauss_pyramid_layer_up_border(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, Self, $layout>,
                rate: f32,
                kernel: &[f32],
                border_type: BorderType,
            ) -> Result<()> {
                $name(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    rate,
                    kernel,
                    border_type,
                )
            }
        }
    };
}

impl_filter_gauss_pyramid_layer_up_border_dispatch!(
    u8,
    C1,
    filter_gauss_pyramid_layer_up_border_u8_c1
);
impl_filter_gauss_pyramid_layer_up_border_dispatch!(
    u8,
    C3,
    filter_gauss_pyramid_layer_up_border_u8_c3
);
impl_filter_gauss_pyramid_layer_up_border_dispatch!(
    u16,
    C1,
    filter_gauss_pyramid_layer_up_border_u16_c1
);
impl_filter_gauss_pyramid_layer_up_border_dispatch!(
    u16,
    C3,
    filter_gauss_pyramid_layer_up_border_u16_c3
);
impl_filter_gauss_pyramid_layer_up_border_dispatch!(
    f32,
    C1,
    filter_gauss_pyramid_layer_up_border_f32_c1
);
impl_filter_gauss_pyramid_layer_up_border_dispatch!(
    f32,
    C3,
    filter_gauss_pyramid_layer_up_border_f32_c3
);

pub fn filter_gauss_pyramid_layer_up_border<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    source_offset: Point,
    destination: &mut ImageViewMut<'_, T, L>,
    rate: f32,
    kernel: &[f32],
    border_type: BorderType,
) -> Result<()>
where
    T: FilterGaussPyramidLayerUpBorder<L>,
    L: ChannelLayout,
{
    T::filter_gauss_pyramid_layer_up_border(
        stream_context,
        source,
        source_offset,
        destination,
        rate,
        kernel,
        border_type,
    )
}
