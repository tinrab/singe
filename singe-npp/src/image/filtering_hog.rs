use singe_cuda::memory::DeviceMemory;
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::view::{C1, C3, ChannelLayout, ImageView},
    try_ffi,
    types::{BorderType, DataTypeLike, HistogramOfGradientsConfig, Point, Size},
    utility::to_usize,
};

use super::filtering_validation::*;

pub fn histogram_of_gradients_border_buffer_size(
    config: HistogramOfGradientsConfig,
    locations: &[Point],
    roi: Size,
) -> Result<usize> {
    validate_positive_size(roi)?;
    let location_count = i32::try_from(locations.len()).map_err(|_| Error::OutOfRange {
        name: "location count".into(),
    })?;
    let mut bytes = 0;
    unsafe {
        try_ffi!(sys::nppiHistogramOfGradientsBorderGetBufferSize(
            config.into(),
            locations.as_ptr().cast(),
            location_count,
            roi.into(),
            &raw mut bytes,
        ))?;
    }
    to_usize(bytes, "buffer size")
}

pub fn histogram_of_gradients_border_descriptors_size(
    config: HistogramOfGradientsConfig,
    location_count: usize,
) -> Result<usize> {
    let location_count = i32::try_from(location_count).map_err(|_| Error::OutOfRange {
        name: "location count".into(),
    })?;
    let mut bytes = 0;
    unsafe {
        try_ffi!(sys::nppiHistogramOfGradientsBorderGetDescriptorsSize(
            config.into(),
            location_count,
            &raw mut bytes,
        ))?;
    }
    to_usize(bytes, "descriptors size")
}

macro_rules! impl_histogram_of_gradients_border {
    ($name:ident, $pixel_ty:ty, $layout:ty, $ffi_name:ident) => {
        pub fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            source_offset: Point,
            locations: &[Point],
            descriptors: &mut DeviceMemory<f32>,
            roi: Size,
            config: HistogramOfGradientsConfig,
            scratch: &mut DeviceMemory<u8>,
            border_type: BorderType,
        ) -> Result<()> {
            validate_border_roi(source.size(), source_offset, roi)?;
            let descriptor_bytes =
                histogram_of_gradients_border_descriptors_size(config, locations.len())?;
            if descriptors.byte_len() < descriptor_bytes {
                return Err(Error::OutOfRange {
                    name: "descriptors length".into(),
                });
            }
            let scratch_bytes = histogram_of_gradients_border_buffer_size(config, locations, roi)?;
            if scratch.len() < scratch_bytes {
                return Err(Error::OutOfRange {
                    name: "scratch length".into(),
                });
            }
            let location_count = i32::try_from(locations.len()).map_err(|_| Error::OutOfRange {
                name: "location count".into(),
            })?;
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source.as_ptr().cast(),
                    source.step(),
                    source.size().into(),
                    source_offset.into(),
                    locations.as_ptr().cast(),
                    location_count,
                    descriptors.as_mut_ptr().cast(),
                    roi.into(),
                    config.into(),
                    scratch.as_mut_ptr().cast(),
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

impl_histogram_of_gradients_border!(
    histogram_of_gradients_border_u8_to_f32_c1,
    u8,
    C1,
    nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx
);
impl_histogram_of_gradients_border!(
    histogram_of_gradients_border_u8_to_f32_c3,
    u8,
    C3,
    nppiHistogramOfGradientsBorder_8u32f_C3R_Ctx
);
impl_histogram_of_gradients_border!(
    histogram_of_gradients_border_u16_to_f32_c1,
    u16,
    C1,
    nppiHistogramOfGradientsBorder_16u32f_C1R_Ctx
);
impl_histogram_of_gradients_border!(
    histogram_of_gradients_border_u16_to_f32_c3,
    u16,
    C3,
    nppiHistogramOfGradientsBorder_16u32f_C3R_Ctx
);
impl_histogram_of_gradients_border!(
    histogram_of_gradients_border_i16_to_f32_c1,
    i16,
    C1,
    nppiHistogramOfGradientsBorder_16s32f_C1R_Ctx
);
impl_histogram_of_gradients_border!(
    histogram_of_gradients_border_i16_to_f32_c3,
    i16,
    C3,
    nppiHistogramOfGradientsBorder_16s32f_C3R_Ctx
);
impl_histogram_of_gradients_border!(
    histogram_of_gradients_border_f32_c1,
    f32,
    C1,
    nppiHistogramOfGradientsBorder_32f_C1R_Ctx
);
impl_histogram_of_gradients_border!(
    histogram_of_gradients_border_f32_c3,
    f32,
    C3,
    nppiHistogramOfGradientsBorder_32f_C3R_Ctx
);

pub trait HistogramOfGradientsBorder<L: ChannelLayout>: DataTypeLike {
    fn histogram_of_gradients_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        source_offset: Point,
        locations: &[Point],
        descriptors: &mut DeviceMemory<f32>,
        roi: Size,
        config: HistogramOfGradientsConfig,
        scratch: &mut DeviceMemory<u8>,
        border_type: BorderType,
    ) -> Result<()>;
}

macro_rules! impl_histogram_of_gradients_border_dispatch {
    ($pixel_ty:ty, $layout:ty, $name:ident) => {
        impl HistogramOfGradientsBorder<$layout> for $pixel_ty {
            fn histogram_of_gradients_border(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                source_offset: Point,
                locations: &[Point],
                descriptors: &mut DeviceMemory<f32>,
                roi: Size,
                config: HistogramOfGradientsConfig,
                scratch: &mut DeviceMemory<u8>,
                border_type: BorderType,
            ) -> Result<()> {
                $name(
                    stream_context,
                    source,
                    source_offset,
                    locations,
                    descriptors,
                    roi,
                    config,
                    scratch,
                    border_type,
                )
            }
        }
    };
}

impl_histogram_of_gradients_border_dispatch!(u8, C1, histogram_of_gradients_border_u8_to_f32_c1);
impl_histogram_of_gradients_border_dispatch!(u8, C3, histogram_of_gradients_border_u8_to_f32_c3);
impl_histogram_of_gradients_border_dispatch!(u16, C1, histogram_of_gradients_border_u16_to_f32_c1);
impl_histogram_of_gradients_border_dispatch!(u16, C3, histogram_of_gradients_border_u16_to_f32_c3);
impl_histogram_of_gradients_border_dispatch!(i16, C1, histogram_of_gradients_border_i16_to_f32_c1);
impl_histogram_of_gradients_border_dispatch!(i16, C3, histogram_of_gradients_border_i16_to_f32_c3);
impl_histogram_of_gradients_border_dispatch!(f32, C1, histogram_of_gradients_border_f32_c1);
impl_histogram_of_gradients_border_dispatch!(f32, C3, histogram_of_gradients_border_f32_c3);

pub fn histogram_of_gradients_border<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    source_offset: Point,
    locations: &[Point],
    descriptors: &mut DeviceMemory<f32>,
    roi: Size,
    config: HistogramOfGradientsConfig,
    scratch: &mut DeviceMemory<u8>,
    border_type: BorderType,
) -> Result<()>
where
    T: HistogramOfGradientsBorder<L>,
    L: ChannelLayout,
{
    T::histogram_of_gradients_border(
        stream_context,
        source,
        source_offset,
        locations,
        descriptors,
        roi,
        config,
        scratch,
        border_type,
    )
}
