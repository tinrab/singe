use crate::{
    context::StreamContext,
    error::Result,
    image::{
        exchange,
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView},
    },
    pipeline::image::ImagePipeline,
};

/// # Safety
/// Complete borrowed native work on every path and preserve the native element ABI.
/// Successful dispatch initializes every physical channel in the ROI, including alpha.
pub unsafe trait ScaleToF32Image<L> {
    fn scale_to_f32_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, L>,
        destination: &mut ImageOutputView<'_, f32, L>,
        min: f32,
        max: f32,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and preserve the native element ABI.
/// Successful dispatch initializes every physical channel in the ROI, including alpha.
pub unsafe trait ScaleToI32Image<L> {
    fn scale_to_i32_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, L>,
        destination: &mut ImageOutputView<'_, i32, L>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and preserve the native element ABI.
/// Successful dispatch initializes every physical channel in the ROI, including alpha.
pub unsafe trait ScaleToU16Image<L> {
    fn scale_to_u16_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, L>,
        destination: &mut ImageOutputView<'_, u16, L>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and preserve the native element ABI.
/// Successful dispatch initializes every physical channel in the ROI, including alpha.
pub unsafe trait ScaleToI16Image<L> {
    fn scale_to_i16_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, L>,
        destination: &mut ImageOutputView<'_, i16, L>,
    ) -> Result<()>;
}

macro_rules! impl_scale_to_f32_image {
    ($layout:ty, $physical:ty, $scale:path) => {
        unsafe impl<'a> ScaleToF32Image<$layout> for ImagePipeline<'a, u8, $layout> {
            fn scale_to_f32_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, u8, $layout>,
                destination: &mut ImageOutputView<'_, f32, $layout>,
                min: f32,
                max: f32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context.sync_scope(|| {
                    let source = source.with_layout::<$physical>()?;
                    let mut destination = destination.output().with_layout::<$physical>()?;
                    // SAFETY: native dispatch writes every physical channel and the scope retains
                    // every borrowed participant through completion.
                    unsafe { $scale(stream_context, &source, &mut destination, min, max) }
                })
            }
        }
    };
}

macro_rules! impl_scale_to_i32_image {
    ($layout:ty, $physical:ty, $scale:path) => {
        unsafe impl<'a> ScaleToI32Image<$layout> for ImagePipeline<'a, u8, $layout> {
            fn scale_to_i32_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, u8, $layout>,
                destination: &mut ImageOutputView<'_, i32, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context.sync_scope(|| {
                    let source = source.with_layout::<$physical>()?;
                    let mut destination = destination.output().with_layout::<$physical>()?;
                    // SAFETY: native dispatch writes every physical channel and the scope retains
                    // every borrowed participant through completion.
                    unsafe { $scale(stream_context, &source, &mut destination) }
                })
            }
        }
    };
}

macro_rules! impl_scale_to_u16_image {
    ($layout:ty, $physical:ty, $scale:path) => {
        unsafe impl<'a> ScaleToU16Image<$layout> for ImagePipeline<'a, u8, $layout> {
            fn scale_to_u16_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, u8, $layout>,
                destination: &mut ImageOutputView<'_, u16, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context.sync_scope(|| {
                    let source = source.with_layout::<$physical>()?;
                    let mut destination = destination.output().with_layout::<$physical>()?;
                    // SAFETY: native dispatch writes every physical channel and the scope retains
                    // every borrowed participant through completion.
                    unsafe { $scale(stream_context, &source, &mut destination) }
                })
            }
        }
    };
}

macro_rules! impl_scale_to_i16_image {
    ($layout:ty, $physical:ty, $scale:path) => {
        unsafe impl<'a> ScaleToI16Image<$layout> for ImagePipeline<'a, u8, $layout> {
            fn scale_to_i16_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, u8, $layout>,
                destination: &mut ImageOutputView<'_, i16, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context.sync_scope(|| {
                    let source = source.with_layout::<$physical>()?;
                    let mut destination = destination.output().with_layout::<$physical>()?;
                    // SAFETY: native dispatch writes every physical channel and the scope retains
                    // every borrowed participant through completion.
                    unsafe { $scale(stream_context, &source, &mut destination) }
                })
            }
        }
    };
}

impl_scale_to_f32_image!(C1, C1, exchange::scale_u8_to_f32_c1);

impl_scale_to_f32_image!(C3, C3, exchange::scale_u8_to_f32_c3);

impl_scale_to_f32_image!(C4, C4, exchange::scale_u8_to_f32_c4);

impl_scale_to_f32_image!(AC4, C4, exchange::scale_u8_to_f32_c4);

impl_scale_to_i32_image!(C1, C1, exchange::scale_u8_to_i32_c1);

impl_scale_to_i32_image!(C3, C3, exchange::scale_u8_to_i32_c3);

impl_scale_to_i32_image!(C4, C4, exchange::scale_u8_to_i32_c4);

impl_scale_to_i32_image!(AC4, C4, exchange::scale_u8_to_i32_c4);

impl_scale_to_u16_image!(C1, C1, exchange::scale_u8_to_u16_c1);

impl_scale_to_u16_image!(C3, C3, exchange::scale_u8_to_u16_c3);

impl_scale_to_u16_image!(C4, C4, exchange::scale_u8_to_u16_c4);

impl_scale_to_u16_image!(AC4, C4, exchange::scale_u8_to_u16_c4);

impl_scale_to_i16_image!(C1, C1, exchange::scale_u8_to_i16_c1);

impl_scale_to_i16_image!(C3, C3, exchange::scale_u8_to_i16_c3);

impl_scale_to_i16_image!(C4, C4, exchange::scale_u8_to_i16_c4);

impl_scale_to_i16_image!(AC4, C4, exchange::scale_u8_to_i16_c4);
