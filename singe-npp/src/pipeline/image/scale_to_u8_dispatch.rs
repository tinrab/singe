use crate::{
    context::StreamContext,
    error::Result,
    image::{
        exchange,
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView},
    },
    pipeline::image::ImagePipeline,
    types::HintAlgorithm,
};

/// # Safety
/// Complete borrowed native work on every path and preserve the native element ABI.
/// Successful dispatch initializes every physical channel in the ROI, including alpha.
pub unsafe trait ScaleToU8Image<L> {
    fn scale_to_u8_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, f32, L>,
        destination: &mut ImageOutputView<'_, u8, L>,
        min: f32,
        max: f32,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and preserve the native element ABI.
/// Successful dispatch initializes every physical channel in the ROI, including alpha.
pub unsafe trait ScaleI32ToU8Image<L> {
    fn scale_i32_to_u8_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, i32, L>,
        destination: &mut ImageOutputView<'_, u8, L>,
        hint: HintAlgorithm,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and preserve the native element ABI.
/// Successful dispatch initializes every physical channel in the ROI, including alpha.
pub unsafe trait ScaleU16ToU8Image<L> {
    fn scale_u16_to_u8_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, u16, L>,
        destination: &mut ImageOutputView<'_, u8, L>,
        hint: HintAlgorithm,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and preserve the native element ABI.
/// Successful dispatch initializes every physical channel in the ROI, including alpha.
pub unsafe trait ScaleI16ToU8Image<L> {
    fn scale_i16_to_u8_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, i16, L>,
        destination: &mut ImageOutputView<'_, u8, L>,
        hint: HintAlgorithm,
    ) -> Result<()>;
}

macro_rules! impl_scale_to_u8_image {
    ($layout:ty, $physical:ty, $scale:path) => {
        unsafe impl<'a> ScaleToU8Image<$layout> for ImagePipeline<'a, f32, $layout> {
            fn scale_to_u8_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, f32, $layout>,
                destination: &mut ImageOutputView<'_, u8, $layout>,
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

macro_rules! impl_scale_i32_to_u8_image {
    ($layout:ty, $physical:ty, $scale:path) => {
        unsafe impl<'a> ScaleI32ToU8Image<$layout> for ImagePipeline<'a, i32, $layout> {
            fn scale_i32_to_u8_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, i32, $layout>,
                destination: &mut ImageOutputView<'_, u8, $layout>,
                hint: HintAlgorithm,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context.sync_scope(|| {
                    let source = source.with_layout::<$physical>()?;
                    let mut destination = destination.output().with_layout::<$physical>()?;
                    // SAFETY: native dispatch writes every physical channel and the scope retains
                    // every borrowed participant through completion.
                    unsafe { $scale(stream_context, &source, &mut destination, hint) }
                })
            }
        }
    };
}

macro_rules! impl_scale_u16_to_u8_image {
    ($layout:ty, $physical:ty, $scale:path) => {
        unsafe impl<'a> ScaleU16ToU8Image<$layout> for ImagePipeline<'a, u16, $layout> {
            fn scale_u16_to_u8_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, u16, $layout>,
                destination: &mut ImageOutputView<'_, u8, $layout>,
                hint: HintAlgorithm,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context.sync_scope(|| {
                    let source = source.with_layout::<$physical>()?;
                    let mut destination = destination.output().with_layout::<$physical>()?;
                    // SAFETY: native dispatch writes every physical channel and the scope retains
                    // every borrowed participant through completion.
                    unsafe { $scale(stream_context, &source, &mut destination, hint) }
                })
            }
        }
    };
}

macro_rules! impl_scale_i16_to_u8_image {
    ($layout:ty, $physical:ty, $scale:path) => {
        unsafe impl<'a> ScaleI16ToU8Image<$layout> for ImagePipeline<'a, i16, $layout> {
            fn scale_i16_to_u8_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, i16, $layout>,
                destination: &mut ImageOutputView<'_, u8, $layout>,
                hint: HintAlgorithm,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context.sync_scope(|| {
                    let source = source.with_layout::<$physical>()?;
                    let mut destination = destination.output().with_layout::<$physical>()?;
                    // SAFETY: native dispatch writes every physical channel and the scope retains
                    // every borrowed participant through completion.
                    unsafe { $scale(stream_context, &source, &mut destination, hint) }
                })
            }
        }
    };
}

impl_scale_to_u8_image!(C1, C1, exchange::scale_f32_to_u8_c1);

impl_scale_to_u8_image!(C3, C3, exchange::scale_f32_to_u8_c3);

impl_scale_to_u8_image!(C4, C4, exchange::scale_f32_to_u8_c4);

impl_scale_to_u8_image!(AC4, C4, exchange::scale_f32_to_u8_c4);

impl_scale_i32_to_u8_image!(C1, C1, exchange::scale_i32_to_u8_c1);

impl_scale_i32_to_u8_image!(C3, C3, exchange::scale_i32_to_u8_c3);

impl_scale_i32_to_u8_image!(C4, C4, exchange::scale_i32_to_u8_c4);

impl_scale_i32_to_u8_image!(AC4, C4, exchange::scale_i32_to_u8_c4);

impl_scale_u16_to_u8_image!(C1, C1, exchange::scale_u16_to_u8_c1);

impl_scale_u16_to_u8_image!(C3, C3, exchange::scale_u16_to_u8_c3);

impl_scale_u16_to_u8_image!(C4, C4, exchange::scale_u16_to_u8_c4);

impl_scale_u16_to_u8_image!(AC4, C4, exchange::scale_u16_to_u8_c4);

impl_scale_i16_to_u8_image!(C1, C1, exchange::scale_i16_to_u8_c1);

impl_scale_i16_to_u8_image!(C3, C3, exchange::scale_i16_to_u8_c3);

impl_scale_i16_to_u8_image!(C4, C4, exchange::scale_i16_to_u8_c4);

impl_scale_i16_to_u8_image!(AC4, C4, exchange::scale_i16_to_u8_c4);
