use singe_cuda::types::f16;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        exchange,
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView},
    },
    pipeline::image::convert::ImagePipeline,
};

/// Dispatches completed image element conversions.
///
/// # Safety
///
/// Implementations must preserve native source and destination representations and numeric
/// conversion semantics. Complete borrowed native work on every path. On success, initialize
/// every physical channel in the ROI, including alpha. Uphold the completed native dispatch
/// contract of [`ImagePipeline`](crate::pipeline::ImagePipeline).
pub unsafe trait ConvertImage<T, U, L> {
    /// Converts each image sample to `U` using native conversion semantics.
    /// Writes the converted samples into the equally sized `destination`.
    /// Processes every physical channel, including alpha in AC4 layouts. Row padding remains unwritten.
    ///
    /// Prepares source and destination regions, waits for producers on other streams, rejects
    /// stream capture, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, view construction, native conversion, or stream synchronization fail.
    /// Source and destination dimensions must match.
    fn convert_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, U, L>,
    ) -> Result<()>;
}

macro_rules! impl_convert_image {
    ($source_ty:ty, $destination_ty:ty, $layout:ty, $physical:ty, $convert:path) => {
        unsafe impl<'a> ConvertImage<$source_ty, $destination_ty, $layout>
            for ImagePipeline<'a, $source_ty, $layout>
        {
            fn convert_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, $layout>,
                destination: &mut ImageOutputView<'_, $destination_ty, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context.sync_scope(|| {
                    let source = source.with_layout::<$physical>()?;
                    let mut destination = destination.output().with_layout::<$physical>()?;
                    // SAFETY: native dispatch writes every physical channel and the scope retains
                    // every borrowed participant through completion.
                    unsafe { $convert(stream_context, &source, &mut destination) }
                })
            }
        }
    };
}

impl_convert_image!(i8, u8, C1, C1, exchange::convert_i8_to_u8_c1);

impl_convert_image!(i8, u16, C1, C1, exchange::convert_i8_to_u16_c1);

impl_convert_image!(i8, i16, C1, C1, exchange::convert_i8_to_i16_c1);

impl_convert_image!(i8, u32, C1, C1, exchange::convert_i8_to_u32_c1);

impl_convert_image!(i8, i32, C1, C1, exchange::convert_i8_to_i32_c1);

impl_convert_image!(i8, i32, C3, C3, exchange::convert_i8_to_i32_c3);

impl_convert_image!(i8, i32, C4, C4, exchange::convert_i8_to_i32_c4);

impl_convert_image!(i8, i32, AC4, C4, exchange::convert_i8_to_i32_c4);

impl_convert_image!(i8, f32, C1, C1, exchange::convert_i8_to_f32_c1);

impl_convert_image!(i8, f32, C3, C3, exchange::convert_i8_to_f32_c3);

impl_convert_image!(i8, f32, C4, C4, exchange::convert_i8_to_f32_c4);

impl_convert_image!(i8, f32, AC4, C4, exchange::convert_i8_to_f32_c4);

impl_convert_image!(i32, i8, C1, C1, exchange::convert_i32_to_i8_c1);

impl_convert_image!(i32, i8, C3, C3, exchange::convert_i32_to_i8_c3);

impl_convert_image!(i32, i8, C4, C4, exchange::convert_i32_to_i8_c4);

impl_convert_image!(i32, i8, AC4, C4, exchange::convert_i32_to_i8_c4);

impl_convert_image!(f16, f32, C1, C1, exchange::convert_f16_to_f32_c1);

impl_convert_image!(f16, f32, C3, C3, exchange::convert_f16_to_f32_c3);

impl_convert_image!(f16, f32, C4, C4, exchange::convert_f16_to_f32_c4);

impl_convert_image!(f16, f32, AC4, C4, exchange::convert_f16_to_f32_c4);

impl_convert_image!(u8, f32, C1, C1, exchange::convert_to_f32_c1);

impl_convert_image!(u8, f32, C3, C3, exchange::convert_to_f32_c3);

impl_convert_image!(u8, f32, C4, C4, exchange::convert_to_f32_c4);

impl_convert_image!(u8, f32, AC4, C4, exchange::convert_to_f32_c4);

impl_convert_image!(u8, u16, C1, C1, exchange::convert_u8_to_u16_c1);

impl_convert_image!(u8, u16, C3, C3, exchange::convert_u8_to_u16_c3);

impl_convert_image!(u8, u16, C4, C4, exchange::convert_u8_to_u16_c4);

impl_convert_image!(u8, u16, AC4, C4, exchange::convert_u8_to_u16_c4);

impl_convert_image!(u8, i16, C1, C1, exchange::convert_u8_to_i16_c1);

impl_convert_image!(u8, i16, C3, C3, exchange::convert_u8_to_i16_c3);

impl_convert_image!(u8, i16, C4, C4, exchange::convert_u8_to_i16_c4);

impl_convert_image!(u8, i16, AC4, C4, exchange::convert_u8_to_i16_c4);

impl_convert_image!(u8, i32, C1, C1, exchange::convert_u8_to_i32_c1);

impl_convert_image!(u8, i32, C3, C3, exchange::convert_u8_to_i32_c3);

impl_convert_image!(u8, i32, C4, C4, exchange::convert_u8_to_i32_c4);

impl_convert_image!(u8, i32, AC4, C4, exchange::convert_u8_to_i32_c4);

impl_convert_image!(u16, u8, C1, C1, exchange::convert_u16_to_u8_c1);

impl_convert_image!(u16, u8, C3, C3, exchange::convert_u16_to_u8_c3);

impl_convert_image!(u16, u8, C4, C4, exchange::convert_u16_to_u8_c4);

impl_convert_image!(u16, u8, AC4, C4, exchange::convert_u16_to_u8_c4);

impl_convert_image!(u16, i32, C1, C1, exchange::convert_u16_to_i32_c1);

impl_convert_image!(u16, i32, C3, C3, exchange::convert_u16_to_i32_c3);

impl_convert_image!(u16, i32, C4, C4, exchange::convert_u16_to_i32_c4);

impl_convert_image!(u16, i32, AC4, C4, exchange::convert_u16_to_i32_c4);

impl_convert_image!(u16, f32, C1, C1, exchange::convert_u16_to_f32_c1);

impl_convert_image!(u16, f32, C3, C3, exchange::convert_u16_to_f32_c3);

impl_convert_image!(u16, f32, C4, C4, exchange::convert_u16_to_f32_c4);

impl_convert_image!(u16, f32, AC4, C4, exchange::convert_u16_to_f32_c4);

impl_convert_image!(i16, u8, C1, C1, exchange::convert_i16_to_u8_c1);

impl_convert_image!(i16, u8, C3, C3, exchange::convert_i16_to_u8_c3);

impl_convert_image!(i16, u8, C4, C4, exchange::convert_i16_to_u8_c4);

impl_convert_image!(i16, u8, AC4, C4, exchange::convert_i16_to_u8_c4);

impl_convert_image!(i16, u16, C1, C1, exchange::convert_i16_to_u16_c1);

impl_convert_image!(i16, i32, C1, C1, exchange::convert_i16_to_i32_c1);

impl_convert_image!(i16, i32, C3, C3, exchange::convert_i16_to_i32_c3);

impl_convert_image!(i16, i32, C4, C4, exchange::convert_i16_to_i32_c4);

impl_convert_image!(i16, i32, AC4, C4, exchange::convert_i16_to_i32_c4);

impl_convert_image!(i16, u32, C1, C1, exchange::convert_i16_to_u32_c1);

impl_convert_image!(i16, f32, C1, C1, exchange::convert_i16_to_f32_c1);

impl_convert_image!(i16, f32, C3, C3, exchange::convert_i16_to_f32_c3);

impl_convert_image!(i16, f32, C4, C4, exchange::convert_i16_to_f32_c4);

impl_convert_image!(i16, f32, AC4, C4, exchange::convert_i16_to_f32_c4);

impl_convert_image!(i32, u8, C1, C1, exchange::convert_i32_to_u8_c1);

impl_convert_image!(i32, u8, C3, C3, exchange::convert_i32_to_u8_c3);

impl_convert_image!(i32, u8, C4, C4, exchange::convert_i32_to_u8_c4);

impl_convert_image!(i32, u8, AC4, C4, exchange::convert_i32_to_u8_c4);

impl_convert_image!(i32, u32, C1, C1, exchange::convert_i32_to_u32_c1);

impl_convert_image!(i32, f32, C1, C1, exchange::convert_i32_to_f32_c1);

impl_convert_image!(u16, u32, C1, C1, exchange::convert_u16_to_u32_c1);

impl_convert_image!(u32, f32, C1, C1, exchange::convert_u32_to_f32_c1);
