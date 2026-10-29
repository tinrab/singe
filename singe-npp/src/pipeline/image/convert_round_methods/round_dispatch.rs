use singe_cuda::types::f16;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        exchange,
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView},
    },
    pipeline::image::convert_round_methods::ImagePipeline,
    types::RoundMode,
};

/// Dispatches completed image element conversions with rounding.
///
/// # Safety
///
/// Implementations must preserve native source and destination representations and numeric
/// conversion semantics. Complete borrowed native work on every path. On success, initialize
/// every physical channel in the ROI, including alpha. Uphold the completed native dispatch
/// contract of [`ImagePipeline`](crate::pipeline::ImagePipeline).
pub unsafe trait RoundConvertImage<T, U, L> {
    /// Converts each image sample to `U` using native conversion semantics.
    /// Uses `round_mode` for the selected native rounding operation.
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
    fn round_convert_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, U, L>,
        round_mode: RoundMode,
    ) -> Result<()>;
}

/// Dispatches completed image element conversions with rounding and scaling.
///
/// # Safety
///
/// Implementations must preserve native source and destination representations and numeric
/// conversion semantics. Complete borrowed native work on every path. On success, initialize
/// every physical channel in the ROI, including alpha. Uphold the completed native dispatch
/// contract of [`ImagePipeline`](crate::pipeline::ImagePipeline).
pub unsafe trait ScaledRoundConvertImage<T, U, L> {
    /// Converts each image sample to `U` using native conversion semantics.
    /// Uses `round_mode` for the selected native rounding operation.
    /// Passes `scale_factor` to the native scaled conversion without validating its numeric range.
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
    fn scaled_round_convert_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, U, L>,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<()>;
}

macro_rules! impl_round_convert_image {
    ($source_ty:ty, $destination_ty:ty, $layout:ty, $physical:ty, $convert:path) => {
        unsafe impl<'a> RoundConvertImage<$source_ty, $destination_ty, $layout>
            for ImagePipeline<'a, $source_ty, $layout>
        {
            fn round_convert_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, $layout>,
                destination: &mut ImageOutputView<'_, $destination_ty, $layout>,
                round_mode: RoundMode,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context.sync_scope(|| {
                    let source = source.with_layout::<$physical>()?;
                    let mut destination = destination.output().with_layout::<$physical>()?;
                    // SAFETY: native dispatch writes every physical channel and the scope retains
                    // every borrowed participant through completion.
                    unsafe { $convert(stream_context, &source, &mut destination, round_mode) }
                })
            }
        }
    };
}

macro_rules! impl_scaled_round_convert_image {
    ($source_ty:ty, $destination_ty:ty, $layout:ty, $physical:ty, $convert:path) => {
        unsafe impl<'a> ScaledRoundConvertImage<$source_ty, $destination_ty, $layout>
            for ImagePipeline<'a, $source_ty, $layout>
        {
            fn scaled_round_convert_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, $layout>,
                destination: &mut ImageOutputView<'_, $destination_ty, $layout>,
                round_mode: RoundMode,
                scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context.sync_scope(|| {
                    let source = source.with_layout::<$physical>()?;
                    let mut destination = destination.output().with_layout::<$physical>()?;
                    // SAFETY: native dispatch writes every physical channel and the scope retains
                    // every borrowed participant through completion.
                    unsafe {
                        $convert(
                            stream_context,
                            &source,
                            &mut destination,
                            round_mode,
                            scale_factor,
                        )
                    }
                })
            }
        }
    };
}

impl_round_convert_image!(f32, u8, C1, C1, exchange::convert_f32_to_u8_c1);

impl_round_convert_image!(f32, u8, C3, C3, exchange::convert_f32_to_u8_c3);

impl_round_convert_image!(f32, u8, C4, C4, exchange::convert_f32_to_u8_c4);

impl_round_convert_image!(f32, u8, AC4, C4, exchange::convert_f32_to_u8_c4);

impl_round_convert_image!(f32, i8, C1, C1, exchange::convert_f32_to_i8_c1);

impl_round_convert_image!(f32, i8, C3, C3, exchange::convert_f32_to_i8_c3);

impl_round_convert_image!(f32, i8, C4, C4, exchange::convert_f32_to_i8_c4);

impl_round_convert_image!(f32, i8, AC4, C4, exchange::convert_f32_to_i8_c4);

impl_round_convert_image!(f32, u16, C1, C1, exchange::convert_f32_to_u16_c1);

impl_round_convert_image!(f32, u16, C3, C3, exchange::convert_f32_to_u16_c3);

impl_round_convert_image!(f32, u16, C4, C4, exchange::convert_f32_to_u16_c4);

impl_round_convert_image!(f32, u16, AC4, C4, exchange::convert_f32_to_u16_c4);

impl_round_convert_image!(f32, i16, C1, C1, exchange::convert_f32_to_i16_c1);

impl_round_convert_image!(f32, i16, C3, C3, exchange::convert_f32_to_i16_c3);

impl_round_convert_image!(f32, i16, C4, C4, exchange::convert_f32_to_i16_c4);

impl_round_convert_image!(f32, i16, AC4, C4, exchange::convert_f32_to_i16_c4);

impl_round_convert_image!(f32, f16, C1, C1, exchange::convert_f32_to_f16_c1);

impl_round_convert_image!(f32, f16, C3, C3, exchange::convert_f32_to_f16_c3);

impl_round_convert_image!(f32, f16, C4, C4, exchange::convert_f32_to_f16_c4);

impl_round_convert_image!(f32, f16, AC4, C4, exchange::convert_f32_to_f16_c4);

impl_scaled_round_convert_image!(u16, i16, C1, C1, exchange::convert_u16_to_i16_scaled_c1);

impl_scaled_round_convert_image!(u16, i8, C1, C1, exchange::convert_u16_to_i8_scaled_c1);

impl_scaled_round_convert_image!(i16, i8, C1, C1, exchange::convert_i16_to_i8_scaled_c1);

impl_scaled_round_convert_image!(u32, u8, C1, C1, exchange::convert_u32_to_u8_scaled_c1);

impl_scaled_round_convert_image!(u32, i8, C1, C1, exchange::convert_u32_to_i8_scaled_c1);

impl_scaled_round_convert_image!(u32, u16, C1, C1, exchange::convert_u32_to_u16_scaled_c1);

impl_scaled_round_convert_image!(u32, i16, C1, C1, exchange::convert_u32_to_i16_scaled_c1);

impl_scaled_round_convert_image!(u32, i32, C1, C1, exchange::convert_u32_to_i32_scaled_c1);

impl_scaled_round_convert_image!(i32, u16, C1, C1, exchange::convert_i32_to_u16_scaled_c1);

impl_scaled_round_convert_image!(i32, i16, C1, C1, exchange::convert_i32_to_i16_scaled_c1);

impl_scaled_round_convert_image!(f32, u8, C1, C1, exchange::convert_f32_to_u8_scaled_c1);

impl_scaled_round_convert_image!(f32, i8, C1, C1, exchange::convert_f32_to_i8_scaled_c1);

impl_scaled_round_convert_image!(f32, u16, C1, C1, exchange::convert_f32_to_u16_scaled_c1);

impl_scaled_round_convert_image!(f32, i16, C1, C1, exchange::convert_f32_to_i16_scaled_c1);

impl_scaled_round_convert_image!(f32, u32, C1, C1, exchange::convert_f32_to_u32_scaled_c1);

impl_scaled_round_convert_image!(f32, i32, C1, C1, exchange::convert_f32_to_i32_scaled_c1);

impl_scaled_round_convert_image!(u8, i8, C1, C1, exchange::convert_u8_to_i8_scaled_c1);
