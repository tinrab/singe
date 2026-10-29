use crate::{
    context::StreamContext,
    error::Result,
    image::{
        statistics,
        view::{C1, ImageOutputView, ImageView},
    },
    pipeline::image::statistics_integral_methods::ImagePipeline,
};

/// Computes a complete integral image before returning.
///
/// # Safety
/// Match the native ABI, initialize every logical output pixel including borders,
/// and drain borrowed work on success, errors and unwind. Reject capture before work.
pub unsafe trait IntegralImage<U> {
    fn integral_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C1>,
        destination: &mut ImageOutputView<'_, U, C1>,
        value: U,
    ) -> Result<()>;
}

unsafe impl IntegralImage<i32> for ImagePipeline<'_, u8, C1> {
    fn integral_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C1>,
        destination: &mut ImageOutputView<'_, i32, C1>,
        value: i32,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        stream_context.sync_scope(|| {
            // SAFETY: native ABI dispatch runs with all borrowed regions retained through completion.
            unsafe { statistics::integral_to_i32(stream_context, source, destination, value) }
        })
    }
}

unsafe impl IntegralImage<f32> for ImagePipeline<'_, u8, C1> {
    fn integral_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C1>,
        destination: &mut ImageOutputView<'_, f32, C1>,
        value: f32,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        stream_context.sync_scope(|| {
            // SAFETY: native ABI dispatch runs with all borrowed regions retained through completion.
            unsafe { statistics::integral_to_f32(stream_context, source, destination, value) }
        })
    }
}

/// Produces complete integral and squared-integral images.
///
/// # Safety
///
/// Successful return establishes valid values in every destination pixel,
/// including borders, without reading prior destination values. Drain borrowed
/// work on success, errors and unwind; reject capture before submission.
pub unsafe trait SquaredIntegralImage<I, S> {
    fn squared_integral_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C1>,
        integral: &mut ImageOutputView<'_, I, C1>,
        squared: &mut ImageOutputView<'_, S, C1>,
        value: I,
        squared_value: S,
    ) -> Result<()>;
}

unsafe impl SquaredIntegralImage<i32, i32> for ImagePipeline<'_, u8, C1> {
    fn squared_integral_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C1>,
        integral: &mut ImageOutputView<'_, i32, C1>,
        squared: &mut ImageOutputView<'_, i32, C1>,
        value: i32,
        squared_value: i32,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(integral)?;
        stream_context.prepare_region(squared)?;
        stream_context.sync_scope(|| {
            // SAFETY: native ABI dispatch runs with all borrowed regions retained through completion.
            unsafe {
                statistics::squared_integral_to_i32_i32(
                    stream_context,
                    source,
                    integral,
                    squared,
                    value,
                    squared_value,
                )
            }
        })
    }
}

unsafe impl SquaredIntegralImage<i32, f64> for ImagePipeline<'_, u8, C1> {
    fn squared_integral_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C1>,
        integral: &mut ImageOutputView<'_, i32, C1>,
        squared: &mut ImageOutputView<'_, f64, C1>,
        value: i32,
        squared_value: f64,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(integral)?;
        stream_context.prepare_region(squared)?;
        stream_context.sync_scope(|| {
            // SAFETY: native ABI dispatch runs with all borrowed regions retained through completion.
            unsafe {
                statistics::squared_integral_to_i32_f64(
                    stream_context,
                    source,
                    integral,
                    squared,
                    value,
                    squared_value,
                )
            }
        })
    }
}

unsafe impl SquaredIntegralImage<f32, f64> for ImagePipeline<'_, u8, C1> {
    fn squared_integral_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C1>,
        integral: &mut ImageOutputView<'_, f32, C1>,
        squared: &mut ImageOutputView<'_, f64, C1>,
        value: f32,
        squared_value: f64,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(integral)?;
        stream_context.prepare_region(squared)?;
        stream_context.sync_scope(|| {
            // SAFETY: native ABI dispatch runs with all borrowed regions retained through completion.
            unsafe {
                statistics::squared_integral_to_f32_f64(
                    stream_context,
                    source,
                    integral,
                    squared,
                    value,
                    squared_value,
                )
            }
        })
    }
}
