use crate::{
    context::StreamContext,
    error::Result,
    image::{
        statistics,
        view::{C1, ImageOutputView, ImageView},
    },
    pipeline::image::ImagePipeline,
};

impl ImagePipeline<'_, u8, C1> {
    /// Computes a C1 integral image and an integral of squared source samples.
    /// At output coordinate `(x, y)`, the integral sums source samples in columns before
    /// `x` and rows before `y`, then adds `value`. This includes a top row and left column
    /// filled with `value`. Each output is one pixel wider and taller than the source.
    /// The squared output sums squared samples over the same prefixes and adds
    /// `squared_value`, which also fills its top and left borders.
    /// Produces `i32` integral samples and `i32` squared-integral samples.
    /// Both supplied destinations must have the complete expanded dimensions.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Writes every logical output
    /// pixel, including the borders; row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail. Destination width and height
    /// must each equal the corresponding source dimension plus one; those additions are checked
    /// for overflow.
    /// Both outputs must have matching expanded dimensions.
    pub fn squared_integral_to_i32_i32_into(
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

    /// Computes a C1 integral image and an integral of squared source samples.
    /// At output coordinate `(x, y)`, the integral sums source samples in columns before
    /// `x` and rows before `y`, then adds `value`. This includes a top row and left column
    /// filled with `value`. Each output is one pixel wider and taller than the source.
    /// The squared output sums squared samples over the same prefixes and adds
    /// `squared_value`, which also fills its top and left borders.
    /// Produces `i32` integral samples and `f64` squared-integral samples.
    /// Both supplied destinations must have the complete expanded dimensions.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Writes every logical output
    /// pixel, including the borders; row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail. Destination width and height
    /// must each equal the corresponding source dimension plus one; those additions are checked
    /// for overflow.
    /// Both outputs must have matching expanded dimensions.
    pub fn squared_integral_to_i32_f64_into(
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

    /// Computes a C1 integral image and an integral of squared source samples.
    /// At output coordinate `(x, y)`, the integral sums source samples in columns before
    /// `x` and rows before `y`, then adds `value`. This includes a top row and left column
    /// filled with `value`. Each output is one pixel wider and taller than the source.
    /// The squared output sums squared samples over the same prefixes and adds
    /// `squared_value`, which also fills its top and left borders.
    /// Produces `f32` integral samples and `f64` squared-integral samples.
    /// Both supplied destinations must have the complete expanded dimensions.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Writes every logical output
    /// pixel, including the borders; row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail. Destination width and height
    /// must each equal the corresponding source dimension plus one; those additions are checked
    /// for overflow.
    /// Both outputs must have matching expanded dimensions.
    pub fn squared_integral_to_f32_f64_into(
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
