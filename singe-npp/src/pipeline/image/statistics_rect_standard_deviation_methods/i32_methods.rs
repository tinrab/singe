use crate::{
    context::StreamContext,
    error::Result,
    image::{
        SupportedImage, statistics,
        view::{C1, ImageOutputView, ImageView},
    },
    pipeline::image::statistics_rect_standard_deviation_methods::{ImageBacking, ImagePipeline},
    types::Rectangle,
};

impl<'a> ImagePipeline<'a, i32, C1>
where
    f32: SupportedImage<C1>,
{
    /// Computes sliding rectangle standard deviations from matching integral images.
    ///
    /// The pipeline `source` is an integral table, and `squared` is the integral of squared
    /// samples for the same underlying image. `rectangle` locates each sampling window
    /// relative to its destination pixel; its origin must be nonnegative and its dimensions
    /// must be positive. The wrapper checks geometry without inspecting table contents.
    /// Uses a C1 `i32` integral table and a C1 `f64` squared-integral table,
    /// producing C1 `f32` results.
    /// Both input sizes must exactly equal the destination size plus
    /// `(rectangle.x + rectangle.width, rectangle.y + rectangle.height)`.
    /// Prepares all regions, waits for producers on other streams, rejects capture, and
    /// returns after successful stream completion. Writes every logical output sample;
    /// row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native statistics execution, or synchronization fail. Rectangle
    /// coordinates must be nonnegative and dimensions positive. Checked geometry arithmetic
    /// or an input size that differs from the exact required extent also causes an error.
    pub fn rect_standard_deviation_to_f32_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, i32, C1>,
        squared: &ImageView<'_, f64, C1>,
        destination: &mut ImageOutputView<'_, f32, C1>,
        rectangle: Rectangle,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(squared)?;
        stream_context.prepare_region(destination)?;
        stream_context.sync_scope(|| {
            // SAFETY: native ABI dispatch runs with all borrowed regions retained through completion.
            unsafe {
                statistics::rect_standard_deviation_to(
                    stream_context,
                    source,
                    squared,
                    destination,
                    rectangle,
                )
            }
        })
    }

    /// Computes sliding rectangle standard deviations from matching integral images.
    ///
    /// The pipeline source is an integral table, and `squared` is the integral of squared
    /// samples for the same underlying image. `rectangle` locates each sampling window
    /// relative to its destination pixel; its origin must be nonnegative and its dimensions
    /// must be positive. The wrapper checks geometry without inspecting table contents.
    /// Uses a C1 `i32` integral table and a C1 `f64` squared-integral table,
    /// producing C1 `f32` results.
    /// Allocates a new owned image whose width is integral width minus `rectangle.x` and
    /// `rectangle.width`, and whose height is integral height minus `rectangle.y` and
    /// `rectangle.height`. The squared-integral image must have the same size as the
    /// pipeline integral table.
    /// Prepares all regions, waits for producers on other streams, rejects capture, and
    /// returns after successful stream completion. Writes every logical output sample;
    /// row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native statistics execution, or synchronization fail. Rectangle
    /// coordinates must be nonnegative and dimensions positive. Checked geometry arithmetic
    /// or an input size that differs from the exact required extent also causes an error.
    /// Output allocation or image view construction can also fail.
    pub fn rect_standard_deviation_to_f32(
        mut self,
        squared: &ImageView<'_, f64, C1>,
        rectangle: Rectangle,
    ) -> Result<ImagePipeline<'a, f32, C1>> {
        self.ensure_eager()?;
        let destination = self.rect_standard_deviation_with(
            squared,
            rectangle,
            statistics::rect_standard_deviation_to,
        )?;
        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
