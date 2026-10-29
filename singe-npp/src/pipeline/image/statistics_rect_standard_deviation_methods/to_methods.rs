use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage, statistics,
        view::{C1, ImageOutputView, ImageView},
    },
    pipeline::image::{
        ImageBacking,
        statistics_rect_standard_deviation_methods::{ImagePipeline, rect_standard_deviation_size},
    },
    types::Rectangle,
};

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
{
    /// Computes sliding rectangle standard deviations from matching integral images.
    ///
    /// The pipeline `source` is an integral table, and `squared` is the integral of squared
    /// samples for the same underlying image. `rectangle` locates each sampling window
    /// relative to its destination pixel; its origin must be nonnegative and its dimensions
    /// must be positive. The wrapper checks geometry without inspecting table contents.
    /// The provided implementations accept a C1 `f32` or `i32` integral table with a C1
    /// `f64` squared-integral table and produce C1 `f32` results.
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
    pub fn rect_standard_deviation_to_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        squared: &ImageView<'_, T::Squared, C1>,
        destination: &mut ImageOutputView<'_, T::Destination, C1>,
        rectangle: Rectangle,
    ) -> Result<()>
    where
        T: statistics::RectStandardDeviation,
    {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(squared)?;
        stream_context.prepare_region(destination)?;
        stream_context.sync_scope(|| {
            // SAFETY: native ABI dispatch runs with all borrowed regions retained through completion.
            unsafe {
                statistics::rect_standard_deviation(
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
    /// The provided implementations accept a C1 `f32` or `i32` integral table with a C1
    /// `f64` squared-integral table and produce C1 `f32` results.
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
    pub fn rect_standard_deviation_to(
        self,
        squared: &ImageView<'_, T::Squared, C1>,
        rectangle: Rectangle,
    ) -> Result<ImagePipeline<'a, T::Destination, C1>>
    where
        T: statistics::RectStandardDeviation,
        T::Destination: SupportedImage<C1>,
    {
        self.ensure_eager()?;
        let size = rect_standard_deviation_size(self.size(), rectangle)?;
        let mut destination =
            Image::<T::Destination, C1>::create_uninit(self.stream_context.cuda_context(), size)?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            self.stream_context.stream().ensure_not_capturing()?;
            self.stream_context.prepare_region(&source)?;
            self.stream_context.prepare_region(squared)?;
            self.stream_context.prepare_region(&destination_view)?;
            self.stream_context.sync_scope(|| {
                // SAFETY: native ABI dispatch runs with all borrowed regions retained through completion.
                unsafe {
                    statistics::rect_standard_deviation(
                        self.stream_context,
                        &source,
                        squared,
                        &mut destination_view,
                        rectangle,
                    )
                }
            })?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}
