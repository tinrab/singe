use crate::{
    context::StreamContext,
    error::Result,
    image::{
        SupportedImage,
        memory::Image,
        statistics,
        view::{C1, ImageOutputView, ImageView},
    },
    pipeline::image::statistics_rect_standard_deviation_methods::{
        ImageBacking, ImagePipeline, rect_standard_deviation_size,
    },
    types::Rectangle,
};

impl ImagePipeline<'_, i32, C1>
where
    i32: SupportedImage<C1>,
{
    /// Computes sliding rectangle standard deviations from matching integral images.
    ///
    /// The pipeline `source` is an integral table, and `squared` is the integral of squared
    /// samples for the same underlying image. `rectangle` locates each sampling window
    /// relative to its destination pixel; its origin must be nonnegative and its dimensions
    /// must be positive. The wrapper checks geometry without inspecting table contents.
    /// Uses C1 `i32` integral and squared-integral tables and produces C1 `i32` results.
    /// `scale_factor` controls native integer output scaling and saturation.
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
    pub fn rect_standard_deviation_scaled_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, i32, C1>,
        squared: &ImageView<'_, i32, C1>,
        destination: &mut ImageOutputView<'_, i32, C1>,
        rectangle: Rectangle,
        scale_factor: i32,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(squared)?;
        stream_context.prepare_region(destination)?;
        stream_context.sync_scope(|| {
            // SAFETY: native ABI dispatch runs with all borrowed regions retained through completion.
            unsafe {
                statistics::rect_standard_deviation_scaled(
                    stream_context,
                    source,
                    squared,
                    destination,
                    rectangle,
                    scale_factor,
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
    /// Uses C1 `i32` integral and squared-integral tables and produces C1 `i32` results.
    /// `scale_factor` controls native integer output scaling and saturation.
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
    pub fn rect_standard_deviation_scaled(
        mut self,
        squared: &ImageView<'_, i32, C1>,
        rectangle: Rectangle,
        scale_factor: i32,
    ) -> Result<Self> {
        self.ensure_eager()?;
        let destination = self.rect_standard_deviation_with_scale(
            squared,
            rectangle,
            scale_factor,
            statistics::rect_standard_deviation_scaled,
        )?;
        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}

impl<T> ImagePipeline<'_, T, C1>
where
    T: Copy,
{
    fn rect_standard_deviation_with_scale<S, D>(
        &mut self,
        squared: &ImageView<'_, S, C1>,
        rectangle: Rectangle,
        scale_factor: i32,
        operation: RectStandardDeviationWithScaleCallback<D, S, T>,
    ) -> Result<Image<D, C1>>
    where
        S: Copy,
        D: Copy + SupportedImage<C1>,
    {
        self.ensure_eager()?;
        let size = rect_standard_deviation_size(self.size(), rectangle)?;
        let mut destination =
            Image::<D, C1>::create_uninit(self.stream_context.cuda_context(), size)?;

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
                    operation(
                        self.stream_context,
                        &source,
                        squared,
                        &mut destination_view,
                        rectangle,
                        scale_factor,
                    )
                }
            })?;
        }

        // SAFETY: completed rectangle-statistics dispatch writes the checked destination ROI.
        Ok(unsafe { destination.assume_init() })
    }
}

type RectStandardDeviationWithScaleCallback<D, S, T> = unsafe fn(
    &StreamContext,
    &ImageView<'_, T, C1>,
    &ImageView<'_, S, C1>,
    &mut ImageOutputView<'_, D, C1>,
    Rectangle,
    i32,
) -> Result<()>;
