use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::{
        SupportedImage,
        memory::Image,
        statistics,
        view::{C1, ImageOutputView, ImageView},
    },
    pipeline::image::{ImageBacking, ImagePipeline},
    types::{Rectangle, Size},
};

mod i32_methods;
mod scaled_methods;
mod to_methods;

impl ImagePipeline<'_, f32, C1>
where
    f32: SupportedImage<C1>,
{
    /// Computes sliding rectangle standard deviations from matching integral images.
    ///
    /// The pipeline `source` is an integral table, and `squared` is the integral of squared
    /// samples for the same underlying image. `rectangle` locates each sampling window
    /// relative to its destination pixel; its origin must be nonnegative and its dimensions
    /// must be positive. The wrapper checks geometry without inspecting table contents.
    /// Uses a C1 `f32` integral table and a C1 `f64` squared-integral table,
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
    pub fn rect_standard_deviation_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, f32, C1>,
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
    /// Uses a C1 `f32` integral table and a C1 `f64` squared-integral table,
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
    pub fn rect_standard_deviation(
        mut self,
        squared: &ImageView<'_, f64, C1>,
        rectangle: Rectangle,
    ) -> Result<Self> {
        self.ensure_eager()?;
        let destination = self.rect_standard_deviation_with(
            squared,
            rectangle,
            statistics::rect_standard_deviation,
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
    pub(crate) fn rect_standard_deviation_with<S, D>(
        &mut self,
        squared: &ImageView<'_, S, C1>,
        rectangle: Rectangle,
        operation: RectStandardDeviationWithCallback<D, S, T>,
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
                    )
                }
            })?;
        }

        // SAFETY: completed native dispatch initialized the complete validated output rectangle.
        Ok(unsafe { destination.assume_init() })
    }
}

pub fn rect_standard_deviation_size(source: Size, rectangle: Rectangle) -> Result<Size> {
    if rectangle.x < 0 || rectangle.y < 0 || rectangle.width <= 0 || rectangle.height <= 0 {
        return Err(Error::out_of_range("rectangle"));
    }

    Ok(Size {
        width: source
            .width
            .checked_sub(rectangle.x)
            .and_then(|value| value.checked_sub(rectangle.width))
            .ok_or_else(|| Error::out_of_range("rect stddev destination width"))?,
        height: source
            .height
            .checked_sub(rectangle.y)
            .and_then(|value| value.checked_sub(rectangle.height))
            .ok_or_else(|| Error::out_of_range("rect stddev destination height"))?,
    })
}

type RectStandardDeviationWithCallback<D, S, T> = unsafe fn(
    &StreamContext,
    &ImageView<'_, T, C1>,
    &ImageView<'_, S, C1>,
    &mut ImageOutputView<'_, D, C1>,
    Rectangle,
) -> Result<()>;
