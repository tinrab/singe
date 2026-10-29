use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::{
        SupportedImage,
        memory::Image,
        view::{C1, ImageOutputView, ImageView},
    },
    pipeline::{ImageBacking, image::ImagePipeline},
    types::Size,
};

mod concrete_methods;
mod integral_dispatch;
mod squared_methods;

use integral_dispatch::{IntegralImage, SquaredIntegralImage};

impl<'a> ImagePipeline<'a, u8, C1> {
    /// Computes a C1 integral image from `u8` samples.
    /// At output coordinate `(x, y)`, the integral sums source samples in columns before
    /// `x` and rows before `y`, then adds `value`. This includes a top row and left column
    /// filled with `value`. Each output is one pixel wider and taller than the source.
    /// The provided dispatch supports `T = i32` and `T = f32`.
    /// The supplied destination must have the complete expanded dimensions.
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
    pub fn integral_to_into<T>(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C1>,
        destination: &mut ImageOutputView<'_, T, C1>,
        value: T,
    ) -> Result<()>
    where
        T: Copy,
        Self: IntegralImage<T>,
    {
        <Self as IntegralImage<T>>::integral_image(stream_context, source, destination, value)
    }

    /// Computes a C1 integral image from `u8` samples.
    /// At output coordinate `(x, y)`, the integral sums source samples in columns before
    /// `x` and rows before `y`, then adds `value`. This includes a top row and left column
    /// filled with `value`. Each output is one pixel wider and taller than the source.
    /// The provided dispatch supports `T = i32` and `T = f32`.
    /// Allocates a new owned C1 image with the expanded dimensions.
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
    /// Output allocation or image view construction can also fail.
    pub fn integral_to<T>(self, value: T) -> Result<ImagePipeline<'a, T, C1>>
    where
        T: Copy + SupportedImage<C1>,
        Self: IntegralImage<T>,
    {
        self.ensure_eager()?;
        let size = integral_size(self.size())?;
        let mut destination =
            Image::<T, C1>::create_uninit(self.stream_context.cuda_context(), size)?;
        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            <Self as IntegralImage<T>>::integral_image(
                self.stream_context,
                &source,
                &mut destination_view,
                value,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }

    pub(crate) fn integral<T>(
        &mut self,
        value: T,
        operation: IntegralCallback<T>,
    ) -> Result<Image<T, C1>>
    where
        T: Copy + SupportedImage<C1>,
    {
        self.ensure_eager()?;
        let size = integral_size(self.size())?;
        let mut destination =
            Image::<T, C1>::create_uninit(self.stream_context.cuda_context(), size)?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            self.stream_context.stream().ensure_not_capturing()?;
            self.stream_context.prepare_region(&source)?;
            self.stream_context.prepare_region(&destination_view)?;
            self.stream_context.sync_scope(|| {
                // SAFETY: native ABI dispatch runs with all borrowed regions retained through completion.
                unsafe { operation(self.stream_context, &source, &mut destination_view, value) }
            })?;
        }

        // SAFETY: completed native dispatch initialized the complete validated output rectangle.
        Ok(unsafe { destination.assume_init() })
    }
}

pub fn integral_size(source: Size) -> Result<Size> {
    Ok(Size {
        width: source
            .width
            .checked_add(1)
            .ok_or_else(|| Error::out_of_range("integral destination width"))?,
        height: source
            .height
            .checked_add(1)
            .ok_or_else(|| Error::out_of_range("integral destination height"))?,
    })
}

type IntegralCallback<T> = unsafe fn(
    &StreamContext,
    &ImageView<'_, u8, C1>,
    &mut ImageOutputView<'_, T, C1>,
    T,
) -> Result<()>;
