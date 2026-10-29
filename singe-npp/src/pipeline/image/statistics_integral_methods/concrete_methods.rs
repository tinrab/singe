use crate::{
    context::StreamContext,
    error::Result,
    image::{
        SupportedImage, statistics,
        view::{C1, ImageOutputView, ImageView},
    },
    pipeline::image::{ImageBacking, ImagePipeline},
};

impl<'a> ImagePipeline<'a, u8, C1>
where
    i32: SupportedImage<C1>,
{
    /// Computes a C1 integral image from `u8` samples.
    /// At output coordinate `(x, y)`, the integral sums source samples in columns before
    /// `x` and rows before `y`, then adds `value`. This includes a top row and left column
    /// filled with `value`. Each output is one pixel wider and taller than the source.
    /// Produces `i32` integral samples.
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
    pub fn integral_to_i32_into(
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

    /// Computes a C1 integral image from `u8` samples.
    /// At output coordinate `(x, y)`, the integral sums source samples in columns before
    /// `x` and rows before `y`, then adds `value`. This includes a top row and left column
    /// filled with `value`. Each output is one pixel wider and taller than the source.
    /// Produces `i32` integral samples.
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
    pub fn integral_to_i32(mut self, value: i32) -> Result<ImagePipeline<'a, i32, C1>> {
        self.ensure_eager()?;
        let destination = self.integral(value, statistics::integral_to_i32)?;

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}

impl<'a> ImagePipeline<'a, u8, C1>
where
    f32: SupportedImage<C1>,
{
    /// Computes a C1 integral image from `u8` samples.
    /// At output coordinate `(x, y)`, the integral sums source samples in columns before
    /// `x` and rows before `y`, then adds `value`. This includes a top row and left column
    /// filled with `value`. Each output is one pixel wider and taller than the source.
    /// Produces `f32` integral samples.
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
    pub fn integral_to_f32_into(
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

    /// Computes a C1 integral image from `u8` samples.
    /// At output coordinate `(x, y)`, the integral sums source samples in columns before
    /// `x` and rows before `y`, then adds `value`. This includes a top row and left column
    /// filled with `value`. Each output is one pixel wider and taller than the source.
    /// Produces `f32` integral samples.
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
    pub fn integral_to_f32(mut self, value: f32) -> Result<ImagePipeline<'a, f32, C1>> {
        self.ensure_eager()?;
        let destination = self.integral(value, statistics::integral_to_f32)?;

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
