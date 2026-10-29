use singe_cuda::view::DeviceRepr;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageOutputView, ImageView},
    pipeline::image::{
        statistics::{ContiguousImage, ImageSquaredIntegral},
        statistics_integral_methods::{ImagePipeline, SquaredIntegralImage, integral_size},
    },
};

impl ImagePipeline<'_, u8, C1> {
    /// Computes a C1 integral image and an integral of squared source samples.
    /// At output coordinate `(x, y)`, the integral sums source samples in columns before
    /// `x` and rows before `y`, then adds `value`. This includes a top row and left column
    /// filled with `value`. Each output is one pixel wider and taller than the source.
    /// The squared output sums squared samples over the same prefixes and adds
    /// `squared_value`, which also fills its top and left borders.
    /// The provided dispatch supports `(I, S)` pairs `(i32, i32)`, `(i32, f64)`, and
    /// `(f32, f64)`.
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
    pub fn squared_integral_to_into<I, S>(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C1>,
        integral: &mut ImageOutputView<'_, I, C1>,
        squared: &mut ImageOutputView<'_, S, C1>,
        value: I,
        squared_value: S,
    ) -> Result<()>
    where
        I: DeviceRepr,
        S: DeviceRepr,
        Self: SquaredIntegralImage<I, S>,
    {
        <Self as SquaredIntegralImage<I, S>>::squared_integral_image(
            stream_context,
            source,
            integral,
            squared,
            value,
            squared_value,
        )
    }

    /// Computes a C1 integral image and an integral of squared source samples.
    /// At output coordinate `(x, y)`, the integral sums source samples in columns before
    /// `x` and rows before `y`, then adds `value`. This includes a top row and left column
    /// filled with `value`. Each output is one pixel wider and taller than the source.
    /// The squared output sums squared samples over the same prefixes and adds
    /// `squared_value`, which also fills its top and left borders.
    /// The provided dispatch supports `(I, S)` pairs `(i32, i32)`, `(i32, f64)`, and
    /// `(f32, f64)`.
    /// Allocates two owned contiguous device images for the expanded outputs.
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
    /// Output allocation or image view construction can also fail.
    pub fn squared_integral_to<I, S>(
        self,
        value: I,
        squared_value: S,
    ) -> Result<ImageSquaredIntegral<I, S>>
    where
        I: DeviceRepr,
        S: DeviceRepr,
        Self: SquaredIntegralImage<I, S>,
    {
        self.ensure_eager()?;
        let size = integral_size(self.size())?;
        let mut integral =
            ContiguousImage::<I, C1>::create_uninit(self.stream_context.cuda_context(), size)?;
        let mut squared =
            ContiguousImage::<S, C1>::create_uninit(self.stream_context.cuda_context(), size)?;

        {
            let source = self.view()?;
            let mut integral_view = integral.output_view()?;
            let mut squared_view = squared.output_view()?;
            <Self as SquaredIntegralImage<I, S>>::squared_integral_image(
                self.stream_context,
                &source,
                &mut integral_view,
                &mut squared_view,
                value,
                squared_value,
            )?;
        }

        // SAFETY: the integral producer writes both complete images, including their border.
        Ok(ImageSquaredIntegral {
            integral: unsafe { integral.assume_init() },
            squared: unsafe { squared.assume_init() },
        })
    }
}
