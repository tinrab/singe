use singe_cuda::view::DeviceRepr;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        statistics,
        view::{C1, ImageOutputView, ImageView},
    },
};

mod into_methods;

use crate::pipeline::image::{
    ImagePipeline,
    statistics::{ContiguousImage, ImageSquaredIntegral},
    statistics_integral_methods::integral_size,
};

impl ImagePipeline<'_, u8, C1> {
    /// Computes a C1 integral image and an integral of squared source samples.
    /// At output coordinate `(x, y)`, the integral sums source samples in columns before
    /// `x` and rows before `y`, then adds `value`. This includes a top row and left column
    /// filled with `value`. Each output is one pixel wider and taller than the source.
    /// The squared output sums squared samples over the same prefixes and adds
    /// `squared_value`, which also fills its top and left borders.
    /// Produces `i32` integral samples and `i32` squared-integral samples.
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
    pub fn squared_integral_to_i32_i32(
        mut self,
        value: i32,
        squared_value: i32,
    ) -> Result<ImageSquaredIntegral<i32, i32>> {
        self.squared_integral(
            value,
            squared_value,
            statistics::squared_integral_to_i32_i32,
        )
    }

    /// Computes a C1 integral image and an integral of squared source samples.
    /// At output coordinate `(x, y)`, the integral sums source samples in columns before
    /// `x` and rows before `y`, then adds `value`. This includes a top row and left column
    /// filled with `value`. Each output is one pixel wider and taller than the source.
    /// The squared output sums squared samples over the same prefixes and adds
    /// `squared_value`, which also fills its top and left borders.
    /// Produces `i32` integral samples and `f64` squared-integral samples.
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
    pub fn squared_integral_to_i32_f64(
        mut self,
        value: i32,
        squared_value: f64,
    ) -> Result<ImageSquaredIntegral<i32, f64>> {
        self.squared_integral(
            value,
            squared_value,
            statistics::squared_integral_to_i32_f64,
        )
    }

    /// Computes a C1 integral image and an integral of squared source samples.
    /// At output coordinate `(x, y)`, the integral sums source samples in columns before
    /// `x` and rows before `y`, then adds `value`. This includes a top row and left column
    /// filled with `value`. Each output is one pixel wider and taller than the source.
    /// The squared output sums squared samples over the same prefixes and adds
    /// `squared_value`, which also fills its top and left borders.
    /// Produces `f32` integral samples and `f64` squared-integral samples.
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
    pub fn squared_integral_to_f32_f64(
        mut self,
        value: f32,
        squared_value: f64,
    ) -> Result<ImageSquaredIntegral<f32, f64>> {
        self.squared_integral(
            value,
            squared_value,
            statistics::squared_integral_to_f32_f64,
        )
    }

    fn squared_integral<T, S>(
        &mut self,
        value: T,
        squared_value: S,
        operation: SquaredIntegralCallback<S, T>,
    ) -> Result<ImageSquaredIntegral<T, S>>
    where
        T: DeviceRepr,
        S: DeviceRepr,
    {
        self.ensure_eager()?;
        let size = integral_size(self.size())?;
        let mut integral =
            ContiguousImage::<T, C1>::create_uninit(self.stream_context.cuda_context(), size)?;
        let mut squared =
            ContiguousImage::<S, C1>::create_uninit(self.stream_context.cuda_context(), size)?;

        {
            let source = self.view()?;
            let mut integral_view = integral.output_view()?;
            let mut squared_view = squared.output_view()?;
            self.stream_context.stream().ensure_not_capturing()?;
            self.stream_context.prepare_region(&source)?;
            self.stream_context.prepare_region(&integral_view)?;
            self.stream_context.prepare_region(&squared_view)?;
            self.stream_context.sync_scope(|| {
                // SAFETY: native ABI dispatch runs with all borrowed regions retained through completion.
                unsafe {
                    operation(
                        self.stream_context,
                        &source,
                        &mut integral_view,
                        &mut squared_view,
                        value,
                        squared_value,
                    )
                }
            })?;
        }

        // SAFETY: the integral producer writes both complete images, including their border.
        Ok(ImageSquaredIntegral {
            integral: unsafe { integral.assume_init() },
            squared: unsafe { squared.assume_init() },
        })
    }
}

type SquaredIntegralCallback<S, T> = unsafe fn(
    &StreamContext,
    &ImageView<'_, u8, C1>,
    &mut ImageOutputView<'_, T, C1>,
    &mut ImageOutputView<'_, S, C1>,
    T,
    S,
) -> Result<()>;
