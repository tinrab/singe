use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageOutputView, ImageView, ImageViewMut},
    pipeline::image::{ImagePipeline, operation_traits::DeviceConstantArithmeticImage},
};

mod multiplicative_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: DeviceConstantArithmeticImage<T, L>,
{
    /// Adds the corresponding constant to each processed sample.
    /// Reads constants from device memory. Uses the first one, three, or four elements for C1,
    /// C3/AC4, or C4 respectively; additional elements are ignored.
    /// The provided half-precision implementations use `f32` constants.
    /// Applies native integer scaling and saturation using `scale_factor`.
    /// The provided floating-point implementations ignore `scale_factor`.
    /// Writes active channels in the supplied destination. AC4 destination alpha retains its
    /// previous initialization state; this call does not initialize alpha.
    /// Prepares participating images and device constants, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The device buffer must contain at least one constant per processed channel.
    /// Source and destination image dimensions must match.
    pub fn add_device_constant_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: &DeviceMemory<<Self as DeviceConstantArithmeticImage<T, L>>::Constant>,
        destination: &mut ImageOutputView<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as DeviceConstantArithmeticImage<T, L>>::add_device_constant_image(
            stream_context,
            source,
            constant,
            destination,
            scale_factor,
        )
    }

    /// Adds the corresponding constant to each processed sample.
    /// Reads constants from device memory. Uses the first one, three, or four elements for C1,
    /// C3/AC4, or C4 respectively; additional elements are ignored.
    /// The provided half-precision implementations use `f32` constants.
    /// Applies native integer scaling and saturation using `scale_factor`.
    /// The provided floating-point implementations ignore `scale_factor`.
    /// Updates the supplied image in place. AC4 alpha remains unchanged.
    /// Prepares participating images and device constants, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The device buffer must contain at least one constant per processed channel.
    pub fn add_device_constant_in_place(
        stream_context: &StreamContext,
        constant: &DeviceMemory<<Self as DeviceConstantArithmeticImage<T, L>>::Constant>,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as DeviceConstantArithmeticImage<T, L>>::add_device_constant_image_in_place(
            stream_context,
            constant,
            source_destination,
            scale_factor,
        )
    }

    /// Subtracts the corresponding constant from each processed sample.
    /// Reads constants from device memory. Uses the first one, three, or four elements for C1,
    /// C3/AC4, or C4 respectively; additional elements are ignored.
    /// The provided half-precision implementations use `f32` constants.
    /// Applies native integer scaling and saturation using `scale_factor`.
    /// The provided floating-point implementations ignore `scale_factor`.
    /// Writes active channels in the supplied destination. AC4 destination alpha retains its
    /// previous initialization state; this call does not initialize alpha.
    /// Prepares participating images and device constants, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The device buffer must contain at least one constant per processed channel.
    /// Source and destination image dimensions must match.
    pub fn subtract_device_constant_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: &DeviceMemory<<Self as DeviceConstantArithmeticImage<T, L>>::Constant>,
        destination: &mut ImageOutputView<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as DeviceConstantArithmeticImage<T, L>>::subtract_device_constant_image(
            stream_context,
            source,
            constant,
            destination,
            scale_factor,
        )
    }

    /// Subtracts the corresponding constant from each processed sample.
    /// Reads constants from device memory. Uses the first one, three, or four elements for C1,
    /// C3/AC4, or C4 respectively; additional elements are ignored.
    /// The provided half-precision implementations use `f32` constants.
    /// Applies native integer scaling and saturation using `scale_factor`.
    /// The provided floating-point implementations ignore `scale_factor`.
    /// Updates the supplied image in place. AC4 alpha remains unchanged.
    /// Prepares participating images and device constants, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The device buffer must contain at least one constant per processed channel.
    pub fn subtract_device_constant_in_place(
        stream_context: &StreamContext,
        constant: &DeviceMemory<<Self as DeviceConstantArithmeticImage<T, L>>::Constant>,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as DeviceConstantArithmeticImage<T, L>>::subtract_device_constant_image_in_place(
            stream_context,
            constant,
            source_destination,
            scale_factor,
        )
    }
}
