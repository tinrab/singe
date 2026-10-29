use singe_cuda::memory::DeviceMemory;

use crate::{
    error::Result,
    image::{Image, SupportedImage, view::ChannelLayout},
    pipeline::image::{
        ImageBacking, ImagePipeline,
        operation_shapes::{DeviceConstantScaleInPlaceOperation, DeviceConstantScaleOperation},
        operation_traits::DeviceConstantArithmeticImage,
    },
};

mod into_methods;
mod scale_multiply_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: DeviceConstantArithmeticImage<T, L>,
{
    /// Adds the corresponding constant to each processed sample.
    /// Reads constants from device memory. Uses the first one, three, or four elements for C1,
    /// C3/AC4, or C4 respectively; additional elements are ignored.
    /// The provided half-precision implementations use `f32` constants.
    /// Applies native integer scaling and saturation using `scale_factor`.
    /// The provided floating-point implementations ignore `scale_factor`.
    /// Reuses owned backing; borrowed input produces a new owned image.
    /// Preserves alpha from the pipeline image for AC4.
    /// Prepares participating images and device constants, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The device buffer must contain at least one constant per processed channel.
    /// Allocation or image view construction, including alpha preservation copies, can also fail.
    pub fn add_device_constant(
        self,
        constant: &DeviceMemory<<Self as DeviceConstantArithmeticImage<T, L>>::Constant>,
        scale_factor: i32,
    ) -> Result<Self> {
        self.device_constant_arithmetic(
            constant,
            scale_factor,
            <Self as DeviceConstantArithmeticImage<T, L>>::add_device_constant_image,
            <Self as DeviceConstantArithmeticImage<T, L>>::add_device_constant_image_in_place,
        )
    }

    /// Subtracts the corresponding constant from each processed sample.
    /// Reads constants from device memory. Uses the first one, three, or four elements for C1,
    /// C3/AC4, or C4 respectively; additional elements are ignored.
    /// The provided half-precision implementations use `f32` constants.
    /// Applies native integer scaling and saturation using `scale_factor`.
    /// The provided floating-point implementations ignore `scale_factor`.
    /// Reuses owned backing; borrowed input produces a new owned image.
    /// Preserves alpha from the pipeline image for AC4.
    /// Prepares participating images and device constants, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The device buffer must contain at least one constant per processed channel.
    /// Allocation or image view construction, including alpha preservation copies, can also fail.
    pub fn subtract_device_constant(
        self,
        constant: &DeviceMemory<<Self as DeviceConstantArithmeticImage<T, L>>::Constant>,
        scale_factor: i32,
    ) -> Result<Self> {
        self.device_constant_arithmetic(
            constant,
            scale_factor,
            <Self as DeviceConstantArithmeticImage<T, L>>::subtract_device_constant_image,
            <Self as DeviceConstantArithmeticImage<T, L>>::subtract_device_constant_image_in_place,
        )
    }

    /// Multiplies each processed sample by the corresponding constant.
    /// Reads constants from device memory. Uses the first one, three, or four elements for C1,
    /// C3/AC4, or C4 respectively; additional elements are ignored.
    /// The provided half-precision implementations use `f32` constants.
    /// Applies native integer scaling and saturation using `scale_factor`.
    /// The provided floating-point implementations ignore `scale_factor`.
    /// Reuses owned backing; borrowed input produces a new owned image.
    /// Preserves alpha from the pipeline image for AC4.
    /// Prepares participating images and device constants, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The device buffer must contain at least one constant per processed channel.
    /// Allocation or image view construction, including alpha preservation copies, can also fail.
    pub fn multiply_device_constant(
        self,
        constant: &DeviceMemory<<Self as DeviceConstantArithmeticImage<T, L>>::Constant>,
        scale_factor: i32,
    ) -> Result<Self> {
        self.device_constant_arithmetic(
            constant,
            scale_factor,
            <Self as DeviceConstantArithmeticImage<T, L>>::multiply_device_constant_image,
            <Self as DeviceConstantArithmeticImage<T, L>>::multiply_device_constant_image_in_place,
        )
    }

    /// Divides each processed sample by the corresponding constant.
    /// Reads constants from device memory. Uses the first one, three, or four elements for C1,
    /// C3/AC4, or C4 respectively; additional elements are ignored.
    /// The provided half-precision implementations use `f32` constants.
    /// Applies native integer scaling and saturation using `scale_factor`.
    /// The provided floating-point implementations ignore `scale_factor`.
    /// Reuses owned backing; borrowed input produces a new owned image.
    /// Preserves alpha from the pipeline image for AC4.
    /// Prepares participating images and device constants, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The device buffer must contain at least one constant per processed channel.
    /// Allocation or image view construction, including alpha preservation copies, can also fail.
    pub fn divide_device_constant(
        self,
        constant: &DeviceMemory<<Self as DeviceConstantArithmeticImage<T, L>>::Constant>,
        scale_factor: i32,
    ) -> Result<Self> {
        self.device_constant_arithmetic(
            constant,
            scale_factor,
            <Self as DeviceConstantArithmeticImage<T, L>>::divide_device_constant_image,
            <Self as DeviceConstantArithmeticImage<T, L>>::divide_device_constant_image_in_place,
        )
    }

    fn device_constant_arithmetic(
        mut self,
        constant: &DeviceMemory<<Self as DeviceConstantArithmeticImage<T, L>>::Constant>,
        scale_factor: i32,
        operation: DeviceConstantScaleOperation<
            T,
            L,
            <Self as DeviceConstantArithmeticImage<T, L>>::Constant,
        >,
        operation_in_place: DeviceConstantScaleInPlaceOperation<
            T,
            L,
            <Self as DeviceConstantArithmeticImage<T, L>>::Constant,
        >,
    ) -> Result<Self> {
        self.ensure_eager()?;

        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                operation_in_place(self.stream_context, constant, &mut image_view, scale_factor)?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = Image::<T, L>::create_uninit(
                    self.stream_context.cuda_context(),
                    source.size(),
                )?;
                let mut destination_view = destination.output()?;
                if L::IGNORES_ALPHA {
                    destination_view.copy_from(source)?;
                }
                operation(
                    self.stream_context,
                    source,
                    constant,
                    &mut destination_view,
                    scale_factor,
                )?;
                // SAFETY: the completed operation above wrote every element.
                self.backing = ImageBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}
