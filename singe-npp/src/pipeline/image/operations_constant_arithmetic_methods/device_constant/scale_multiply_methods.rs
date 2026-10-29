use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView, ImageViewMut},
    },
    pipeline::image::{
        ImageBacking, ImagePipeline,
        operation_shapes::{DeviceConstantInPlaceOperation, DeviceConstantOperation},
        operation_traits::MultiplyDeviceConstantScaleImage,
    },
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: MultiplyDeviceConstantScaleImage<T, L>,
{
    /// Multiplies each processed sample by the corresponding constant.
    /// Reads constants from device memory. Uses the first one, three, or four elements for C1,
    /// C3/AC4, or C4 respectively; additional elements are ignored.
    /// Uses NPP's fixed scaling by the maximum value of the unsigned sample bit width.
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
    pub fn multiply_device_constant_scale_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: &DeviceMemory<<Self as MultiplyDeviceConstantScaleImage<T, L>>::Constant>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()> {
        <Self as MultiplyDeviceConstantScaleImage<T, L>>::multiply_device_constant_scale_image(
            stream_context,
            source,
            constant,
            destination,
        )
    }

    /// Multiplies each processed sample by the corresponding constant.
    /// Reads constants from device memory. Uses the first one, three, or four elements for C1,
    /// C3/AC4, or C4 respectively; additional elements are ignored.
    /// Uses NPP's fixed scaling by the maximum value of the unsigned sample bit width.
    /// Updates the supplied image in place. AC4 alpha remains unchanged.
    /// Prepares participating images and device constants, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The device buffer must contain at least one constant per processed channel.
    pub fn multiply_device_constant_scale_in_place(
        stream_context: &StreamContext,
        constant: &DeviceMemory<<Self as MultiplyDeviceConstantScaleImage<T, L>>::Constant>,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as MultiplyDeviceConstantScaleImage<T, L>>::multiply_device_constant_scale_image_in_place(
            stream_context,
            constant,
            source_destination,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: MultiplyDeviceConstantScaleImage<T, L>,
{
    /// Multiplies each processed sample by the corresponding constant.
    /// Reads constants from device memory. Uses the first one, three, or four elements for C1,
    /// C3/AC4, or C4 respectively; additional elements are ignored.
    /// Uses NPP's fixed scaling by the maximum value of the unsigned sample bit width.
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
    pub fn multiply_device_constant_scale(
        self,
        constant: &DeviceMemory<<Self as MultiplyDeviceConstantScaleImage<T, L>>::Constant>,
    ) -> Result<Self> {
        self.multiply_device_constant_scale_operation(
            constant,
            <Self as MultiplyDeviceConstantScaleImage<T, L>>::multiply_device_constant_scale_image,
            <Self as MultiplyDeviceConstantScaleImage<T, L>>::multiply_device_constant_scale_image_in_place,
        )
    }

    fn multiply_device_constant_scale_operation(
        mut self,
        constant: &DeviceMemory<<Self as MultiplyDeviceConstantScaleImage<T, L>>::Constant>,
        operation: DeviceConstantOperation<
            T,
            L,
            <Self as MultiplyDeviceConstantScaleImage<T, L>>::Constant,
        >,
        operation_in_place: DeviceConstantInPlaceOperation<
            T,
            L,
            <Self as MultiplyDeviceConstantScaleImage<T, L>>::Constant,
        >,
    ) -> Result<Self> {
        self.ensure_eager()?;

        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                operation_in_place(self.stream_context, constant, &mut image_view)?;
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
                operation(self.stream_context, source, constant, &mut destination_view)?;
                // SAFETY: the completed operation above wrote every element.
                self.backing = ImageBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}
