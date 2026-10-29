use singe_cuda::memory::DeviceMemory;

use crate::{
    error::Result,
    image::{Image, SupportedImage, view::C1},
    pipeline::image::{
        ImageBacking, ImagePipeline, operation_traits::AbsoluteDifferenceConstantImage,
    },
};

impl<T> ImagePipeline<'_, T, C1>
where
    T: Copy + SupportedImage<C1>,
    Self: AbsoluteDifferenceConstantImage<T>,
{
    /// Computes the absolute difference between each C1 sample and `constant`.
    /// Always allocates a new owned C1 image, including when the input already owns storage.
    /// Prepares participating images, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Allocation or image view construction can also fail.
    pub fn absolute_difference_constant(self, constant: T) -> Result<Self> {
        self.ensure_eager()?;

        let mut destination =
            Image::<T, C1>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            <Self as AbsoluteDifferenceConstantImage<T>>::absolute_difference_constant_image(
                self.stream_context,
                &source,
                constant,
                &mut destination_view,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }

    /// Computes the absolute difference between each C1 sample and a device-resident constant.
    /// Reads constants from device memory. Uses the first element; additional elements are ignored.
    /// Always allocates a new owned C1 image, including when the input already owns storage.
    /// Prepares participating images and device constants, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The device buffer must contain at least one element.
    /// Allocation or image view construction can also fail.
    pub fn absolute_difference_device_constant(self, constant: &DeviceMemory<T>) -> Result<Self> {
        self.ensure_eager()?;

        let mut destination =
            Image::<T, C1>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            <Self as AbsoluteDifferenceConstantImage<T>>::absolute_difference_device_constant_image(
                self.stream_context,
                &source,
                constant,
                &mut destination_view,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}
