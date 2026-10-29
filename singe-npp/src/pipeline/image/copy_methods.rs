use singe_cuda::view::DeviceRepr;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::image::{ImageBacking, ImagePipeline},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: SupportedImage<L>,
    L: ChannelLayout,
{
    /// Copies every logical channel, including alpha, into the supplied destination.
    ///
    /// Source and destination dimensions must match. Prepares both regions, waits for
    /// producers on other streams, rejects capture, and completes CUDA row transfers before
    /// returning successfully. Destination row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, transfer or native execution, or stream synchronization fail.
    /// Source and destination dimensions must match.
    pub fn copy_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()> {
        <Self as CopyImage<T, L>>::copy(stream_context, source, destination)
    }

    /// Copies every logical channel, including alpha, into a new owned image.
    ///
    /// Always allocates a source-sized image. Prepares source and destination regions,
    /// waits for producers on other streams, rejects capture, and completes CUDA row
    /// transfers before returning successfully. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, transfer or native execution, or stream synchronization fail.
    /// Allocation, copying borrowed input, or image view construction can also fail.
    pub fn copy(self) -> Result<Self> {
        self.ensure_eager()?;
        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), self.size())?;
        <Self as CopyImage<T, L>>::copy(
            self.stream_context,
            &self.view()?,
            &mut destination.output()?,
        )?;
        // SAFETY: completed copy wrote all logical channels, including alpha, without exposing pitch padding.
        let destination = unsafe { destination.assume_init() };
        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }

    /// Completes outstanding stream work and returns an initialized owned image.
    ///
    /// For owned input, synchronizes the selected stream and returns the existing image.
    /// For borrowed input, allocates a source-sized image and completes a copy of every
    /// logical channel, including alpha. The copy prepares participating regions and waits
    /// for producers on other streams. Rejects capture. Row padding remains outside the
    /// initialized logical image.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, transfer or native execution, or stream synchronization fail.
    /// Allocation, copying borrowed input, or image view construction can also fail.
    pub fn finish(self) -> Result<Image<T, L>> {
        self.ensure_eager()?;
        match self.backing {
            ImageBacking::Owned(image) => {
                self.stream_context.sync_scope(|| Ok(()))?;
                Ok(image)
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = Image::<T, L>::create_uninit(
                    self.stream_context.cuda_context(),
                    source.size(),
                )?;
                <Self as CopyImage<T, L>>::copy(
                    self.stream_context,
                    &source,
                    &mut destination.output()?,
                )?;
                // SAFETY: the completed copy established every logical channel.
                Ok(unsafe { destination.assume_init() })
            }
        }
    }
}

/// # Safety
/// Complete borrowed work on every path and initialize every logical channel, including alpha.
pub unsafe trait CopyImage<T, L> {
    fn copy(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>;
}

// SAFETY: each completed CUDA row transfer copies valid T values for every logical channel.
unsafe impl<T: DeviceRepr, L: ChannelLayout> CopyImage<T, L> for ImagePipeline<'_, T, L> {
    fn copy(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        stream_context.sync_scope(|| destination.copy_from(source))
    }
}
