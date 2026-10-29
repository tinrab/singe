use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView},
    },
    pipeline::image::{ImageBacking, ImagePipeline},
};

/// # Safety
/// Complete native work on every path and write valid values to every active channel.
/// AC4 dispatch must preserve the existing alpha channel.
pub unsafe trait SetImage<T, L> {
    type Value;

    fn set_image(
        stream_context: &StreamContext,
        value: Self::Value,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>;
}

macro_rules! impl_set_image {
    ($ty:ty, $layout:ty, $value_ty:ty, $set:path) => {
        // SAFETY: sealed native dispatch writes every active channel and the scope completes borrowed work.
        unsafe impl<'a> SetImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            type Value = $value_ty;

            fn set_image(
                stream_context: &StreamContext,
                value: Self::Value,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(destination)?;
                stream_context.sync_scope(|| {
                    // SAFETY: native ABI dispatch retains exclusive output storage through completion.
                    unsafe { $set(stream_context, value, destination) }
                })
            }
        }
    };
}

mod dispatch_impls;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: SetImage<T, L>,
    <Self as SetImage<T, L>>::Value: Copy,
{
    /// Fills every processed channel with the supplied value.
    ///
    /// `value` is a scalar for C1 or an array of per-channel values for C3, C4, or AC4;
    /// AC4 uses three values and does not process alpha.
    /// AC4 destination alpha remains unwritten; initialized alpha values are preserved.
    /// Prepares the destination region, waits for its producers on other streams, rejects
    /// capture, and returns after successful native execution and stream completion.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, transfer or native execution, or stream synchronization fail.
    pub fn set_into(
        stream_context: &StreamContext,
        value: <Self as SetImage<T, L>>::Value,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()> {
        <Self as SetImage<T, L>>::set_image(stream_context, value, destination)
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: SetImage<T, L>,
    <Self as SetImage<T, L>>::Value: Copy,
{
    /// Fills every processed channel with the supplied value.
    ///
    /// `value` is a scalar for C1 or an array of per-channel values for C3, C4, or AC4;
    /// AC4 uses three values and does not process alpha.
    /// Reuses an owned image in place. For borrowed input, allocates a source-sized image,
    /// copying the source first when needed to preserve AC4 alpha.
    /// Prepares the destination region, waits for its producers on other streams, rejects
    /// capture, and returns after successful native execution and stream completion.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, transfer or native execution, or stream synchronization fail.
    /// Allocation, copying borrowed input, or image view construction can also fail.
    pub fn set(mut self, value: <Self as SetImage<T, L>>::Value) -> Result<Self> {
        self.ensure_eager()?;
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.output_view()?;
                <Self as SetImage<T, L>>::set_image(self.stream_context, value, &mut image_view)?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = Image::<T, L>::create_uninit(
                    self.stream_context.cuda_context(),
                    source.size(),
                )?;
                let mut destination_view = destination.output()?;
                if L::IGNORES_ALPHA {
                    // Preserve the source alpha that the native AC4 set deliberately leaves untouched.
                    destination_view.copy_from(source)?;
                }
                <Self as SetImage<T, L>>::set_image(
                    self.stream_context,
                    value,
                    &mut destination_view,
                )?;
                // SAFETY: completed set wrote every active channel; AC4 alpha was copied from the source.
                self.backing = ImageBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}
