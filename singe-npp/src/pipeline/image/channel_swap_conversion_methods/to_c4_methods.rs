use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage, exchange,
        view::{C3, C4, ImageOutputView, ImageView},
    },
    pipeline::image::{ImageBacking, channel_swap_conversion_methods::ImagePipeline},
};

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
pub unsafe trait SwapChannelsC3ToC4Image<T> {
    fn swap_channels_c3_to_c4_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C3>,
        destination: &mut ImageOutputView<'_, T, C4>,
        order: [i32; 4],
        value: T,
    ) -> Result<()>;
}

macro_rules! impl_swap_channels_c3_to_c4_image {
    ($ty:ty, $swap:path) => {
        unsafe impl<'a> SwapChannelsC3ToC4Image<$ty> for ImagePipeline<'a, $ty, C3> {
            fn swap_channels_c3_to_c4_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C3>,
                destination: &mut ImageOutputView<'_, $ty, C4>,
                order: [i32; 4],
                value: $ty,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $swap(stream_context, source, destination, order, value)
                })
            }
        }
    };
}

impl_swap_channels_c3_to_c4_image!(u8, exchange::swap_channels_c3_to_c4);

impl_swap_channels_c3_to_c4_image!(u16, exchange::swap_channels_c3_to_c4);

impl_swap_channels_c3_to_c4_image!(i16, exchange::swap_channels_c3_to_c4);

impl_swap_channels_c3_to_c4_image!(i32, exchange::swap_channels_c3_to_c4);

impl_swap_channels_c3_to_c4_image!(f32, exchange::swap_channels_c3_to_c4);

impl<T> ImagePipeline<'_, T, C3>
where
    T: Copy,
    Self: SwapChannelsC3ToC4Image<T>,
{
    /// Reorders C3 channels into C4 with constant fill.
    ///
    /// Destination channel `i` receives source channel `order[i]`. An order entry of `3`
    /// writes `value` instead. Entries must lie in `0..=3`; source indices zero through
    /// two must not repeat, while fill entries may repeat.
    /// Source and destination pixel dimensions must match.
    /// The provided implementations retain sample type: `u8`, `u16`, `i16`, `i32`, or `f32`.
    /// Prepares source and destination regions, waits for producers on other streams,
    /// rejects capture, and returns after successful stream completion. Row padding remains
    /// unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or synchronization fail.
    /// Source and destination pixel dimensions must match.
    /// An out-of-range order index or repeated source index is rejected.
    pub fn swap_channels_to_c4_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C3>,
        destination: &mut ImageOutputView<'_, T, C4>,
        order: [i32; 4],
        value: T,
    ) -> Result<()> {
        <Self as SwapChannelsC3ToC4Image<T>>::swap_channels_c3_to_c4_image(
            stream_context,
            source,
            destination,
            order,
            value,
        )
    }
}

impl<'a, T> ImagePipeline<'a, T, C3>
where
    T: Copy + SupportedImage<C4>,
    Self: SwapChannelsC3ToC4Image<T>,
{
    /// Reorders C3 channels into C4 with constant fill.
    ///
    /// Destination channel `i` receives source channel `order[i]`. An order entry of `3`
    /// writes `value` instead. Entries must lie in `0..=3`; source indices zero through
    /// two must not repeat, while fill entries may repeat.
    /// Allocates a new source-sized owned image in the destination layout.
    /// The provided implementations retain sample type: `u8`, `u16`, `i16`, `i32`, or `f32`.
    /// Prepares source and destination regions, waits for producers on other streams,
    /// rejects capture, and returns after successful stream completion. Row padding remains
    /// unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or synchronization fail.
    /// An out-of-range order index or repeated source index is rejected.
    /// Output allocation or image view construction can also fail.
    pub fn swap_channels_to_c4(
        self,
        order: [i32; 4],
        value: T,
    ) -> Result<ImagePipeline<'a, T, C4>> {
        self.ensure_eager()?;

        let mut destination =
            Image::<T, C4>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            <Self as SwapChannelsC3ToC4Image<T>>::swap_channels_c3_to_c4_image(
                self.stream_context,
                &source,
                &mut destination_view,
                order,
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
}
