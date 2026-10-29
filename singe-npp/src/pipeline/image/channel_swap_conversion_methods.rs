use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage, exchange,
        view::{C3, C4, ImageOutputView, ImageView},
    },
    pipeline::{ImageBacking, image::ImagePipeline},
};

mod to_c4_methods;

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
pub unsafe trait SwapChannelsC4ToC3Image<T> {
    fn swap_channels_c4_to_c3_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C4>,
        destination: &mut ImageOutputView<'_, T, C3>,
        order: [i32; 3],
    ) -> Result<()>;
}

macro_rules! impl_swap_channels_c4_to_c3_image {
    ($ty:ty, $swap:path) => {
        unsafe impl<'a> SwapChannelsC4ToC3Image<$ty> for ImagePipeline<'a, $ty, C4> {
            fn swap_channels_c4_to_c3_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C4>,
                destination: &mut ImageOutputView<'_, $ty, C3>,
                order: [i32; 3],
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $swap(stream_context, source, destination, order) })
            }
        }
    };
}

impl_swap_channels_c4_to_c3_image!(u8, exchange::swap_channels_c4_to_c3);

impl_swap_channels_c4_to_c3_image!(u16, exchange::swap_channels_c4_to_c3);

impl_swap_channels_c4_to_c3_image!(i16, exchange::swap_channels_c4_to_c3);

impl_swap_channels_c4_to_c3_image!(i32, exchange::swap_channels_c4_to_c3);

impl_swap_channels_c4_to_c3_image!(f32, exchange::swap_channels_c4_to_c3);

impl<T> ImagePipeline<'_, T, C4>
where
    T: Copy,
    Self: SwapChannelsC4ToC3Image<T>,
{
    /// Reorders three selected C4 channels into C3.
    ///
    /// Destination channel `i` receives source channel `order[i]`. The three entries
    /// must be distinct indices in `0..4`, allowing the fourth source channel to be selected.
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
    pub fn swap_channels_to_c3_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C4>,
        destination: &mut ImageOutputView<'_, T, C3>,
        order: [i32; 3],
    ) -> Result<()> {
        <Self as SwapChannelsC4ToC3Image<T>>::swap_channels_c4_to_c3_image(
            stream_context,
            source,
            destination,
            order,
        )
    }
}

impl<'a, T> ImagePipeline<'a, T, C4>
where
    T: Copy + SupportedImage<C3>,
    Self: SwapChannelsC4ToC3Image<T>,
{
    /// Reorders three selected C4 channels into C3.
    ///
    /// Destination channel `i` receives source channel `order[i]`. The three entries
    /// must be distinct indices in `0..4`, allowing the fourth source channel to be selected.
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
    pub fn swap_channels_to_c3(self, order: [i32; 3]) -> Result<ImagePipeline<'a, T, C3>> {
        self.ensure_eager()?;

        let mut destination =
            Image::<T, C3>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            <Self as SwapChannelsC4ToC3Image<T>>::swap_channels_c4_to_c3_image(
                self.stream_context,
                &source,
                &mut destination_view,
                order,
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
