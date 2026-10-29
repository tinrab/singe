use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView, ImageViewMut},
    },
    pipeline::image::{ImageBacking, ImagePipeline},
};

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
pub unsafe trait SwapChannelsImage<T, L, const CHANNELS: usize> {
    fn swap_channels_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        order: [i32; CHANNELS],
    ) -> Result<()>;

    fn swap_channels_image_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, L>,
        order: [i32; CHANNELS],
    ) -> Result<()>;
}

macro_rules! impl_swap_channels_image {
    ($ty:ty, $layout:ty, $channels:literal, $swap:path, $swap_in_place:path) => {
        unsafe impl<'a> SwapChannelsImage<$ty, $layout, $channels>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn swap_channels_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                order: [i32; $channels],
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $swap(stream_context, source, destination, order) })
            }

            fn swap_channels_image_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, $ty, $layout>,
                order: [i32; $channels],
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(image)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $swap_in_place(stream_context, image, order) })
            }
        }
    };
}

mod dispatch_impls;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    /// Reorders image channels using zero-based source indices in `order`.
    ///
    /// Destination channel `i` receives source channel `order[i]`. The provided
    /// implementations support C3 and C4 layouts, with three or four distinct indices in
    /// `0..3` or `0..4`, respectively.
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
    pub fn swap_channels_into<const CHANNELS: usize>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        order: [i32; CHANNELS],
    ) -> Result<()>
    where
        Self: SwapChannelsImage<T, L, CHANNELS>,
    {
        <Self as SwapChannelsImage<T, L, CHANNELS>>::swap_channels_image(
            stream_context,
            source,
            destination,
            order,
        )
    }

    /// Reorders image channels using zero-based source indices in `order`.
    ///
    /// Destination channel `i` receives source channel `order[i]`. The provided
    /// implementations support C3 and C4 layouts, with three or four distinct indices in
    /// `0..3` or `0..4`, respectively.
    /// Updates the supplied initialized image in place.
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
    pub fn swap_channels_in_place<const CHANNELS: usize>(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, L>,
        order: [i32; CHANNELS],
    ) -> Result<()>
    where
        Self: SwapChannelsImage<T, L, CHANNELS>,
    {
        <Self as SwapChannelsImage<T, L, CHANNELS>>::swap_channels_image_in_place(
            stream_context,
            image,
            order,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
{
    /// Reorders image channels using zero-based source indices in `order`.
    ///
    /// Destination channel `i` receives source channel `order[i]`. The provided
    /// implementations support C3 and C4 layouts, with three or four distinct indices in
    /// `0..3` or `0..4`, respectively.
    /// Reuses an owned image in place; for borrowed input, allocates a new source-sized
    /// image before reordering.
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
    pub fn swap_channels<const CHANNELS: usize>(mut self, order: [i32; CHANNELS]) -> Result<Self>
    where
        Self: SwapChannelsImage<T, L, CHANNELS>,
    {
        self.ensure_eager()?;

        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as SwapChannelsImage<T, L, CHANNELS>>::swap_channels_image_in_place(
                    self.stream_context,
                    &mut image_view,
                    order,
                )?;
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
                <Self as SwapChannelsImage<T, L, CHANNELS>>::swap_channels_image(
                    self.stream_context,
                    source,
                    &mut destination_view,
                    order,
                )?;
                // SAFETY: the completed operation above wrote every element.
                self.backing = ImageBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}
