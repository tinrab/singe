use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{AC4, ImageOutputView, ImageView},
    },
    pipeline::image::{ImageBacking, ImagePipeline},
};

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
pub unsafe trait SwapChannelsAc4Image<T> {
    fn swap_channels_ac4_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, AC4>,
        destination: &mut ImageOutputView<'_, T, AC4>,
        order: [i32; 3],
    ) -> Result<()>;
}

impl<T> ImagePipeline<'_, T, AC4>
where
    T: Copy,
    Self: SwapChannelsAc4Image<T>,
{
    /// Reorders the three AC4 color channels while preserving destination alpha.
    ///
    /// Color channel `i` receives source channel `order[i]`. The three order entries
    /// must be distinct indices in `0..3`.
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
    pub fn swap_channels_ac4_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, AC4>,
        destination: &mut ImageOutputView<'_, T, AC4>,
        order: [i32; 3],
    ) -> Result<()> {
        <Self as SwapChannelsAc4Image<T>>::swap_channels_ac4_image(
            stream_context,
            source,
            destination,
            order,
        )
    }
}

impl<T> ImagePipeline<'_, T, AC4>
where
    T: Copy + SupportedImage<AC4>,
    Self: SwapChannelsAc4Image<T>,
{
    /// Reorders the three AC4 color channels while preserving destination alpha.
    ///
    /// Color channel `i` receives source channel `order[i]`. The three order entries
    /// must be distinct indices in `0..3`.
    /// Allocates a new source-sized owned image and copies the source before reordering
    /// to preserve its alpha values.
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
    /// Output allocation or image view construction can also fail. Copying source alpha
    /// can fail.
    pub fn swap_channels_ac4(self, order: [i32; 3]) -> Result<Self> {
        self.ensure_eager()?;

        let mut destination =
            Image::<T, AC4>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            destination_view.copy_from(&source)?;
            <Self as SwapChannelsAc4Image<T>>::swap_channels_ac4_image(
                self.stream_context,
                &source,
                &mut destination_view,
                order,
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
