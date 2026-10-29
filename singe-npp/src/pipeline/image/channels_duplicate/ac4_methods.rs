use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{AC4, C1, C4, ImageOutputView, ImageView},
    },
    pipeline::image::{
        ImageBacking,
        channels::SetChannelImage,
        channels_duplicate::{DuplicateToAC4Image, ImagePipeline},
    },
};

impl<T> ImagePipeline<'_, T, C1>
where
    T: Copy,
    Self: DuplicateToAC4Image<T>,
{
    /// Duplicates each C1 source sample into the three AC4 color channels.
    ///
    /// For AC4, destination alpha remains unwritten.
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
    pub fn duplicate_to_ac4_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageOutputView<'_, T, AC4>,
    ) -> Result<()> {
        <Self as DuplicateToAC4Image<T>>::duplicate_to_ac4_image(
            stream_context,
            source,
            destination,
        )
    }
}

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy + SupportedImage<AC4>,
    Self: DuplicateToAC4Image<T>,
{
    /// Duplicates each C1 source sample into the three AC4 color channels.
    ///
    /// Initializes the fourth storage channel to the supplied `alpha` value after
    /// duplicating the three color channels.
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
    /// Output allocation or image view construction can also fail. Alpha channel fill or
    /// layout conversion can fail.
    pub fn duplicate_to_ac4(self, alpha: T) -> Result<ImagePipeline<'a, T, AC4>>
    where
        ImagePipeline<'a, T, C4>: SetChannelImage<T, C4>,
    {
        self.ensure_eager()?;
        let mut destination =
            Image::<T, AC4>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            <Self as DuplicateToAC4Image<T>>::duplicate_to_ac4_image(
                self.stream_context,
                &source,
                &mut destination_view,
            )?;
            <ImagePipeline<'a, T, C4> as SetChannelImage<T, C4>>::set_channel_image(
                self.stream_context,
                alpha,
                &mut destination_view.output().with_layout::<C4>()?,
                3,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: completed duplication writes RGB and completed channel fill writes alpha.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}
