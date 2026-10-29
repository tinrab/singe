use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{C1, C4, ImageOutputView, ImageView},
    },
    pipeline::image::{
        ImageBacking,
        channels_duplicate::{DuplicateToC4Image, ImagePipeline},
    },
};

impl<T> ImagePipeline<'_, T, C1>
where
    T: Copy,
    Self: DuplicateToC4Image<T>,
{
    /// Duplicates each C1 source sample into all four C4 channels.
    ///
    /// Every destination channel receives the same source sample.
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
    pub fn duplicate_to_c4_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageOutputView<'_, T, C4>,
    ) -> Result<()> {
        <Self as DuplicateToC4Image<T>>::duplicate_to_c4_image(stream_context, source, destination)
    }
}

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy + SupportedImage<C4>,
    Self: DuplicateToC4Image<T>,
{
    /// Duplicates each C1 source sample into all four C4 channels.
    ///
    /// Every destination channel receives the same source sample.
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
    /// Output allocation or image view construction can also fail.
    pub fn duplicate_to_c4(self) -> Result<ImagePipeline<'a, T, C4>> {
        self.ensure_eager()?;

        let mut destination =
            Image::<T, C4>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            <Self as DuplicateToC4Image<T>>::duplicate_to_c4_image(
                self.stream_context,
                &source,
                &mut destination_view,
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
