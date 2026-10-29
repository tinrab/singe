mod into_methods;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::{
        ImageBacking,
        image::{ImagePipeline, filtering::HighLowGaussBorderFilterImage},
    },
    types::{BorderType, MaskSize, Point},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: HighLowGaussBorderFilterImage<T, L>,
{
    /// Applies the native high-pass filter with source border control.
    ///
    /// `source_offset` locates the output ROI in the full source image. Allocates a new
    /// owned image from that offset to the lower-right source edge.
    /// Forwards `mask_size` and `border_type` to NPP, which may reject unsupported values.
    /// For AC4, copies the source window before filtering to preserve its alpha values.
    /// Row padding remains unwritten.
    /// Prepares image regions and waits for producers on other streams. Rejects stream
    /// capture and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// The source offset must address a source pixel; destination-sized ROI endpoints must
    /// not overflow or exceed the source image.
    /// Output allocation, image view construction, or copying source alpha can also fail.
    pub fn filter_high_pass_border_partial(
        self,
        source_offset: Point,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<Self> {
        self.high_low_gauss_border_filter(
            source_offset,
            mask_size,
            border_type,
            <Self as HighLowGaussBorderFilterImage<T, L>>::filter_high_pass_border_image,
        )
    }

    /// Applies the native low-pass filter with source border control.
    ///
    /// `source_offset` locates the output ROI in the full source image. Allocates a new
    /// owned image from that offset to the lower-right source edge.
    /// Forwards `mask_size` and `border_type` to NPP, which may reject unsupported values.
    /// For AC4, copies the source window before filtering to preserve its alpha values.
    /// Row padding remains unwritten.
    /// Prepares image regions and waits for producers on other streams. Rejects stream
    /// capture and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// The source offset must address a source pixel; destination-sized ROI endpoints must
    /// not overflow or exceed the source image.
    /// Output allocation, image view construction, or copying source alpha can also fail.
    pub fn filter_low_pass_border_partial(
        self,
        source_offset: Point,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<Self> {
        self.high_low_gauss_border_filter(
            source_offset,
            mask_size,
            border_type,
            <Self as HighLowGaussBorderFilterImage<T, L>>::filter_low_pass_border_image,
        )
    }

    /// Applies the native Gaussian filter with source border control.
    ///
    /// `source_offset` locates the output ROI in the full source image. Allocates a new
    /// owned image from that offset to the lower-right source edge.
    /// Forwards `mask_size` and `border_type` to NPP, which may reject unsupported values.
    /// For AC4, copies the source window before filtering to preserve its alpha values.
    /// Row padding remains unwritten.
    /// Prepares image regions and waits for producers on other streams. Rejects stream
    /// capture and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// The source offset must address a source pixel; destination-sized ROI endpoints must
    /// not overflow or exceed the source image.
    /// Output allocation, image view construction, or copying source alpha can also fail.
    pub fn filter_gauss_border_partial(
        self,
        source_offset: Point,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<Self> {
        self.high_low_gauss_border_filter(
            source_offset,
            mask_size,
            border_type,
            <Self as HighLowGaussBorderFilterImage<T, L>>::filter_gauss_border_image,
        )
    }

    fn high_low_gauss_border_filter(
        self,
        source_offset: Point,
        mask_size: MaskSize,
        border_type: BorderType,
        filter: HighLowGaussBorderFilterCallback<L, T>,
    ) -> Result<Self> {
        self.ensure_eager()?;
        let source = self.view()?;
        let output_window = source.window_from(source_offset)?;
        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), output_window.size())?;

        {
            let mut destination_view = destination.output()?;
            if L::IGNORES_ALPHA {
                destination_view.copy_from(&output_window.view())?;
            }
            filter(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                mask_size,
                border_type,
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

type HighLowGaussBorderFilterCallback<L, T> = fn(
    &StreamContext,
    &ImageView<'_, T, L>,
    Point,
    &mut ImageOutputView<'_, T, L>,
    MaskSize,
    BorderType,
) -> Result<()>;
