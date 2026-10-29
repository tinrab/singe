use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{AC4, ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::{
        ImageBacking,
        image::{
            ImagePipeline,
            operation_traits::{AlphaCompColorKeyImage, ColorKeyImage},
        },
    },
    types::AlphaOperation,
};

impl<L> ImagePipeline<'_, u8, L>
where
    L: ChannelLayout,
    Self: ColorKeyImage<L>,
    <Self as ColorKeyImage<L>>::ColorKey: Copy,
{
    /// Performs native complement color-key replacement of `left` by `right`.
    /// The provided C1 implementation takes a scalar key; C3 and C4 take three and four
    /// key components respectively. Uses the native color-key comparison and replacement rules.
    /// Writes the native composition into the supplied destination.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    pub fn comp_color_key_into(
        stream_context: &StreamContext,
        left: &ImageView<'_, u8, L>,
        right: &ImageView<'_, u8, L>,
        color_key: <Self as ColorKeyImage<L>>::ColorKey,
        destination: &mut ImageOutputView<'_, u8, L>,
    ) -> Result<()> {
        <Self as ColorKeyImage<L>>::comp_color_key_image(
            stream_context,
            left,
            right,
            color_key,
            destination,
        )
    }
}

impl ImagePipeline<'_, u8, AC4>
where
    Self: AlphaCompColorKeyImage,
{
    /// Performs native complement color-key replacement of `source` by `other` with alpha blending.
    /// `alpha` and `other_alpha` are constant source opacities from zero through 255.
    /// `color_key` supplies four native key components. `operation` selects native composition;
    /// the native API excludes premultiplied-alpha operations, and the wrapper forwards the
    /// selector without validating it.
    /// Writes the native composition into the supplied destination.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    pub fn alpha_comp_color_key_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, AC4>,
        alpha: u8,
        other: &ImageView<'_, u8, AC4>,
        other_alpha: u8,
        color_key: [u8; 4],
        destination: &mut ImageOutputView<'_, u8, AC4>,
        operation: AlphaOperation,
    ) -> Result<()> {
        <Self as AlphaCompColorKeyImage>::alpha_comp_color_key_image(
            stream_context,
            source,
            alpha,
            other,
            other_alpha,
            color_key,
            destination,
            operation,
        )
    }
}

impl ImagePipeline<'_, u8, AC4>
where
    u8: SupportedImage<AC4>,
    Self: AlphaCompColorKeyImage,
{
    /// Performs native complement color-key replacement of the pipeline image by `other` with alpha blending.
    /// `alpha` and `other_alpha` are constant source opacities from zero through 255.
    /// `color_key` supplies four native key components. `operation` selects native composition;
    /// the native API excludes premultiplied-alpha operations, and the wrapper forwards the
    /// selector without validating it.
    /// Always allocates a new owned image with the pipeline image dimensions.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    /// Image allocation, view construction can also fail.
    pub fn alpha_comp_color_key(
        self,
        alpha: u8,
        other: &ImageView<'_, u8, AC4>,
        other_alpha: u8,
        color_key: [u8; 4],
        operation: AlphaOperation,
    ) -> Result<Self> {
        self.ensure_eager()?;

        let mut destination =
            Image::<u8, AC4>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            <Self as AlphaCompColorKeyImage>::alpha_comp_color_key_image(
                self.stream_context,
                &source,
                alpha,
                other,
                other_alpha,
                color_key,
                &mut destination_view,
                operation,
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

impl<L> ImagePipeline<'_, u8, L>
where
    L: ChannelLayout,
    u8: SupportedImage<L>,
    Self: ColorKeyImage<L>,
    <Self as ColorKeyImage<L>>::ColorKey: Copy,
{
    /// Performs native complement color-key replacement of the pipeline image by `other`.
    /// The provided C1 implementation takes a scalar key; C3 and C4 take three and four
    /// key components respectively. Uses the native color-key comparison and replacement rules.
    /// Always allocates a new owned image with the pipeline image dimensions.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    /// Image allocation, view construction can also fail.
    pub fn comp_color_key(
        self,
        other: &ImageView<'_, u8, L>,
        color_key: <Self as ColorKeyImage<L>>::ColorKey,
    ) -> Result<Self> {
        self.ensure_eager()?;

        let mut destination =
            Image::<u8, L>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            if L::IGNORES_ALPHA {
                destination_view.copy_from(&source)?;
            }
            <Self as ColorKeyImage<L>>::comp_color_key_image(
                self.stream_context,
                &source,
                other,
                color_key,
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
