use crate::{
    error::Result,
    image::{Image, SupportedImage, view::ChannelLayout},
    pipeline::image::{ImageBacking, ImagePipeline, morphology_traits::MaskMorphologyImage},
    types::{Point, Size},
};

mod border_methods;
mod composite_into_methods;
mod composite_methods;
mod gray_methods;
mod into_methods;
mod three_by_three_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: MaskMorphologyImage<T, L>,
{
    /// Dilates each active channel using the nonzero entries of `mask`.
    /// Passes `mask.as_ptr()` directly to NPP as its device mask address. The slice storage
    /// must also be GPU-readable in the selected CUDA context; this call does not upload it.
    /// Allocates a new owned image for the valid source window, with width and height reduced
    /// by `mask_size.width - 1` and `mask_size.height - 1` respectively.
    /// Preserves AC4 alpha from the corresponding source window.
    /// Prepares participating image regions, waits for their producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Mask dimensions must be positive, their checked product must equal `mask.len()`, and
    /// `anchor` must lie within the mask.
    /// The source must contain the complete neighborhood and its valid output window.
    /// Image allocation, view construction, or alpha preservation copies can also fail.
    pub fn dilate(self, mask: &[u8], mask_size: Size, anchor: Point) -> Result<Self> {
        self.ensure_eager()?;
        let source = self.view()?.valid_window(mask_size, anchor)?;
        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), source.size())?;

        {
            let mut destination_view = destination.output()?;
            if L::IGNORES_ALPHA {
                destination_view.copy_from(&source.view())?;
            }
            <Self as MaskMorphologyImage<T, L>>::dilate_image(
                self.stream_context,
                &source,
                &mut destination_view,
                mask,
                mask_size,
                anchor,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }

    /// Erodes each active channel using the nonzero entries of `mask`.
    /// Passes `mask.as_ptr()` directly to NPP as its device mask address. The slice storage
    /// must also be GPU-readable in the selected CUDA context; this call does not upload it.
    /// Allocates a new owned image for the valid source window, with width and height reduced
    /// by `mask_size.width - 1` and `mask_size.height - 1` respectively.
    /// Preserves AC4 alpha from the corresponding source window.
    /// Prepares participating image regions, waits for their producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Mask dimensions must be positive, their checked product must equal `mask.len()`, and
    /// `anchor` must lie within the mask.
    /// The source must contain the complete neighborhood and its valid output window.
    /// Image allocation, view construction, or alpha preservation copies can also fail.
    pub fn erode(self, mask: &[u8], mask_size: Size, anchor: Point) -> Result<Self> {
        self.ensure_eager()?;
        let source = self.view()?.valid_window(mask_size, anchor)?;
        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), source.size())?;

        {
            let mut destination_view = destination.output()?;
            if L::IGNORES_ALPHA {
                destination_view.copy_from(&source.view())?;
            }
            <Self as MaskMorphologyImage<T, L>>::erode_image(
                self.stream_context,
                &source,
                &mut destination_view,
                mask,
                mask_size,
                anchor,
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
