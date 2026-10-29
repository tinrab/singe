use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageWindow},
    },
    pipeline::{
        ImageBacking,
        image::{
            ImagePipeline,
            filtering::{TypedEdgeDirectionalFilterImage, centered_filter_geometry},
        },
    },
    types::{MaskSize, Point, Size},
};

mod extended_methods;
mod filtering_typed_edge_border_methods;
mod scharr_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    /// Applies the native horizontal Sobel filter into the selected output type and layout.
    /// The provided typed implementations convert C1 `u8` or `i8` source samples to C1
    /// `i16` output samples.
    /// Uses the selected `mask_size`; NPP may reject unsupported masks.
    /// Allocates a new owned output for the valid source window. Reduces width and height
    /// by the selected square mask extent minus one.
    /// The wrapper requires a square mask and uses its center as the anchor.
    /// Prepares source and destination regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Writes every logical output
    /// sample; row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native filter execution, or stream synchronization fail.
    /// The source must contain the complete neighborhood and its valid output window.
    /// One-dimensional mask variants are rejected.
    /// Output allocation or image view construction can also fail.
    pub fn filter_sobel_horizontal_to<D, M>(
        self,
        mask_size: MaskSize,
    ) -> Result<ImagePipeline<'a, D, M>>
    where
        D: Copy + SupportedImage<M>,
        M: ChannelLayout,
        Self: TypedEdgeDirectionalFilterImage<T, L, D, M>,
    {
        self.typed_masked_edge_directional_filter(
            mask_size,
            <Self as TypedEdgeDirectionalFilterImage<T, L, D, M>>::filter_sobel_horizontal_to_image,
        )
    }

    /// Applies the native vertical Sobel filter into the selected output type and layout.
    /// The provided typed implementations convert C1 `u8` or `i8` source samples to C1
    /// `i16` output samples.
    /// Uses the selected `mask_size`; NPP may reject unsupported masks.
    /// Allocates a new owned output for the valid source window. Reduces width and height
    /// by the selected square mask extent minus one.
    /// The wrapper requires a square mask and uses its center as the anchor.
    /// Prepares source and destination regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Writes every logical output
    /// sample; row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native filter execution, or stream synchronization fail.
    /// The source must contain the complete neighborhood and its valid output window.
    /// One-dimensional mask variants are rejected.
    /// Output allocation or image view construction can also fail.
    pub fn filter_sobel_vertical_to<D, M>(
        self,
        mask_size: MaskSize,
    ) -> Result<ImagePipeline<'a, D, M>>
    where
        D: Copy + SupportedImage<M>,
        M: ChannelLayout,
        Self: TypedEdgeDirectionalFilterImage<T, L, D, M>,
    {
        self.typed_masked_edge_directional_filter(
            mask_size,
            <Self as TypedEdgeDirectionalFilterImage<T, L, D, M>>::filter_sobel_vertical_to_image,
        )
    }

    pub(crate) fn typed_edge_directional_filter<D, M>(
        self,
        filter: TypedEdgeDirectionalFilterCallback<D, L, M, T>,
    ) -> Result<ImagePipeline<'a, D, M>>
    where
        D: Copy + SupportedImage<M>,
        M: ChannelLayout,
    {
        self.ensure_eager()?;
        let source = self
            .view()?
            .valid_window(Size::new(3, 3), Point { x: 1, y: 1 })?;
        let mut destination =
            Image::<D, M>::create_uninit(self.stream_context.cuda_context(), source.size())?;

        {
            let mut destination_view = destination.output()?;
            filter(self.stream_context, &source, &mut destination_view)?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }

    pub(crate) fn typed_masked_edge_directional_filter<D, M>(
        self,
        mask_size: MaskSize,
        filter: TypedMaskedEdgeDirectionalFilterCallback<D, L, M, T>,
    ) -> Result<ImagePipeline<'a, D, M>>
    where
        D: Copy + SupportedImage<M>,
        M: ChannelLayout,
    {
        self.ensure_eager()?;
        let (mask, anchor) = centered_filter_geometry(mask_size)?;
        let source = self.view()?.valid_window(mask, anchor)?;
        let mut destination =
            Image::<D, M>::create_uninit(self.stream_context.cuda_context(), source.size())?;

        {
            let mut destination_view = destination.output()?;
            filter(
                self.stream_context,
                &source,
                &mut destination_view,
                mask_size,
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

type TypedEdgeDirectionalFilterCallback<D, L, M, T> =
    fn(&StreamContext, &ImageWindow<'_, T, L>, &mut ImageOutputView<'_, D, M>) -> Result<()>;

type TypedMaskedEdgeDirectionalFilterCallback<D, L, M, T> = fn(
    &StreamContext,
    &ImageWindow<'_, T, L>,
    &mut ImageOutputView<'_, D, M>,
    MaskSize,
) -> Result<()>;
