use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::image::{ImageBacking, ImagePipeline},
    types::{BorderType, MaskSize, Point},
};

mod extended_methods;
mod scharr_methods;
mod sobel_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub(crate) fn typed_edge_directional_border_filter<D, M>(
        self,
        source_offset: Point,
        border_type: BorderType,
        filter: TypedEdgeDirectionalBorderFilterCallback<D, L, M, T>,
    ) -> Result<ImagePipeline<'a, D, M>>
    where
        D: Copy + SupportedImage<M>,
        M: ChannelLayout,
    {
        self.ensure_eager()?;
        let source = self.view()?;
        let output_size = source.window_from(source_offset)?.size();
        let mut destination =
            Image::<D, M>::create_uninit(self.stream_context.cuda_context(), output_size)?;

        {
            let mut destination_view = destination.output()?;
            filter(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                border_type,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }

    pub(crate) fn typed_masked_edge_directional_border_filter<D, M>(
        self,
        source_offset: Point,
        mask_size: MaskSize,
        border_type: BorderType,
        filter: TypedMaskedEdgeDirectionalBorderFilterCallback<D, L, M, T>,
    ) -> Result<ImagePipeline<'a, D, M>>
    where
        D: Copy + SupportedImage<M>,
        M: ChannelLayout,
    {
        self.ensure_eager()?;
        let source = self.view()?;
        let output_size = source.window_from(source_offset)?.size();
        let mut destination =
            Image::<D, M>::create_uninit(self.stream_context.cuda_context(), output_size)?;

        {
            let mut destination_view = destination.output()?;
            filter(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                mask_size,
                border_type,
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

type TypedEdgeDirectionalBorderFilterCallback<D, L, M, T> = fn(
    &StreamContext,
    &ImageView<'_, T, L>,
    Point,
    &mut ImageOutputView<'_, D, M>,
    BorderType,
) -> Result<()>;

type TypedMaskedEdgeDirectionalBorderFilterCallback<D, L, M, T> = fn(
    &StreamContext,
    &ImageView<'_, T, L>,
    Point,
    &mut ImageOutputView<'_, D, M>,
    MaskSize,
    BorderType,
) -> Result<()>;
