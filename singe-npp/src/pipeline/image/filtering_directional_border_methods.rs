use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::{
        ImageBacking,
        image::{ImagePipeline, filtering::EdgeDirectionalBorderFilterImage},
    },
    types::{BorderType, Point},
};

mod into_methods;
mod roberts_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: EdgeDirectionalBorderFilterImage<T, L>,
{
    /// Applies the native horizontal Prewitt filter with border control.
    /// Allocates a new owned image covering the source window from `source_offset` to the
    /// full image's lower-right corner. Native border handling uses `border_type`.
    /// For AC4, copies the corresponding source pixels before filtering to preserve source alpha.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native filtering, or stream synchronization fail.
    /// Selecting the output window, allocating the image, constructing views, or preserving
    /// source alpha can also fail.
    pub fn filter_prewitt_horizontal_border(
        self,
        source_offset: Point,
        border_type: BorderType,
    ) -> Result<Self> {
        self.edge_directional_border_filter(
            source_offset,
            border_type,
            <Self as EdgeDirectionalBorderFilterImage<T, L>>::filter_prewitt_horizontal_border_image,
        )
    }

    /// Applies the native vertical Prewitt filter with border control.
    /// Allocates a new owned image covering the source window from `source_offset` to the
    /// full image's lower-right corner. Native border handling uses `border_type`.
    /// For AC4, copies the corresponding source pixels before filtering to preserve source alpha.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native filtering, or stream synchronization fail.
    /// Selecting the output window, allocating the image, constructing views, or preserving
    /// source alpha can also fail.
    pub fn filter_prewitt_vertical_border(
        self,
        source_offset: Point,
        border_type: BorderType,
    ) -> Result<Self> {
        self.edge_directional_border_filter(
            source_offset,
            border_type,
            <Self as EdgeDirectionalBorderFilterImage<T, L>>::filter_prewitt_vertical_border_image,
        )
    }

    /// Applies the native horizontal Sobel filter with border control.
    /// Allocates a new owned image covering the source window from `source_offset` to the
    /// full image's lower-right corner. Native border handling uses `border_type`.
    /// For AC4, copies the corresponding source pixels before filtering to preserve source alpha.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native filtering, or stream synchronization fail.
    /// Selecting the output window, allocating the image, constructing views, or preserving
    /// source alpha can also fail.
    pub fn filter_sobel_horizontal_border(
        self,
        source_offset: Point,
        border_type: BorderType,
    ) -> Result<Self> {
        self.edge_directional_border_filter(
            source_offset,
            border_type,
            <Self as EdgeDirectionalBorderFilterImage<T, L>>::filter_sobel_horizontal_border_image,
        )
    }

    /// Applies the native vertical Sobel filter with border control.
    /// Allocates a new owned image covering the source window from `source_offset` to the
    /// full image's lower-right corner. Native border handling uses `border_type`.
    /// For AC4, copies the corresponding source pixels before filtering to preserve source alpha.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native filtering, or stream synchronization fail.
    /// Selecting the output window, allocating the image, constructing views, or preserving
    /// source alpha can also fail.
    pub fn filter_sobel_vertical_border(
        self,
        source_offset: Point,
        border_type: BorderType,
    ) -> Result<Self> {
        self.edge_directional_border_filter(
            source_offset,
            border_type,
            <Self as EdgeDirectionalBorderFilterImage<T, L>>::filter_sobel_vertical_border_image,
        )
    }

    pub(crate) fn edge_directional_border_filter(
        self,
        source_offset: Point,
        border_type: BorderType,
        filter: EdgeDirectionalBorderFilterCallback<L, T>,
    ) -> Result<Self> {
        self.ensure_eager()?;
        let source = self.view()?;
        let output_source = source.window_from(source_offset)?;
        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), output_source.size())?;

        {
            let mut destination_view = destination.output()?;
            if L::IGNORES_ALPHA {
                destination_view.copy_from(&output_source.view())?;
            }
            filter(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
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

type EdgeDirectionalBorderFilterCallback<L, T> = fn(
    &StreamContext,
    &ImageView<'_, T, L>,
    Point,
    &mut ImageOutputView<'_, T, L>,
    BorderType,
) -> Result<()>;
