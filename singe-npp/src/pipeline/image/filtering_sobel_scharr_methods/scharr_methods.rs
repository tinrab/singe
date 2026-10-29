use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView, ImageWindow},
    },
    pipeline::image::{
        ImageBacking, ImagePipeline,
        filtering::{ScharrBorderFilterImage, ScharrFilterImage},
    },
    types::{BorderType, Point, Size},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: ScharrBorderFilterImage<T, L>,
{
    /// Applies the native horizontal Scharr filter with border control.
    /// Allocates a new owned image covering the source window from `source_offset` to the
    /// image's lower-right corner. Border handling uses `border_type`.
    /// For AC4, copies the corresponding source window first to preserve source alpha.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Selecting the source window, allocating output, constructing views, or preserving source
    /// alpha can also fail.
    pub fn filter_scharr_horizontal_border(
        self,
        source_offset: Point,
        border_type: BorderType,
    ) -> Result<Self> {
        self.scharr_border_filter(
            source_offset,
            border_type,
            <Self as ScharrBorderFilterImage<T, L>>::filter_scharr_horizontal_border_image,
        )
    }

    /// Applies the native vertical Scharr filter with border control.
    /// Allocates a new owned image covering the source window from `source_offset` to the
    /// image's lower-right corner. Border handling uses `border_type`.
    /// For AC4, copies the corresponding source window first to preserve source alpha.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Selecting the source window, allocating output, constructing views, or preserving source
    /// alpha can also fail.
    pub fn filter_scharr_vertical_border(
        self,
        source_offset: Point,
        border_type: BorderType,
    ) -> Result<Self> {
        self.scharr_border_filter(
            source_offset,
            border_type,
            <Self as ScharrBorderFilterImage<T, L>>::filter_scharr_vertical_border_image,
        )
    }

    fn scharr_border_filter(
        self,
        source_offset: Point,
        border_type: BorderType,
        filter: ScharrBorderFilterCallback<L, T>,
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

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: ScharrFilterImage<T, L>,
{
    /// Applies the native horizontal Scharr filter.
    /// Allocates a new owned image over the valid source window for the centered 3-by-3 mask.
    /// Each output dimension is the input dimension minus the mask dimension plus one;
    /// the source image must contain the entire mask.
    /// For AC4, copies the corresponding source window first to preserve source alpha.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Selecting the source window, allocating output, constructing views, or preserving source
    /// alpha can also fail.
    pub fn filter_scharr_horizontal(self) -> Result<Self> {
        self.scharr_filter(<Self as ScharrFilterImage<T, L>>::filter_scharr_horizontal_image)
    }

    /// Applies the native vertical Scharr filter.
    /// Allocates a new owned image over the valid source window for the centered 3-by-3 mask.
    /// Each output dimension is the input dimension minus the mask dimension plus one;
    /// the source image must contain the entire mask.
    /// For AC4, copies the corresponding source window first to preserve source alpha.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Selecting the source window, allocating output, constructing views, or preserving source
    /// alpha can also fail.
    pub fn filter_scharr_vertical(self) -> Result<Self> {
        self.scharr_filter(<Self as ScharrFilterImage<T, L>>::filter_scharr_vertical_image)
    }

    fn scharr_filter(self, filter: ScharrFilterCallback<L, T>) -> Result<Self> {
        self.ensure_eager()?;
        let source = self
            .view()?
            .valid_window(Size::new(3, 3), Point { x: 1, y: 1 })?;
        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), source.size())?;

        {
            let mut destination_view = destination.output()?;
            if L::IGNORES_ALPHA {
                destination_view.copy_from(&source.view())?;
            }
            filter(self.stream_context, &source, &mut destination_view)?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}

type ScharrBorderFilterCallback<L, T> = fn(
    &StreamContext,
    &ImageView<'_, T, L>,
    Point,
    &mut ImageOutputView<'_, T, L>,
    BorderType,
) -> Result<()>;

type ScharrFilterCallback<L, T> =
    fn(&StreamContext, &ImageWindow<'_, T, L>, &mut ImageOutputView<'_, T, L>) -> Result<()>;
