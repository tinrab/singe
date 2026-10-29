use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageWindow},
    },
    pipeline::{
        ImageBacking,
        image::{ImagePipeline, filtering::EdgeDirectionalFilterImage},
    },
    types::{Point, Size},
};

mod into_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: EdgeDirectionalFilterImage<T, L>,
{
    /// Applies the native horizontal Prewitt filter.
    /// Allocates a new owned image of width minus two and height minus two, excluding the
    /// one-pixel outer edge. The input must be at least 3 by 3 pixels; the helper selects a
    /// centered 3-by-3 valid source window for this operation.
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
    pub fn filter_prewitt_horizontal(self) -> Result<Self> {
        self.edge_directional_filter(
            <Self as EdgeDirectionalFilterImage<T, L>>::filter_prewitt_horizontal_image,
        )
    }

    /// Applies the native vertical Prewitt filter.
    /// Allocates a new owned image of width minus two and height minus two, excluding the
    /// one-pixel outer edge. The input must be at least 3 by 3 pixels; the helper selects a
    /// centered 3-by-3 valid source window for this operation.
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
    pub fn filter_prewitt_vertical(self) -> Result<Self> {
        self.edge_directional_filter(
            <Self as EdgeDirectionalFilterImage<T, L>>::filter_prewitt_vertical_image,
        )
    }

    /// Applies the native down Roberts filter.
    /// Allocates a new owned image of width minus two and height minus two, excluding the
    /// one-pixel outer edge. The input must be at least 3 by 3 pixels; the helper selects a
    /// centered 3-by-3 valid source window for this operation.
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
    pub fn filter_roberts_down(self) -> Result<Self> {
        self.edge_directional_filter(
            <Self as EdgeDirectionalFilterImage<T, L>>::filter_roberts_down_image,
        )
    }

    /// Applies the native up Roberts filter.
    /// Allocates a new owned image of width minus two and height minus two, excluding the
    /// one-pixel outer edge. The input must be at least 3 by 3 pixels; the helper selects a
    /// centered 3-by-3 valid source window for this operation.
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
    pub fn filter_roberts_up(self) -> Result<Self> {
        self.edge_directional_filter(
            <Self as EdgeDirectionalFilterImage<T, L>>::filter_roberts_up_image,
        )
    }

    /// Applies the native horizontal Sobel filter.
    /// Allocates a new owned image of width minus two and height minus two, excluding the
    /// one-pixel outer edge. The input must be at least 3 by 3 pixels; the helper selects a
    /// centered 3-by-3 valid source window for this operation.
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
    pub fn filter_sobel_horizontal(self) -> Result<Self> {
        self.edge_directional_filter(
            <Self as EdgeDirectionalFilterImage<T, L>>::filter_sobel_horizontal_image,
        )
    }

    /// Applies the native vertical Sobel filter.
    /// Allocates a new owned image of width minus two and height minus two, excluding the
    /// one-pixel outer edge. The input must be at least 3 by 3 pixels; the helper selects a
    /// centered 3-by-3 valid source window for this operation.
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
    pub fn filter_sobel_vertical(self) -> Result<Self> {
        self.edge_directional_filter(
            <Self as EdgeDirectionalFilterImage<T, L>>::filter_sobel_vertical_image,
        )
    }

    fn edge_directional_filter(self, filter: EdgeDirectionalFilterCallback<L, T>) -> Result<Self> {
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

type EdgeDirectionalFilterCallback<L, T> =
    fn(&StreamContext, &ImageWindow<'_, T, L>, &mut ImageOutputView<'_, T, L>) -> Result<()>;
