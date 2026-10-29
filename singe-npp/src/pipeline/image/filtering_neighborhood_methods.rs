use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageWindow},
    },
    pipeline::image::{
        ImageBacking, ImagePipeline,
        filtering::{MedianFilterImage, NeighborhoodFilterImage},
    },
    types::{Point, Size},
};

mod border_methods;
mod into_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: MedianFilterImage<T, L>,
{
    /// Applies a neighborhood median filter.
    /// Allocates a new owned image over the valid source window for the `mask_size` neighborhood at `anchor`.
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
    /// Mask dimensions must be positive and `anchor` must lie inside the mask.
    /// Native scratch-size queries and temporary scratch allocation can also fail.
    /// Selecting the source window, allocating output, constructing views, or preserving source
    /// alpha can also fail.
    pub fn filter_median(self, mask_size: Size, anchor: Point) -> Result<Self> {
        self.ensure_eager()?;
        let source = self.view()?.valid_window(mask_size, anchor)?;
        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), source.size())?;

        {
            let mut destination_view = destination.output()?;
            if L::IGNORES_ALPHA {
                destination_view.copy_from(&source.view())?;
            }
            <Self as MedianFilterImage<T, L>>::filter_median_image(
                self.stream_context,
                &source,
                &mut destination_view,
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

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: NeighborhoodFilterImage<T, L>,
{
    /// Applies a neighborhood maximum filter.
    /// Allocates a new owned image over the valid source window for the `mask_size` neighborhood at `anchor`.
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
    /// Mask dimensions must be positive and `anchor` must lie inside the mask.
    /// Selecting the source window, allocating output, constructing views, or preserving source
    /// alpha can also fail.
    pub fn filter_max(self, mask_size: Size, anchor: Point) -> Result<Self> {
        self.neighborhood_filter(
            mask_size,
            anchor,
            <Self as NeighborhoodFilterImage<T, L>>::filter_max_image,
        )
    }

    /// Applies a neighborhood minimum filter.
    /// Allocates a new owned image over the valid source window for the `mask_size` neighborhood at `anchor`.
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
    /// Mask dimensions must be positive and `anchor` must lie inside the mask.
    /// Selecting the source window, allocating output, constructing views, or preserving source
    /// alpha can also fail.
    pub fn filter_min(self, mask_size: Size, anchor: Point) -> Result<Self> {
        self.neighborhood_filter(
            mask_size,
            anchor,
            <Self as NeighborhoodFilterImage<T, L>>::filter_min_image,
        )
    }

    fn neighborhood_filter(
        self,
        mask_size: Size,
        anchor: Point,
        filter: NeighborhoodFilterCallback<L, T>,
    ) -> Result<Self> {
        self.ensure_eager()?;
        let source = self.view()?.valid_window(mask_size, anchor)?;
        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), source.size())?;

        {
            let mut destination_view = destination.output()?;
            if L::IGNORES_ALPHA {
                destination_view.copy_from(&source.view())?;
            }
            filter(
                self.stream_context,
                &source,
                &mut destination_view,
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

type NeighborhoodFilterCallback<L, T> = fn(
    &StreamContext,
    &ImageWindow<'_, T, L>,
    &mut ImageOutputView<'_, T, L>,
    Size,
    Point,
) -> Result<()>;
