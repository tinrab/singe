use crate::{
    error::Result,
    image::{
        Image, SupportedImage,
        view::{C1, ImageOutputView},
    },
    pipeline::image::{ImageBacking, ImagePipeline, filtering::WatershedSegmentImage},
    types::{ImageNormalization, WatershedSegmentBoundaryType},
};

impl<T> ImagePipeline<'_, T, C1>
where
    T: Copy + SupportedImage<C1>,
    Self: WatershedSegmentImage<T>,
{
    /// Segments the C1 image with the native watershed algorithm.
    ///
    /// The provided implementations accept `u8` or `u16` images. Reuses an owned image in
    /// place or copies borrowed input into a new owned image. Optional `marker_labels`
    /// receives a complete source-sized `u32` label image and requires writable capacity
    /// without initialized old labels. `ImageNormalization::L1` selects four-way
    /// connectivity and `ImageNormalization::Infinity` selects eight-way connectivity;
    /// `boundary_type` selects how segment boundaries are rendered. Owns temporary device
    /// scratch, prepares participating regions, waits for producers on other streams,
    /// rejects capture, and returns after successful stream completion. Row padding
    /// remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, copying borrowed input, image view construction,
    /// CUDA context binding or region preparation, producer waits, workspace queries,
    /// scratch allocation, native segmentation, or synchronization fail. A supplied marker
    /// image must match source dimensions. NPP may reject connectivity or boundary selectors.
    pub fn segment_watershed(
        self,
        marker_labels: Option<&mut ImageOutputView<'_, u32, C1>>,
        norm: ImageNormalization,
        boundary_type: WatershedSegmentBoundaryType,
    ) -> Result<Self> {
        self.ensure_eager()?;
        let ImagePipeline {
            stream_context,
            workspace,
            backing,
        } = self;
        let mut image = match backing {
            ImageBacking::Owned(image) => image,
            ImageBacking::Borrowed(source) => Image::<T, C1>::from_view(&source)?,
        };

        {
            let mut image_view = image.view_mut()?;
            <Self as WatershedSegmentImage<T>>::segment_watershed_image(
                stream_context,
                &mut image_view,
                marker_labels,
                norm,
                boundary_type,
            )?;
        }

        Ok(Self {
            stream_context,
            workspace,
            backing: ImageBacking::Owned(image),
        })
    }
}
