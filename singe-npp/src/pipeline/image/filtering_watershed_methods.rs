use crate::{
    error::Result,
    image::view::{C1, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::{ImageNormalization, WatershedSegmentBoundaryType},
};

use super::{CopyImage, ImageBacking, ImagePipeline, filtering::WatershedSegmentImage};

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
    Workspace: ImageAllocator<T, C1>,
    Self: CopyImage<T, C1> + WatershedSegmentImage<T>,
{
    pub fn segment_watershed(
        self,
        marker_labels: Option<&mut ImageViewMut<'_, u32, C1>>,
        norm: ImageNormalization,
        boundary_type: WatershedSegmentBoundaryType,
    ) -> Result<Self> {
        let ImagePipeline {
            stream_context,
            workspace,
            backing,
        } = self;
        let mut image = match backing {
            ImageBacking::Owned(image) => image,
            ImageBacking::Borrowed(source) => {
                let mut destination = workspace.image::<T, C1>(source.size())?;
                let mut destination_view = destination.view_mut()?;
                <Self as CopyImage<T, C1>>::copy(stream_context, &source, &mut destination_view)?;
                destination
            }
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
