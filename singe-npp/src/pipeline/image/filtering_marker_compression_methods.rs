use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{C1, ImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
};

use super::super::{CompressedMarkerLabels, CopyImage, ImageBacking, ImagePipeline};

impl<'a> ImagePipeline<'a, u32, C1>
where
    Workspace: ImageAllocator<u32, C1>,
    Self: CopyImage<u32, C1>,
{
    pub fn compress_marker_labels_uf_batch(
        stream_context: &StreamContext,
        source_destinations: &mut [ImageViewMut<'_, u32, C1>],
    ) -> Result<DeviceMemory<u32>> {
        let mut new_max_label_ids = DeviceMemory::create(source_destinations.len())?;
        filtering::compress_marker_labels_uf_batch_u32_c1_in_place(
            stream_context,
            source_destinations,
            &mut new_max_label_ids,
        )?;
        Ok(new_max_label_ids)
    }

    pub fn compress_marker_labels_uf_batch_advanced(
        stream_context: &StreamContext,
        source_destinations: &mut [ImageViewMut<'_, u32, C1>],
    ) -> Result<DeviceMemory<u32>> {
        let mut new_max_label_ids = DeviceMemory::create(source_destinations.len())?;
        filtering::compress_marker_labels_uf_batch_u32_c1_in_place_advanced(
            stream_context,
            source_destinations,
            &mut new_max_label_ids,
        )?;
        Ok(new_max_label_ids)
    }

    pub fn compress_marker_labels_uf(
        mut self,
        starting_number: i32,
    ) -> Result<CompressedMarkerLabels<'a>> {
        let new_max_label_id = match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                filtering::compress_marker_labels_uf_u32_c1_in_place(
                    self.stream_context,
                    &mut image_view,
                    starting_number,
                )?
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = self.workspace.image::<u32, C1>(source.size())?;
                let mut destination_view = destination.view_mut()?;
                <Self as CopyImage<u32, C1>>::copy(
                    self.stream_context,
                    source,
                    &mut destination_view,
                )?;
                let new_max_label_id = filtering::compress_marker_labels_uf_u32_c1_in_place(
                    self.stream_context,
                    &mut destination_view,
                    starting_number,
                )?;
                self.backing = ImageBacking::Owned(destination);
                new_max_label_id
            }
        };

        Ok(CompressedMarkerLabels {
            labels: self,
            new_max_label_id,
        })
    }
}
