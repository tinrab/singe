use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::{BorderType, Point, Size},
};

use super::{ImageBacking, ImagePipeline, filtering::*};

#[path = "filtering_typed_edge_to_methods.rs"]
mod typed_edge_to_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn filter_wiener_border_into<const CHANNELS: usize>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: Size,
        anchor: Point,
        noise: &mut [f32; CHANNELS],
        border_type: BorderType,
    ) -> Result<()>
    where
        Self: WienerBorderFilterImage<T, L, CHANNELS>,
    {
        <Self as WienerBorderFilterImage<T, L, CHANNELS>>::filter_wiener_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            anchor,
            noise,
            border_type,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: AdaptiveBoxThresholdBorderImage<T, L>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn filter_threshold_adaptive_box_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: Size,
        delta: f32,
        value_greater_than: T,
        value_less_or_equal: T,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as AdaptiveBoxThresholdBorderImage<T, L>>::filter_threshold_adaptive_box_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            delta,
            value_greater_than,
            value_less_or_equal,
            border_type,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
{
    pub fn filter_wiener_border<const CHANNELS: usize>(
        self,
        source_offset: Point,
        mask_size: Size,
        anchor: Point,
        noise: &mut [f32; CHANNELS],
        border_type: BorderType,
    ) -> Result<Self>
    where
        Self: WienerBorderFilterImage<T, L, CHANNELS>,
    {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as WienerBorderFilterImage<T, L, CHANNELS>>::filter_wiener_border_image(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                mask_size,
                anchor,
                noise,
                border_type,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
