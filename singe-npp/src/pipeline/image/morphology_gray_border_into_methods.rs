use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageView, ImageViewMut},
    types::{BorderType, Point, Size},
};

use super::super::super::{ImagePipeline, morphology_traits::GrayMaskMorphologyBorderImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
{
    pub fn gray_dilate_border_into<M>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, C1>,
        mask: &[M],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>
    where
        Self: GrayMaskMorphologyBorderImage<T, M>,
    {
        <Self as GrayMaskMorphologyBorderImage<T, M>>::gray_dilate_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            mask,
            mask_size,
            anchor,
            border_type,
        )
    }

    pub fn gray_erode_border_into<M>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, C1>,
        mask: &[M],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>
    where
        Self: GrayMaskMorphologyBorderImage<T, M>,
    {
        <Self as GrayMaskMorphologyBorderImage<T, M>>::gray_erode_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            mask,
            mask_size,
            anchor,
            border_type,
        )
    }
}
