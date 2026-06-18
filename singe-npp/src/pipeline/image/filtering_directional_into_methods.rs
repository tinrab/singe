use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
};

use super::super::{ImagePipeline, filtering::EdgeDirectionalFilterImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: EdgeDirectionalFilterImage<T, L>,
{
    pub fn filter_prewitt_horizontal_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as EdgeDirectionalFilterImage<T, L>>::filter_prewitt_horizontal_image(
            stream_context,
            source,
            destination,
        )
    }

    pub fn filter_prewitt_vertical_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as EdgeDirectionalFilterImage<T, L>>::filter_prewitt_vertical_image(
            stream_context,
            source,
            destination,
        )
    }

    pub fn filter_roberts_down_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as EdgeDirectionalFilterImage<T, L>>::filter_roberts_down_image(
            stream_context,
            source,
            destination,
        )
    }

    pub fn filter_roberts_up_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as EdgeDirectionalFilterImage<T, L>>::filter_roberts_up_image(
            stream_context,
            source,
            destination,
        )
    }

    pub fn filter_sobel_horizontal_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as EdgeDirectionalFilterImage<T, L>>::filter_sobel_horizontal_image(
            stream_context,
            source,
            destination,
        )
    }

    pub fn filter_sobel_vertical_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as EdgeDirectionalFilterImage<T, L>>::filter_sobel_vertical_image(
            stream_context,
            source,
            destination,
        )
    }
}
