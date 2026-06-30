use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    types::{BorderType, Point, Size},
};

use super::super::{ImagePipeline, filtering::*};

#[path = "filtering_float_kernel_into_methods.rs"]
mod float_kernel_into_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: IntegerKernelFilterImage<T, L>,
{
    pub fn filter_kernel_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        kernel: &[i32],
        kernel_size: Size,
        anchor: Point,
        divisor: i32,
    ) -> Result<()> {
        <Self as IntegerKernelFilterImage<T, L>>::filter_kernel_image(
            stream_context,
            source,
            destination,
            kernel,
            kernel_size,
            anchor,
            divisor,
        )
    }

    pub fn filter_kernel_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        kernel: &[i32],
        kernel_size: Size,
        anchor: Point,
        divisor: i32,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as IntegerKernelFilterImage<T, L>>::filter_kernel_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            kernel,
            kernel_size,
            anchor,
            divisor,
            border_type,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: DirectionalBorderFilterImage<T, L>,
{
    pub fn filter_sharpen_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as DirectionalBorderFilterImage<T, L>>::filter_sharpen_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            border_type,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: UnsharpBorderFilterImage<T, L>,
{
    pub fn filter_unsharp_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        radius: f32,
        sigma: f32,
        weight: f32,
        threshold: f32,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as UnsharpBorderFilterImage<T, L>>::filter_unsharp_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            radius,
            sigma,
            weight,
            threshold,
            border_type,
        )
    }
}
