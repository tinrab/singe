use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        view::{AC4, C3, ImageView, ImageViewMut},
    },
    types::{ColorSpace, Point},
};

use super::ImagePipeline;

pub trait Uyvp10uToRgbImage<T, L> {
    type Alpha;

    fn convert(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        color_space: ColorSpace,
        alpha: Option<Self::Alpha>,
    ) -> Result<()>;
}

impl Uyvp10uToRgbImage<u8, C3> for ImagePipeline<'_, u8, C3> {
    type Alpha = ();

    fn convert(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, u8, C3>,
        color_space: ColorSpace,
        _alpha: Option<Self::Alpha>,
    ) -> Result<()> {
        color::uyvp_10u_to_rgb(
            stream_context,
            source,
            source_offset,
            destination,
            color_space,
        )
    }
}

impl Uyvp10uToRgbImage<u16, C3> for ImagePipeline<'_, u8, C3> {
    type Alpha = ();

    fn convert(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, u16, C3>,
        color_space: ColorSpace,
        _alpha: Option<Self::Alpha>,
    ) -> Result<()> {
        color::uyvp_10u_to_rgb(
            stream_context,
            source,
            source_offset,
            destination,
            color_space,
        )
    }
}

impl Uyvp10uToRgbImage<u16, AC4> for ImagePipeline<'_, u8, C3> {
    type Alpha = u16;

    fn convert(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C3>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, u16, AC4>,
        color_space: ColorSpace,
        alpha: Option<Self::Alpha>,
    ) -> Result<()> {
        color::uyvp_10u_to_rgba(
            stream_context,
            source,
            source_offset,
            destination,
            color_space,
            alpha.unwrap_or_default(),
        )
    }
}
