use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C3, C4, ImageView, ImageViewMut, PlanarImageView, PlanarImageViewMut},
    types::{ColorSpace, Point},
};

use super::{
    ColorTwistBatchConstantsMatrix4, ColorTwistConstants4, ColorTwistMatrix, ColorTwistMatrix4,
    ImagePipeline,
};

#[macro_use]
#[path = "color_batch_dispatch.rs"]
mod batch_dispatch;

#[macro_use]
#[path = "color_static_dispatch.rs"]
mod static_dispatch;
#[macro_use]
#[path = "color_static_twist_dispatch.rs"]
mod static_twist_dispatch;

pub(super) use super::color_gamma_dispatch::*;
pub(super) use super::color_gray_dispatch::*;

pub(super) type Uyvp10uToRgbOperation = fn(
    &StreamContext,
    &ImageView<'_, u8, C3>,
    Point,
    &mut ImageViewMut<'_, u8, C3>,
    ColorSpace,
) -> Result<()>;
pub(super) type SameLayoutColorOperation<L> = for<'source, 'destination> fn(
    &StreamContext,
    &ImageView<'source, u8, L>,
    &mut ImageViewMut<'destination, u8, L>,
) -> Result<()>;

#[path = "color_uyvp_dispatch.rs"]
mod uyvp_dispatch;

pub use uyvp_dispatch::Uyvp10uToRgbImage;

pub trait ColorTwistImage<T, L> {
    fn color_twist_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        twist: ColorTwistMatrix,
    ) -> Result<()>;

    fn color_twist_image_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, L>,
        twist: ColorTwistMatrix,
    ) -> Result<()>;
}

pub trait PlanarColorTwistImage<T> {
    fn color_twist_planar_image(
        stream_context: &StreamContext,
        source: &PlanarImageView<'_, T, 3>,
        destination: &mut PlanarImageViewMut<'_, T, 3>,
        twist: ColorTwistMatrix,
    ) -> Result<()>;

    fn color_twist_planar_image_in_place(
        stream_context: &StreamContext,
        image: &mut PlanarImageViewMut<'_, T, 3>,
        twist: ColorTwistMatrix,
    ) -> Result<()>;
}

pub trait ColorTwistWithConstantsImage<T> {
    fn color_twist_with_constants_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C4>,
        destination: &mut ImageViewMut<'_, T, C4>,
        twist: ColorTwistMatrix4,
        constants: ColorTwistConstants4,
    ) -> Result<()>;

    fn color_twist_with_constants_image_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, C4>,
        twist: ColorTwistMatrix4,
        constants: ColorTwistConstants4,
    ) -> Result<()>;
}

pub trait ColorTwistBatchImage<T, L> {
    fn color_twist_batch_image(
        stream_context: &StreamContext,
        sources: &[ImageView<'_, T, L>],
        twists: &[ColorTwistMatrix],
        destinations: &mut [ImageViewMut<'_, T, L>],
        min: f32,
        max: f32,
    ) -> Result<()>;

    fn color_twist_batch_image_in_place(
        stream_context: &StreamContext,
        images: &mut [ImageViewMut<'_, T, L>],
        twists: &[ColorTwistMatrix],
        min: f32,
        max: f32,
    ) -> Result<()>;
}

pub trait ColorTwistBatchWithConstantsImage<T> {
    fn color_twist_batch_with_constants_image(
        stream_context: &StreamContext,
        sources: &[ImageView<'_, T, C4>],
        twists: &[ColorTwistBatchConstantsMatrix4],
        destinations: &mut [ImageViewMut<'_, T, C4>],
        min: f32,
        max: f32,
    ) -> Result<()>;

    fn color_twist_batch_with_constants_image_in_place(
        stream_context: &StreamContext,
        images: &mut [ImageViewMut<'_, T, C4>],
        twists: &[ColorTwistBatchConstantsMatrix4],
        min: f32,
        max: f32,
    ) -> Result<()>;
}

pub(super) type SameLayoutColorConvert<T, L> = for<'source, 'destination> fn(
    &StreamContext,
    &ImageView<'source, T, L>,
    &mut ImageViewMut<'destination, T, L>,
) -> Result<()>;

pub(super) type DifferentLayoutColorConvert<T, S, D> = for<'source, 'destination> fn(
    &StreamContext,
    &ImageView<'source, T, S>,
    &mut ImageViewMut<'destination, T, D>,
) -> Result<()>;

pub(super) type DifferentLayoutColorConvertConstantAlpha<T, S> =
    for<'source, 'destination> fn(
        &StreamContext,
        &ImageView<'source, T, S>,
        &mut ImageViewMut<'destination, T, C4>,
        u8,
    ) -> Result<()>;

#[macro_use]
#[path = "color_twist_dispatch_macros.rs"]
mod twist_dispatch_macros;
