use crate::{
    context::StreamContext,
    error::Result,
    image::view::{
        C4, ImageOutputView, ImageView, ImageViewMut, PlanarImageOutputView, PlanarImageView,
        PlanarImageViewMut,
    },
    pipeline::image::{
        ColorTwistBatchConstantsMatrix4, ColorTwistConstants4, ColorTwistMatrix, ColorTwistMatrix4,
    },
};

#[macro_use]
mod batch_dispatch;

#[macro_use]
mod static_dispatch;
#[macro_use]
mod static_twist_dispatch;

pub use crate::pipeline::image::{color_gamma_dispatch::*, color_gray_dispatch::*};

pub type SameLayoutColorOperation<L> = for<'source, 'destination> fn(
    &StreamContext,
    &ImageView<'source, u8, L>,
    &mut ImageOutputView<'destination, u8, L>,
) -> Result<()>;

/// # Safety
/// Complete borrowed native work on every path and uphold the operation's declared write coverage.
pub unsafe trait ColorTwistImage<T, L> {
    fn color_twist_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        twist: ColorTwistMatrix,
    ) -> Result<()>;

    fn color_twist_image_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, L>,
        twist: ColorTwistMatrix,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and uphold the operation's declared write coverage.
pub unsafe trait PlanarColorTwistImage<T> {
    fn color_twist_planar_image(
        stream_context: &StreamContext,
        source: &PlanarImageView<'_, T, 3>,
        destination: &mut PlanarImageOutputView<'_, T, 3>,
        twist: ColorTwistMatrix,
    ) -> Result<()>;

    fn color_twist_planar_image_in_place(
        stream_context: &StreamContext,
        image: &mut PlanarImageViewMut<'_, T, 3>,
        twist: ColorTwistMatrix,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and uphold the operation's declared write coverage.
pub unsafe trait ColorTwistWithConstantsImage<T> {
    fn color_twist_with_constants_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C4>,
        destination: &mut ImageOutputView<'_, T, C4>,
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

/// # Safety
/// Complete borrowed native work on every path and uphold the operation's declared write coverage.
pub unsafe trait ColorTwistBatchImage<T, L> {
    fn color_twist_batch_image(
        stream_context: &StreamContext,
        sources: &[ImageView<'_, T, L>],
        twists: &[ColorTwistMatrix],
        destinations: &mut [ImageOutputView<'_, T, L>],
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

/// # Safety
/// Complete borrowed native work on every path and uphold the operation's declared write coverage.
pub unsafe trait ColorTwistBatchWithConstantsImage<T> {
    fn color_twist_batch_with_constants_image(
        stream_context: &StreamContext,
        sources: &[ImageView<'_, T, C4>],
        twists: &[ColorTwistBatchConstantsMatrix4],
        destinations: &mut [ImageOutputView<'_, T, C4>],
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

pub type SameLayoutColorConvert<T, L> = for<'source, 'destination> unsafe fn(
    &StreamContext,
    &ImageView<'source, T, L>,
    &mut ImageOutputView<'destination, T, L>,
) -> Result<()>;

pub type DifferentLayoutColorConvert<T, S, D> = for<'source, 'destination> unsafe fn(
    &StreamContext,
    &ImageView<'source, T, S>,
    &mut ImageOutputView<'destination, T, D>,
) -> Result<()>;

pub type DifferentLayoutColorConvertConstantAlpha<T, S> =
    for<'source, 'destination> unsafe fn(
        &StreamContext,
        &ImageView<'source, T, S>,
        &mut ImageOutputView<'destination, T, C4>,
        u8,
    ) -> Result<()>;

#[macro_use]
mod twist_dispatch_macros;
