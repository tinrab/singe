use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageView, ImageViewMut},
    types::{BorderType, MaskSize, Point},
};

#[path = "filtering_border_traits.rs"]
mod border_traits;

pub use border_traits::*;

pub trait MaskedKernelFilterImage<T, L> {
    fn filter_high_pass_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: MaskSize,
    ) -> Result<()>;

    fn filter_low_pass_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: MaskSize,
    ) -> Result<()>;

    fn filter_gauss_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: MaskSize,
    ) -> Result<()>;

    fn filter_laplace_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: MaskSize,
    ) -> Result<()>;
}

pub trait HighLowGaussFilterImage<T, L> {
    fn filter_high_pass_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: MaskSize,
    ) -> Result<()>;

    fn filter_low_pass_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: MaskSize,
    ) -> Result<()>;

    fn filter_gauss_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: MaskSize,
    ) -> Result<()>;
}

pub trait AdvancedGaussFilterImage<T, L> {
    fn filter_gauss_advanced_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        kernel: &[f32],
    ) -> Result<()>;

    fn filter_gauss_advanced_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        kernel: &[f32],
        border_type: BorderType,
    ) -> Result<()>;
}
