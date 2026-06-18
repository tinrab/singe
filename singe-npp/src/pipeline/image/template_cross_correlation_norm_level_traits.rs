use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageView, ImageViewMut},
};

pub trait CrossCorrelationNormLevelImage<T, L> {
    fn full(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;

    fn same(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;

    fn valid(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;
}

pub trait CrossCorrelationNormLevelScaledImage<T, L> {
    fn full(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;

    fn same(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;

    fn valid(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;
}

pub trait CrossCorrelationNormLevelToImage<T, D, L> {
    fn full(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, D, L>,
    ) -> Result<()>;

    fn same(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, D, L>,
    ) -> Result<()>;

    fn valid(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, D, L>,
    ) -> Result<()>;
}
