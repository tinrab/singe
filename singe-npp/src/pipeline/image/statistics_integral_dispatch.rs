use crate::{
    context::StreamContext,
    error::Result,
    image::{
        statistics,
        view::{C1, ImageView, ImageViewMut},
    },
};

use super::ImagePipeline;

pub trait IntegralImage<U> {
    fn integral_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C1>,
        destination: &mut ImageViewMut<'_, U, C1>,
        value: U,
    ) -> Result<()>;
}

impl<'a> IntegralImage<i32> for ImagePipeline<'a, u8, C1> {
    fn integral_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C1>,
        destination: &mut ImageViewMut<'_, i32, C1>,
        value: i32,
    ) -> Result<()> {
        statistics::integral_to_i32(stream_context, source, destination, value)
    }
}

impl<'a> IntegralImage<f32> for ImagePipeline<'a, u8, C1> {
    fn integral_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C1>,
        destination: &mut ImageViewMut<'_, f32, C1>,
        value: f32,
    ) -> Result<()> {
        statistics::integral_to_f32(stream_context, source, destination, value)
    }
}

pub trait SquaredIntegralImage<I, S> {
    fn squared_integral_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C1>,
        integral: &mut ImageViewMut<'_, I, C1>,
        squared: &mut ImageViewMut<'_, S, C1>,
        value: I,
        squared_value: S,
    ) -> Result<()>;
}

impl<'a> SquaredIntegralImage<i32, i32> for ImagePipeline<'a, u8, C1> {
    fn squared_integral_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C1>,
        integral: &mut ImageViewMut<'_, i32, C1>,
        squared: &mut ImageViewMut<'_, i32, C1>,
        value: i32,
        squared_value: i32,
    ) -> Result<()> {
        statistics::squared_integral_to_i32_i32(
            stream_context,
            source,
            integral,
            squared,
            value,
            squared_value,
        )
    }
}

impl<'a> SquaredIntegralImage<i32, f64> for ImagePipeline<'a, u8, C1> {
    fn squared_integral_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C1>,
        integral: &mut ImageViewMut<'_, i32, C1>,
        squared: &mut ImageViewMut<'_, f64, C1>,
        value: i32,
        squared_value: f64,
    ) -> Result<()> {
        statistics::squared_integral_to_i32_f64(
            stream_context,
            source,
            integral,
            squared,
            value,
            squared_value,
        )
    }
}

impl<'a> SquaredIntegralImage<f32, f64> for ImagePipeline<'a, u8, C1> {
    fn squared_integral_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C1>,
        integral: &mut ImageViewMut<'_, f32, C1>,
        squared: &mut ImageViewMut<'_, f64, C1>,
        value: f32,
        squared_value: f64,
    ) -> Result<()> {
        statistics::squared_integral_to_f32_f64(
            stream_context,
            source,
            integral,
            squared,
            value,
            squared_value,
        )
    }
}
