use crate::{
    context::StreamContext,
    error::Result,
    image::{
        statistics,
        view::{C1, ImageView, ImageViewMut},
    },
};

use super::super::ImagePipeline;

impl<'a> ImagePipeline<'a, u8, C1> {
    pub fn squared_integral_to_i32_i32_into(
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

    pub fn squared_integral_to_i32_f64_into(
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

    pub fn squared_integral_to_f32_f64_into(
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
