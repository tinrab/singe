use crate::{
    context::StreamContext,
    error::Result,
    image::{
        statistics,
        view::{C1, ImageView, ImageViewMut},
    },
};

#[path = "statistics_squared_integral_into_methods.rs"]
mod into_methods;

use super::{
    ImagePipeline,
    statistics::{ContiguousImage, ImageSquaredIntegral},
    statistics_integral_methods::integral_size,
};

impl<'a> ImagePipeline<'a, u8, C1> {
    pub fn squared_integral_to_i32_i32(
        mut self,
        value: i32,
        squared_value: i32,
    ) -> Result<ImageSquaredIntegral<i32, i32>> {
        self.squared_integral(
            value,
            squared_value,
            statistics::squared_integral_to_i32_i32,
        )
    }

    pub fn squared_integral_to_i32_f64(
        mut self,
        value: i32,
        squared_value: f64,
    ) -> Result<ImageSquaredIntegral<i32, f64>> {
        self.squared_integral(
            value,
            squared_value,
            statistics::squared_integral_to_i32_f64,
        )
    }

    pub fn squared_integral_to_f32_f64(
        mut self,
        value: f32,
        squared_value: f64,
    ) -> Result<ImageSquaredIntegral<f32, f64>> {
        self.squared_integral(
            value,
            squared_value,
            statistics::squared_integral_to_f32_f64,
        )
    }

    fn squared_integral<T, S>(
        &mut self,
        value: T,
        squared_value: S,
        operation: fn(
            &StreamContext,
            &ImageView<'_, u8, C1>,
            &mut ImageViewMut<'_, T, C1>,
            &mut ImageViewMut<'_, S, C1>,
            T,
            S,
        ) -> Result<()>,
    ) -> Result<ImageSquaredIntegral<T, S>>
    where
        T: Copy,
        S: Copy,
    {
        let size = integral_size(self.size())?;
        let mut integral = ContiguousImage::<T, C1>::create(size)?;
        let mut squared = ContiguousImage::<S, C1>::create(size)?;

        {
            let source = self.view()?;
            let mut integral_view = integral.view_mut()?;
            let mut squared_view = squared.view_mut()?;
            operation(
                self.stream_context,
                &source,
                &mut integral_view,
                &mut squared_view,
                value,
                squared_value,
            )?;
        }

        Ok(ImageSquaredIntegral { integral, squared })
    }
}
