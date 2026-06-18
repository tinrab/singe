use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageView, ImageViewMut},
};

use super::{
    super::statistics::{ContiguousImage, ImageSquaredIntegral},
    ImagePipeline, SquaredIntegralImage, integral_size,
};

impl<'a> ImagePipeline<'a, u8, C1> {
    pub fn squared_integral_to_into<I, S>(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, C1>,
        integral: &mut ImageViewMut<'_, I, C1>,
        squared: &mut ImageViewMut<'_, S, C1>,
        value: I,
        squared_value: S,
    ) -> Result<()>
    where
        I: Copy,
        S: Copy,
        Self: SquaredIntegralImage<I, S>,
    {
        <Self as SquaredIntegralImage<I, S>>::squared_integral_image(
            stream_context,
            source,
            integral,
            squared,
            value,
            squared_value,
        )
    }

    pub fn squared_integral_to<I, S>(
        self,
        value: I,
        squared_value: S,
    ) -> Result<ImageSquaredIntegral<I, S>>
    where
        I: Copy,
        S: Copy,
        Self: SquaredIntegralImage<I, S>,
    {
        let size = integral_size(self.size())?;
        let mut integral = ContiguousImage::<I, C1>::create(size)?;
        let mut squared = ContiguousImage::<S, C1>::create(size)?;

        {
            let source = self.view()?;
            let mut integral_view = integral.view_mut()?;
            let mut squared_view = squared.view_mut()?;
            <Self as SquaredIntegralImage<I, S>>::squared_integral_image(
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
