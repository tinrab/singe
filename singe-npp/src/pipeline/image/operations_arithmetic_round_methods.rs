use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::RoundMode,
};

use super::{ImageBacking, ImagePipeline, operation_shapes::*, operation_traits::*};

#[path = "operations_natural_logarithm_methods.rs"]
mod natural_logarithm_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: DivideRoundImage<T, L>,
{
    pub fn divide_round_into(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
        round_mode: RoundMode,
    ) -> Result<()> {
        <Self as DivideRoundImage<T, L>>::divide_round_image(
            stream_context,
            left,
            right,
            destination,
            scale_factor,
            round_mode,
        )
    }

    pub fn divide_round_in_place(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
        round_mode: RoundMode,
    ) -> Result<()> {
        <Self as DivideRoundImage<T, L>>::divide_round_image_in_place(
            stream_context,
            source,
            source_destination,
            scale_factor,
            round_mode,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: DivideRoundImage<T, L>,
{
    pub fn divide_round(
        self,
        other: &ImageView<'_, T, L>,
        scale_factor: i32,
        round_mode: RoundMode,
    ) -> Result<Self> {
        self.binary_arithmetic_round(
            other,
            scale_factor,
            round_mode,
            <Self as DivideRoundImage<T, L>>::divide_round_image,
            <Self as DivideRoundImage<T, L>>::divide_round_image_in_place,
        )
    }

    fn binary_arithmetic_round(
        mut self,
        other: &ImageView<'_, T, L>,
        scale_factor: i32,
        round_mode: RoundMode,
        operation: RoundedBinaryImageOperation<T, L>,
        operation_in_place: RoundedBinaryImageInPlaceOperation<T, L>,
    ) -> Result<Self> {
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                operation_in_place(
                    self.stream_context,
                    other,
                    &mut image_view,
                    scale_factor,
                    round_mode,
                )?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = self.workspace.image::<T, L>(source.size())?;
                let mut destination_view = destination.view_mut()?;
                operation(
                    self.stream_context,
                    source,
                    other,
                    &mut destination_view,
                    scale_factor,
                    round_mode,
                )?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}
