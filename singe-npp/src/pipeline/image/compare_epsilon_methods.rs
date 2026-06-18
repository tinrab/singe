use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ChannelLayout, ImageView, MaskViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::{
    CompareEqualEpsilonConstantImage, CompareEqualEpsilonImage, ImagePipeline as Pipeline,
};

impl<'a, L> Pipeline<'a, f32, L>
where
    L: ChannelLayout,
    Self: CompareEqualEpsilonImage<L>,
{
    pub fn compare_equal_epsilon_into(
        stream_context: &StreamContext,
        source1: &ImageView<'_, f32, L>,
        source2: &ImageView<'_, f32, L>,
        destination: &mut MaskViewMut<'_>,
        epsilon: f32,
    ) -> Result<()> {
        <Self as CompareEqualEpsilonImage<L>>::compare_equal_epsilon_image(
            stream_context,
            source1,
            source2,
            destination,
            epsilon,
        )
    }
}

impl<'a, L> Pipeline<'a, f32, L>
where
    L: ChannelLayout,
    Workspace: ImageAllocator<u8, C1>,
    Self: CompareEqualEpsilonImage<L>,
{
    pub fn compare_equal_epsilon(
        self,
        other: &ImageView<'_, f32, L>,
        epsilon: f32,
    ) -> Result<Pipeline<'a, u8, C1>> {
        let mut destination = self.workspace.image::<u8, C1>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as CompareEqualEpsilonImage<L>>::compare_equal_epsilon_image(
                self.stream_context,
                &source,
                other,
                &mut destination_view,
                epsilon,
            )?;
        }

        Ok(Pipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: super::super::ImageBacking::Owned(destination),
        })
    }
}

impl<'a, L> Pipeline<'a, f32, L>
where
    L: ChannelLayout,
    Self: CompareEqualEpsilonConstantImage<L>,
{
    pub fn compare_equal_epsilon_constant_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, f32, L>,
        constant: <Self as CompareEqualEpsilonConstantImage<L>>::Constant,
        destination: &mut MaskViewMut<'_>,
        epsilon: f32,
    ) -> Result<()> {
        <Self as CompareEqualEpsilonConstantImage<L>>::compare_equal_epsilon_constant_image(
            stream_context,
            source,
            constant,
            destination,
            epsilon,
        )
    }
}

impl<'a, L> Pipeline<'a, f32, L>
where
    L: ChannelLayout,
    Workspace: ImageAllocator<u8, C1>,
    Self: CompareEqualEpsilonConstantImage<L>,
{
    pub fn compare_equal_epsilon_constant(
        self,
        constant: <Self as CompareEqualEpsilonConstantImage<L>>::Constant,
        epsilon: f32,
    ) -> Result<Pipeline<'a, u8, C1>> {
        let mut destination = self.workspace.image::<u8, C1>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as CompareEqualEpsilonConstantImage<L>>::compare_equal_epsilon_constant_image(
                self.stream_context,
                &source,
                constant,
                &mut destination_view,
                epsilon,
            )?;
        }

        Ok(Pipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: super::super::ImageBacking::Owned(destination),
        })
    }
}
