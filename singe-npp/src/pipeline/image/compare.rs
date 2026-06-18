use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ChannelLayout, ImageView, MaskViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::ComparisonOperation,
};

use super::{ImagePipeline, ImagePipeline as Pipeline};

#[path = "compare_dispatch.rs"]
mod dispatch;

pub(super) use dispatch::{
    CompareConstantImage, CompareEqualEpsilonConstantImage, CompareEqualEpsilonImage, CompareImage,
};

impl<'a, T, L> Pipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: CompareImage<T, L>,
{
    pub fn compare_into(
        stream_context: &StreamContext,
        source1: &ImageView<'_, T, L>,
        source2: &ImageView<'_, T, L>,
        destination: &mut MaskViewMut<'_>,
        operation: ComparisonOperation,
    ) -> Result<()> {
        <Self as CompareImage<T, L>>::compare_image(
            stream_context,
            source1,
            source2,
            destination,
            operation,
        )
    }
}

impl<'a, T, L> Pipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<u8, C1>,
    Self: CompareImage<T, L>,
{
    pub fn compare(
        self,
        other: &ImageView<'_, T, L>,
        operation: ComparisonOperation,
    ) -> Result<Pipeline<'a, u8, C1>> {
        let mut destination = self.workspace.image::<u8, C1>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as CompareImage<T, L>>::compare_image(
                self.stream_context,
                &source,
                other,
                &mut destination_view,
                operation,
            )?;
        }

        Ok(Pipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: super::ImageBacking::Owned(destination),
        })
    }
}

impl<'a, T, L> Pipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: CompareConstantImage<T, L>,
{
    pub fn compare_constant_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: <Self as CompareConstantImage<T, L>>::Constant,
        destination: &mut MaskViewMut<'_>,
        operation: ComparisonOperation,
    ) -> Result<()> {
        <Self as CompareConstantImage<T, L>>::compare_constant_image(
            stream_context,
            source,
            constant,
            destination,
            operation,
        )
    }
}

impl<'a, T, L> Pipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<u8, C1>,
    Self: CompareConstantImage<T, L>,
{
    pub fn compare_constant(
        self,
        constant: <Self as CompareConstantImage<T, L>>::Constant,
        operation: ComparisonOperation,
    ) -> Result<Pipeline<'a, u8, C1>> {
        let mut destination = self.workspace.image::<u8, C1>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as CompareConstantImage<T, L>>::compare_constant_image(
                self.stream_context,
                &source,
                constant,
                &mut destination_view,
                operation,
            )?;
        }

        Ok(Pipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: super::ImageBacking::Owned(destination),
        })
    }
}

#[path = "compare_epsilon_methods.rs"]
mod epsilon_methods;
