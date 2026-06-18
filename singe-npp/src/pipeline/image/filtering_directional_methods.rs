use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::{ImageBacking, ImagePipeline, filtering::*};

#[path = "filtering_directional_into_methods.rs"]
mod into_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: EdgeDirectionalFilterImage<T, L>,
{
    pub fn filter_prewitt_horizontal(self) -> Result<Self> {
        self.edge_directional_filter(
            <Self as EdgeDirectionalFilterImage<T, L>>::filter_prewitt_horizontal_image,
        )
    }

    pub fn filter_prewitt_vertical(self) -> Result<Self> {
        self.edge_directional_filter(
            <Self as EdgeDirectionalFilterImage<T, L>>::filter_prewitt_vertical_image,
        )
    }

    pub fn filter_roberts_down(self) -> Result<Self> {
        self.edge_directional_filter(
            <Self as EdgeDirectionalFilterImage<T, L>>::filter_roberts_down_image,
        )
    }

    pub fn filter_roberts_up(self) -> Result<Self> {
        self.edge_directional_filter(
            <Self as EdgeDirectionalFilterImage<T, L>>::filter_roberts_up_image,
        )
    }

    pub fn filter_sobel_horizontal(self) -> Result<Self> {
        self.edge_directional_filter(
            <Self as EdgeDirectionalFilterImage<T, L>>::filter_sobel_horizontal_image,
        )
    }

    pub fn filter_sobel_vertical(self) -> Result<Self> {
        self.edge_directional_filter(
            <Self as EdgeDirectionalFilterImage<T, L>>::filter_sobel_vertical_image,
        )
    }

    fn edge_directional_filter(
        self,
        filter: fn(&StreamContext, &ImageView<'_, T, L>, &mut ImageViewMut<'_, T, L>) -> Result<()>,
    ) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            filter(self.stream_context, &source, &mut destination_view)?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
