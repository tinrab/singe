use crate::{
    error::{Error, Result},
    image::view::ChannelLayout,
    pipeline::{ImageAllocator, Workspace},
    types::{ConnectedRegion, ImageNormalization, Point},
};

use super::{CopyImage, ImagePipeline, filtering::FloodFillImage};

#[path = "filtering_flood_range_methods.rs"]
mod range_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: CopyImage<T, L> + FloodFillImage<T, L>,
    <Self as FloodFillImage<T, L>>::Value: Copy,
{
    pub fn flood_fill(
        self,
        seed: Point,
        new_value: <Self as FloodFillImage<T, L>>::Value,
        norm: ImageNormalization,
        connected_region: Option<&mut ConnectedRegion>,
    ) -> Result<Self> {
        self.flood_fill_operation(
            seed,
            new_value,
            None,
            None,
            None,
            norm,
            connected_region,
            |stream_context,
             image,
             seed,
             new_value,
             _min,
             _max,
             _boundary_value,
             norm,
             connected_region| {
                <Self as FloodFillImage<T, L>>::flood_fill_image(
                    stream_context,
                    image,
                    seed,
                    new_value,
                    norm,
                    connected_region,
                )
            },
        )
    }

    pub fn flood_fill_boundary(
        self,
        seed: Point,
        new_value: <Self as FloodFillImage<T, L>>::Value,
        boundary_value: <Self as FloodFillImage<T, L>>::Value,
        norm: ImageNormalization,
        connected_region: Option<&mut ConnectedRegion>,
    ) -> Result<Self> {
        self.flood_fill_operation(
            seed,
            new_value,
            None,
            None,
            Some(boundary_value),
            norm,
            connected_region,
            |stream_context,
             image,
             seed,
             new_value,
             _min,
             _max,
             boundary_value,
             norm,
             connected_region| {
                <Self as FloodFillImage<T, L>>::flood_fill_boundary_image(
                    stream_context,
                    image,
                    seed,
                    new_value,
                    boundary_value.ok_or(Error::MissingParameter {
                        name: "boundary_value".into(),
                    })?,
                    norm,
                    connected_region,
                )
            },
        )
    }
}
