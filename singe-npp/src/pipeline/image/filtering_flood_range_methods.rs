use crate::{
    error::{Error, Result},
    image::view::ChannelLayout,
    pipeline::{ImageAllocator, Workspace},
    types::{ConnectedRegion, ImageNormalization, Point},
};

use super::{CopyImage, FloodFillImage, ImagePipeline};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: CopyImage<T, L> + FloodFillImage<T, L>,
    <Self as FloodFillImage<T, L>>::Value: Copy,
{
    pub fn flood_fill_range(
        self,
        seed: Point,
        min: <Self as FloodFillImage<T, L>>::Value,
        max: <Self as FloodFillImage<T, L>>::Value,
        new_value: <Self as FloodFillImage<T, L>>::Value,
        norm: ImageNormalization,
        connected_region: Option<&mut ConnectedRegion>,
    ) -> Result<Self> {
        self.flood_fill_operation(
            seed,
            new_value,
            Some(min),
            Some(max),
            None,
            norm,
            connected_region,
            |stream_context,
             image,
             seed,
             new_value,
             min,
             max,
             _boundary_value,
             norm,
             connected_region| {
                <Self as FloodFillImage<T, L>>::flood_fill_range_image(
                    stream_context,
                    image,
                    seed,
                    min.ok_or(Error::MissingParameter { name: "min".into() })?,
                    max.ok_or(Error::MissingParameter { name: "max".into() })?,
                    new_value,
                    norm,
                    connected_region,
                )
            },
        )
    }

    pub fn flood_fill_range_boundary(
        self,
        seed: Point,
        min: <Self as FloodFillImage<T, L>>::Value,
        max: <Self as FloodFillImage<T, L>>::Value,
        new_value: <Self as FloodFillImage<T, L>>::Value,
        boundary_value: <Self as FloodFillImage<T, L>>::Value,
        norm: ImageNormalization,
        connected_region: Option<&mut ConnectedRegion>,
    ) -> Result<Self> {
        self.flood_fill_operation(
            seed,
            new_value,
            Some(min),
            Some(max),
            Some(boundary_value),
            norm,
            connected_region,
            |stream_context,
             image,
             seed,
             new_value,
             min,
             max,
             boundary_value,
             norm,
             connected_region| {
                <Self as FloodFillImage<T, L>>::flood_fill_range_boundary_image(
                    stream_context,
                    image,
                    seed,
                    min.ok_or(Error::MissingParameter { name: "min".into() })?,
                    max.ok_or(Error::MissingParameter { name: "max".into() })?,
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
