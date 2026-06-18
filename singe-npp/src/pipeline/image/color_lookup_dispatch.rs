use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageView, ImageViewMut},
};

use super::ImagePipeline;

pub use self::packed_traits::*;
pub use super::color_lookup_palette_dispatch::*;

pub trait LookupTableImage<T> {
    type TableValue;

    fn lookup_table_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageViewMut<'_, T, C1>,
        values: &DeviceMemory<Self::TableValue>,
        levels: &DeviceMemory<Self::TableValue>,
    ) -> Result<()>;

    fn lookup_table_image_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, C1>,
        values: &DeviceMemory<Self::TableValue>,
        levels: &DeviceMemory<Self::TableValue>,
    ) -> Result<()>;

    fn lookup_table_linear_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageViewMut<'_, T, C1>,
        values: &DeviceMemory<Self::TableValue>,
        levels: &DeviceMemory<Self::TableValue>,
    ) -> Result<()>;

    fn lookup_table_linear_image_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, C1>,
        values: &DeviceMemory<Self::TableValue>,
        levels: &DeviceMemory<Self::TableValue>,
    ) -> Result<()>;

    fn lookup_table_cubic_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageViewMut<'_, T, C1>,
        values: &DeviceMemory<Self::TableValue>,
        levels: &DeviceMemory<Self::TableValue>,
    ) -> Result<()>;

    fn lookup_table_cubic_image_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, C1>,
        values: &DeviceMemory<Self::TableValue>,
        levels: &DeviceMemory<Self::TableValue>,
    ) -> Result<()>;
}

#[macro_use]
#[path = "color_lookup_c1_dispatch_macros.rs"]
mod c1_dispatch_macros;

#[macro_use]
#[path = "color_lookup_packed_dispatch_macros.rs"]
mod packed_dispatch_macros;

#[path = "color_lookup_packed_dispatch.rs"]
mod packed;

#[path = "color_lookup_c1_dispatch.rs"]
mod c1;

#[path = "color_lookup_packed_traits.rs"]
mod packed_traits;
