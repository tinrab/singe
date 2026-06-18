use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageView, ImageViewMut},
};

use super::super::{ImagePipeline, color_lookup_dispatch::LookupTableImage};

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
    Self: LookupTableImage<T>,
{
    pub fn lookup_table_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageViewMut<'_, T, C1>,
        values: &DeviceMemory<<Self as LookupTableImage<T>>::TableValue>,
        levels: &DeviceMemory<<Self as LookupTableImage<T>>::TableValue>,
    ) -> Result<()> {
        <Self as LookupTableImage<T>>::lookup_table_image(
            stream_context,
            source,
            destination,
            values,
            levels,
        )
    }

    pub fn lookup_table_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, C1>,
        values: &DeviceMemory<<Self as LookupTableImage<T>>::TableValue>,
        levels: &DeviceMemory<<Self as LookupTableImage<T>>::TableValue>,
    ) -> Result<()> {
        <Self as LookupTableImage<T>>::lookup_table_image_in_place(
            stream_context,
            image,
            values,
            levels,
        )
    }

    pub fn lookup_table_linear_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageViewMut<'_, T, C1>,
        values: &DeviceMemory<<Self as LookupTableImage<T>>::TableValue>,
        levels: &DeviceMemory<<Self as LookupTableImage<T>>::TableValue>,
    ) -> Result<()> {
        <Self as LookupTableImage<T>>::lookup_table_linear_image(
            stream_context,
            source,
            destination,
            values,
            levels,
        )
    }

    pub fn lookup_table_linear_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, C1>,
        values: &DeviceMemory<<Self as LookupTableImage<T>>::TableValue>,
        levels: &DeviceMemory<<Self as LookupTableImage<T>>::TableValue>,
    ) -> Result<()> {
        <Self as LookupTableImage<T>>::lookup_table_linear_image_in_place(
            stream_context,
            image,
            values,
            levels,
        )
    }

    pub fn lookup_table_cubic_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageViewMut<'_, T, C1>,
        values: &DeviceMemory<<Self as LookupTableImage<T>>::TableValue>,
        levels: &DeviceMemory<<Self as LookupTableImage<T>>::TableValue>,
    ) -> Result<()> {
        <Self as LookupTableImage<T>>::lookup_table_cubic_image(
            stream_context,
            source,
            destination,
            values,
            levels,
        )
    }

    pub fn lookup_table_cubic_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, C1>,
        values: &DeviceMemory<<Self as LookupTableImage<T>>::TableValue>,
        levels: &DeviceMemory<<Self as LookupTableImage<T>>::TableValue>,
    ) -> Result<()> {
        <Self as LookupTableImage<T>>::lookup_table_cubic_image_in_place(
            stream_context,
            image,
            values,
            levels,
        )
    }
}
