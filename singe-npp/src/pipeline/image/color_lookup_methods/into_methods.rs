use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageOutputView, ImageView, ImageViewMut},
    pipeline::image::{ImagePipeline, color_lookup_dispatch::LookupTableImage},
};

impl<T> ImagePipeline<'_, T, C1>
where
    T: Copy,
    Self: LookupTableImage<T>,
{
    /// Maps source samples through device lookup values and input levels without interpolation.
    ///
    /// `values` and `levels` must contain matching counts of at least two entries, fitting
    /// `i32`. Input levels must satisfy the selected native ordering and domain requirements;
    /// the wrapper validates table counts, not their contents.
    /// Writes into an equally sized `destination`.
    /// Prepares images and both device tables, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if table lengths are invalid, contexts differ, capture checks, producer
    /// waits, native conversion, or stream synchronization fail. NPP may reject additional
    /// type-specific table limits or mapping parameters.
    /// Source and destination dimensions must match.
    pub fn lookup_table_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageOutputView<'_, T, C1>,
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

    /// Maps source samples through device lookup values and input levels without interpolation.
    ///
    /// `values` and `levels` must contain matching counts of at least two entries, fitting
    /// `i32`. Input levels must satisfy the selected native ordering and domain requirements;
    /// the wrapper validates table counts, not their contents.
    /// Updates the image in place.
    /// Prepares images and both device tables, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if table lengths are invalid, contexts differ, capture checks, producer
    /// waits, native conversion, or stream synchronization fail. NPP may reject additional
    /// type-specific table limits or mapping parameters.
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

    /// Maps source samples through device lookup values and input levels using linear interpolation.
    ///
    /// `values` and `levels` must contain matching counts of at least two entries, fitting
    /// `i32`. Input levels must satisfy the selected native ordering and domain requirements;
    /// the wrapper validates table counts, not their contents.
    /// Writes into an equally sized `destination`.
    /// Prepares images and both device tables, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if table lengths are invalid, contexts differ, capture checks, producer
    /// waits, native conversion, or stream synchronization fail. NPP may reject additional
    /// type-specific table limits or mapping parameters.
    /// Source and destination dimensions must match.
    pub fn lookup_table_linear_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageOutputView<'_, T, C1>,
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

    /// Maps source samples through device lookup values and input levels using linear interpolation.
    ///
    /// `values` and `levels` must contain matching counts of at least two entries, fitting
    /// `i32`. Input levels must satisfy the selected native ordering and domain requirements;
    /// the wrapper validates table counts, not their contents.
    /// Updates the image in place.
    /// Prepares images and both device tables, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if table lengths are invalid, contexts differ, capture checks, producer
    /// waits, native conversion, or stream synchronization fail. NPP may reject additional
    /// type-specific table limits or mapping parameters.
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

    /// Maps source samples through device lookup values and input levels using cubic interpolation.
    ///
    /// `values` and `levels` must contain matching counts of at least two entries, fitting
    /// `i32`. Input levels must satisfy the selected native ordering and domain requirements;
    /// the wrapper validates table counts, not their contents.
    /// Writes into an equally sized `destination`.
    /// Prepares images and both device tables, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if table lengths are invalid, contexts differ, capture checks, producer
    /// waits, native conversion, or stream synchronization fail. NPP may reject additional
    /// type-specific table limits or mapping parameters.
    /// Source and destination dimensions must match.
    pub fn lookup_table_cubic_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageOutputView<'_, T, C1>,
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

    /// Maps source samples through device lookup values and input levels using cubic interpolation.
    ///
    /// `values` and `levels` must contain matching counts of at least two entries, fitting
    /// `i32`. Input levels must satisfy the selected native ordering and domain requirements;
    /// the wrapper validates table counts, not their contents.
    /// Updates the image in place.
    /// Prepares images and both device tables, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if table lengths are invalid, contexts differ, capture checks, producer
    /// waits, native conversion, or stream synchronization fail. NPP may reject additional
    /// type-specific table limits or mapping parameters.
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
