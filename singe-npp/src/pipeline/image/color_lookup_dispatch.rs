use singe_cuda::memory::DeviceMemory;

pub use self::packed_traits::*;
pub use crate::pipeline::image::color_lookup_palette_dispatch::*;
use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageOutputView, ImageView, ImageViewMut},
    pipeline::image::ImagePipeline,
};

/// Dispatches completed single-channel lookup-table color conversions.
///
/// # Safety
///
/// Implementations must preserve native pixel and table element representations and the
/// selected interpolation semantics. Uphold table and image validation, complete borrowed
/// work on every path, and initialize every output sample on success.
pub unsafe trait LookupTableImage<T> {
    /// Element type of both device lookup tables: `i32` for integer pixels and `f32` for `f32` pixels in the provided implementations.
    type TableValue;

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
    fn lookup_table_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageOutputView<'_, T, C1>,
        values: &DeviceMemory<Self::TableValue>,
        levels: &DeviceMemory<Self::TableValue>,
    ) -> Result<()>;

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
    fn lookup_table_image_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, C1>,
        values: &DeviceMemory<Self::TableValue>,
        levels: &DeviceMemory<Self::TableValue>,
    ) -> Result<()>;

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
    fn lookup_table_linear_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageOutputView<'_, T, C1>,
        values: &DeviceMemory<Self::TableValue>,
        levels: &DeviceMemory<Self::TableValue>,
    ) -> Result<()>;

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
    fn lookup_table_linear_image_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, C1>,
        values: &DeviceMemory<Self::TableValue>,
        levels: &DeviceMemory<Self::TableValue>,
    ) -> Result<()>;

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
    fn lookup_table_cubic_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageOutputView<'_, T, C1>,
        values: &DeviceMemory<Self::TableValue>,
        levels: &DeviceMemory<Self::TableValue>,
    ) -> Result<()>;

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
    fn lookup_table_cubic_image_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, C1>,
        values: &DeviceMemory<Self::TableValue>,
        levels: &DeviceMemory<Self::TableValue>,
    ) -> Result<()>;
}

#[macro_use]
mod c1_dispatch_macros;

#[macro_use]
mod packed_dispatch_macros;

mod packed;

mod c1;

mod packed_traits;
