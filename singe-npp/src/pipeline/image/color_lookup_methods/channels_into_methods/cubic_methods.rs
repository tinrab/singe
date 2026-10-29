use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageOutputView, ImageView, ImageViewMut},
    pipeline::image::{ImagePipeline, color_lookup_dispatch::PackedLookupTableImage},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    /// Maps each processed channel through its own lookup table using cubic interpolation.
    /// `levels` contains source-domain breakpoints and `values` contains mapped outputs in
    /// device memory, with one table pair per processed channel: three for C3 and AC4, four
    /// for C4. The provided integer-image implementations use `i32` tables; `f32` images use
    /// `f32` tables. Ordering, coefficient values, and native domain coverage are not checked.
    /// AC4 destination alpha retains its previous initialization state; this call does not
    /// initialize alpha.
    /// All table buffers must belong to the selected CUDA context and be ready before
    /// native execution. This packed dispatch does not check their contexts or wait for
    /// their producers. Prepares image regions and waits for image producers on other streams.
    /// Rejects capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Source and destination image dimensions must match.
    /// Each channel must have equal value and level counts, at least two levels, and a count
    /// representable as `i32`. Additional native table constraints can cause NPP errors.
    pub fn lookup_table_channels_cubic_into<const CHANNELS: usize>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        values: &[&DeviceMemory<<Self as PackedLookupTableImage<T, L, CHANNELS>>::TableValue>;
             CHANNELS],
        levels: &[&DeviceMemory<<Self as PackedLookupTableImage<T, L, CHANNELS>>::TableValue>;
             CHANNELS],
    ) -> Result<()>
    where
        Self: PackedLookupTableImage<T, L, CHANNELS>,
    {
        <Self as PackedLookupTableImage<T, L, CHANNELS>>::lookup_table_channels_cubic_image(
            stream_context,
            source,
            destination,
            values,
            levels,
        )
    }

    /// Maps each processed channel through its own lookup table using cubic interpolation.
    /// `levels` contains source-domain breakpoints and `values` contains mapped outputs in
    /// device memory, with one table pair per processed channel: three for C3 and AC4, four
    /// for C4. The provided integer-image implementations use `i32` tables; `f32` images use
    /// `f32` tables. Ordering, coefficient values, and native domain coverage are not checked.
    /// Updates the image in place; AC4 alpha remains unchanged.
    /// All table buffers must belong to the selected CUDA context and be ready before
    /// native execution. This packed dispatch does not check their contexts or wait for
    /// their producers. Prepares image regions and waits for image producers on other streams.
    /// Rejects capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Each channel must have equal value and level counts, at least two levels, and a count
    /// representable as `i32`. Additional native table constraints can cause NPP errors.
    pub fn lookup_table_channels_cubic_in_place<const CHANNELS: usize>(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, L>,
        values: &[&DeviceMemory<<Self as PackedLookupTableImage<T, L, CHANNELS>>::TableValue>;
             CHANNELS],
        levels: &[&DeviceMemory<<Self as PackedLookupTableImage<T, L, CHANNELS>>::TableValue>;
             CHANNELS],
    ) -> Result<()>
    where
        Self: PackedLookupTableImage<T, L, CHANNELS>,
    {
        <Self as PackedLookupTableImage<T, L, CHANNELS>>::lookup_table_channels_cubic_image_in_place(
            stream_context,
            image,
            values,
            levels,
        )
    }
}
