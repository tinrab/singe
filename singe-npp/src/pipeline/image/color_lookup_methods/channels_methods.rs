use singe_cuda::memory::DeviceMemory;

use crate::{
    error::Result,
    image::{Image, SupportedImage, view::ChannelLayout},
    pipeline::image::{ImageBacking, ImagePipeline, color_lookup_dispatch::PackedLookupTableImage},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
{
    /// Maps each processed channel through its own lookup table using no interpolation.
    /// `levels` contains source-domain breakpoints and `values` contains mapped outputs in
    /// device memory, with one table pair per processed channel: three for C3 and AC4, four
    /// for C4. The provided integer-image implementations use `i32` tables; `f32` images use
    /// `f32` tables. Ordering, coefficient values, and native domain coverage are not checked.
    /// Always allocates a new owned image with the source pixel dimensions.
    /// Preserves AC4 alpha from the source image.
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
    /// Image allocation or view construction, including alpha preservation copies, can also fail.
    pub fn lookup_table_channels<const CHANNELS: usize>(
        self,
        values: &[&DeviceMemory<<Self as PackedLookupTableImage<T, L, CHANNELS>>::TableValue>;
             CHANNELS],
        levels: &[&DeviceMemory<<Self as PackedLookupTableImage<T, L, CHANNELS>>::TableValue>;
             CHANNELS],
    ) -> Result<Self>
    where
        Self: PackedLookupTableImage<T, L, CHANNELS>,
    {
        self.ensure_eager()?;
        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            if L::IGNORES_ALPHA {
                destination_view.copy_from(&source)?;
            }
            <Self as PackedLookupTableImage<T, L, CHANNELS>>::lookup_table_channels_image(
                self.stream_context,
                &source,
                &mut destination_view,
                values,
                levels,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: completed dispatch initialized every output pixel, including preserved alpha where required.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }

    /// Maps each processed channel through its own lookup table using linear interpolation.
    /// `levels` contains source-domain breakpoints and `values` contains mapped outputs in
    /// device memory, with one table pair per processed channel: three for C3 and AC4, four
    /// for C4. The provided integer-image implementations use `i32` tables; `f32` images use
    /// `f32` tables. Ordering, coefficient values, and native domain coverage are not checked.
    /// Always allocates a new owned image with the source pixel dimensions.
    /// Preserves AC4 alpha from the source image.
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
    /// Image allocation or view construction, including alpha preservation copies, can also fail.
    pub fn lookup_table_channels_linear<const CHANNELS: usize>(
        self,
        values: &[&DeviceMemory<<Self as PackedLookupTableImage<T, L, CHANNELS>>::TableValue>;
             CHANNELS],
        levels: &[&DeviceMemory<<Self as PackedLookupTableImage<T, L, CHANNELS>>::TableValue>;
             CHANNELS],
    ) -> Result<Self>
    where
        Self: PackedLookupTableImage<T, L, CHANNELS>,
    {
        self.ensure_eager()?;
        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            if L::IGNORES_ALPHA {
                destination_view.copy_from(&source)?;
            }
            <Self as PackedLookupTableImage<T, L, CHANNELS>>::lookup_table_channels_linear_image(
                self.stream_context,
                &source,
                &mut destination_view,
                values,
                levels,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: completed dispatch initialized every output pixel, including preserved alpha where required.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }

    /// Maps each processed channel through its own lookup table using cubic interpolation.
    /// `levels` contains source-domain breakpoints and `values` contains mapped outputs in
    /// device memory, with one table pair per processed channel: three for C3 and AC4, four
    /// for C4. The provided integer-image implementations use `i32` tables; `f32` images use
    /// `f32` tables. Ordering, coefficient values, and native domain coverage are not checked.
    /// Always allocates a new owned image with the source pixel dimensions.
    /// Preserves AC4 alpha from the source image.
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
    /// Image allocation or view construction, including alpha preservation copies, can also fail.
    pub fn lookup_table_channels_cubic<const CHANNELS: usize>(
        self,
        values: &[&DeviceMemory<<Self as PackedLookupTableImage<T, L, CHANNELS>>::TableValue>;
             CHANNELS],
        levels: &[&DeviceMemory<<Self as PackedLookupTableImage<T, L, CHANNELS>>::TableValue>;
             CHANNELS],
    ) -> Result<Self>
    where
        Self: PackedLookupTableImage<T, L, CHANNELS>,
    {
        self.ensure_eager()?;
        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            if L::IGNORES_ALPHA {
                destination_view.copy_from(&source)?;
            }
            <Self as PackedLookupTableImage<T, L, CHANNELS>>::lookup_table_channels_cubic_image(
                self.stream_context,
                &source,
                &mut destination_view,
                values,
                levels,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: completed dispatch initialized every output pixel, including preserved alpha where required.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}
