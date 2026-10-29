use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::{
        ImageBacking,
        image::{
            ImagePipeline,
            color_lookup_dispatch::{
                PackedPaletteLookupTableImage, PaletteLookupTableImage, PaletteLookupTableToImage,
            },
        },
    },
};

mod palette_swap_methods;
mod palette_to_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: PaletteLookupTableImage<T, L>,
{
    /// Maps C1 samples through a device palette.
    /// Uses the least significant `bit_size` bits of each source sample as a palette index.
    /// Prepares image and table regions and waits for producers on other streams.
    /// Rejects capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Source and destination image dimensions must match.
    /// `bit_size` must be between one and the source element width in bits. Each table must
    /// contain at least `2^bit_size` values. Required-length arithmetic is checked.
    pub fn lookup_table_palette_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        table: &DeviceMemory<T>,
        bit_size: i32,
    ) -> Result<()> {
        <Self as PaletteLookupTableImage<T, L>>::lookup_table_palette_image(
            stream_context,
            source,
            destination,
            table,
            bit_size,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: PaletteLookupTableImage<T, L>,
{
    /// Maps C1 samples through a device palette.
    /// Uses the least significant `bit_size` bits of each source sample as a palette index.
    /// Always allocates a new owned image with the source pixel dimensions.
    /// Prepares image and table regions and waits for producers on other streams.
    /// Rejects capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// `bit_size` must be between one and the source element width in bits. Each table must
    /// contain at least `2^bit_size` values. Required-length arithmetic is checked.
    /// Image allocation or view construction can also fail.
    pub fn lookup_table_palette(self, table: &DeviceMemory<T>, bit_size: i32) -> Result<Self> {
        self.ensure_eager()?;
        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            if L::IGNORES_ALPHA {
                destination_view.copy_from(&source)?;
            }
            <Self as PaletteLookupTableImage<T, L>>::lookup_table_palette_image(
                self.stream_context,
                &source,
                &mut destination_view,
                table,
                bit_size,
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

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    /// Maps each processed channel through its own device palette.
    /// Uses the least significant `bit_size` bits of each source sample as a palette index.
    /// Supplies three palettes for C3 and AC4 or four for C4, one per processed channel.
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
    /// `bit_size` must be between one and the source element width in bits. Each table must
    /// contain at least `2^bit_size` values. Required-length arithmetic is checked.
    pub fn lookup_table_palette_channels_into<const CHANNELS: usize>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        tables: &[&DeviceMemory<T>; CHANNELS],
        bit_size: i32,
    ) -> Result<()>
    where
        Self: PackedPaletteLookupTableImage<T, L, CHANNELS>,
    {
        <Self as PackedPaletteLookupTableImage<T, L, CHANNELS>>::lookup_table_palette_channels_image(
            stream_context,
            source,
            destination,
            tables,
            bit_size,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
{
    /// Maps each processed channel through its own device palette.
    /// Uses the least significant `bit_size` bits of each source sample as a palette index.
    /// Supplies three palettes for C3 and AC4 or four for C4, one per processed channel.
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
    /// `bit_size` must be between one and the source element width in bits. Each table must
    /// contain at least `2^bit_size` values. Required-length arithmetic is checked.
    /// Image allocation or view construction, including alpha preservation copies, can also fail.
    pub fn lookup_table_palette_channels<const CHANNELS: usize>(
        self,
        tables: &[&DeviceMemory<T>; CHANNELS],
        bit_size: i32,
    ) -> Result<Self>
    where
        Self: PackedPaletteLookupTableImage<T, L, CHANNELS>,
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
            <Self as PackedPaletteLookupTableImage<T, L, CHANNELS>>::lookup_table_palette_channels_image(
                self.stream_context,
                &source,
                &mut destination_view,
                tables,
                bit_size,
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
