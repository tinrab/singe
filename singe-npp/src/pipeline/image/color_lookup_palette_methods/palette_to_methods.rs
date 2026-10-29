use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{C1, ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::{
        ImagePipeline,
        image::{ImageBacking, color_lookup_palette_methods::PaletteLookupTableToImage},
    },
};

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
{
    /// Expands C1 palette indices into the selected output sample type and channel layout.
    /// Uses the least significant `bit_size` bits of each source sample as a palette index.
    /// The provided C1 outputs use one table value per index; packed C3 outputs use three
    /// consecutive values per index. The table element type follows the selected conversion.
    /// Prepares image and table regions and waits for producers on other streams.
    /// Rejects capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Source and destination image dimensions must match.
    /// `bit_size` must be between one and the source element width in bits. The table must
    /// contain at least `2^bit_size` entries of one C1 value or three packed C3 values. Required-length arithmetic is checked.
    pub fn lookup_table_palette_to_into<D, L>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageOutputView<'_, D, L>,
        table: &DeviceMemory<<Self as PaletteLookupTableToImage<T, D, L>>::TableValue>,
        bit_size: i32,
    ) -> Result<()>
    where
        D: Copy,
        L: ChannelLayout,
        Self: PaletteLookupTableToImage<T, D, L>,
    {
        <Self as PaletteLookupTableToImage<T, D, L>>::lookup_table_palette_to_image(
            stream_context,
            source,
            destination,
            table,
            bit_size,
        )
    }

    /// Expands C1 palette indices into the selected output sample type and channel layout.
    /// Uses the least significant `bit_size` bits of each source sample as a palette index.
    /// The provided C1 outputs use one table value per index; packed C3 outputs use three
    /// consecutive values per index. The table element type follows the selected conversion.
    /// Always allocates a new owned image with the source pixel dimensions.
    /// Prepares image and table regions and waits for producers on other streams.
    /// Rejects capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// `bit_size` must be between one and the source element width in bits. The table must
    /// contain at least `2^bit_size` entries of one C1 value or three packed C3 values. Required-length arithmetic is checked.
    /// Image allocation or view construction can also fail.
    pub fn lookup_table_palette_to<D, L>(
        self,
        table: &DeviceMemory<<Self as PaletteLookupTableToImage<T, D, L>>::TableValue>,
        bit_size: i32,
    ) -> Result<ImagePipeline<'a, D, L>>
    where
        D: Copy + SupportedImage<L>,
        L: ChannelLayout,
        Self: PaletteLookupTableToImage<T, D, L>,
    {
        self.ensure_eager()?;

        let mut destination =
            Image::<D, L>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            <Self as PaletteLookupTableToImage<T, D, L>>::lookup_table_palette_to_image(
                self.stream_context,
                &source,
                &mut destination_view,
                table,
                bit_size,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}
