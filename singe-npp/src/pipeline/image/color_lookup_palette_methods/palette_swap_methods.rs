use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::{
        Image, SupportedImage,
        view::{C3, C4, ImageOutputView, ImageView},
    },
    pipeline::image::{
        ImageBacking, ImagePipeline, color_lookup_dispatch::PaletteLookupTableSwapImage,
    },
};

impl<T> ImagePipeline<'_, T, C3>
where
    T: Copy,
    Self: PaletteLookupTableSwapImage<T>,
{
    /// Maps three source channels through device palettes into the native C4 palette-swap layout.
    /// Uses the least significant `bit_size` bits of each source sample as a palette index.
    /// Each of the three tables supplies one color channel. `alpha` controls the native
    /// destination alpha slot.
    /// If `alpha` lies outside the unsigned sample range, NPP leaves destination alpha
    /// untouched. Such a call does not establish that the entire output is initialized.
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
    pub fn lookup_table_palette_swap_to_c4_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C3>,
        destination: &mut ImageOutputView<'_, T, C4>,
        alpha: i32,
        tables: &[&DeviceMemory<T>; 3],
        bit_size: i32,
    ) -> Result<()> {
        <Self as PaletteLookupTableSwapImage<T>>::lookup_table_palette_swap_image(
            stream_context,
            source,
            destination,
            alpha,
            tables,
            bit_size,
        )
    }
}

impl<'a, T> ImagePipeline<'a, T, C3>
where
    T: Copy + SupportedImage<C4> + TryFrom<i32>,
    Self: PaletteLookupTableSwapImage<T>,
{
    /// Maps three source channels through device palettes into the native C4 palette-swap layout.
    /// Uses the least significant `bit_size` bits of each source sample as a palette index.
    /// Each of the three tables supplies one color channel. `alpha` controls the native
    /// destination alpha slot.
    /// Rejects `alpha` outside the destination sample range so every output channel is
    /// initialized by the native conversion.
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
    /// `alpha` must fit the unsigned destination sample type.
    /// Image allocation or view construction can also fail.
    pub fn lookup_table_palette_swap_to_c4(
        self,
        alpha: i32,
        tables: &[&DeviceMemory<T>; 3],
        bit_size: i32,
    ) -> Result<ImagePipeline<'a, T, C4>> {
        self.ensure_eager()?;
        T::try_from(alpha).map_err(|_| Error::out_of_range("alpha"))?;

        let mut destination =
            Image::<T, C4>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            <Self as PaletteLookupTableSwapImage<T>>::lookup_table_palette_swap_image(
                self.stream_context,
                &source,
                &mut destination_view,
                alpha,
                tables,
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
