use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage, color,
        view::{AC4, ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::image::{
        CopyImage, ImageBacking, ImagePipeline, color_lookup_dispatch::TrilinearLookupTableImage,
    },
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: TrilinearLookupTableImage<T, L>,
{
    /// Converts colors through a three-dimensional lookup table using trilinear interpolation.
    /// `values` stores device-resident, four-byte packed RGBX grid values as contiguous x/y
    /// planes along the z axis. `levels` contains three host arrays of source-domain breakpoints,
    /// one per grid axis. Native ordering and domain requirements are not validated.
    /// The supplied destination must match the source pixel dimensions. C4 conversion
    /// copies source alpha unchanged; AC4 leaves destination alpha in its previous
    /// initialization state, so the call does not initialize AC4 alpha.
    /// Prepares image and device-value regions and waits for producers on other streams.
    /// Rejects capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Source and destination image dimensions must match.
    /// Each level array must contain at least two entries and have a length representable
    /// as `i32`. Their checked length product must not exceed `values.len()`.
    pub fn lookup_table_trilinear_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        values: &DeviceMemory<u32>,
        levels: [&[T]; 3],
    ) -> Result<()> {
        <Self as TrilinearLookupTableImage<T, L>>::lookup_table_trilinear_image(
            stream_context,
            source,
            destination,
            values,
            levels,
        )
    }
}

impl ImagePipeline<'_, u8, AC4>
where
    u8: SupportedImage<AC4>,
    Self: CopyImage<u8, AC4>,
{
    /// Converts colors through a three-dimensional lookup table using trilinear interpolation.
    /// `values` stores device-resident, four-byte packed RGBX grid values as contiguous x/y
    /// planes along the z axis. `levels` contains three host arrays of source-domain breakpoints,
    /// one per grid axis. Native ordering and domain requirements are not validated.
    /// Reuses owned AC4 backing; borrowed input is copied into a new owned image before
    /// updating it in place. Alpha remains unchanged.
    /// Prepares image and device-value regions and waits for producers on other streams.
    /// Rejects capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Each level array must contain at least two entries and have a length representable
    /// as `i32`. Their checked length product must not exceed `values.len()`.
    /// Image allocation, view construction, or source preservation copies can also fail.
    pub fn lookup_table_trilinear_in_place(
        mut self,
        values: &DeviceMemory<u32>,
        levels: [&[u8]; 3],
    ) -> Result<Self> {
        self.ensure_eager()?;
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                self.stream_context.stream().ensure_not_capturing()?;
                self.stream_context.prepare_region(&image_view)?;
                self.stream_context.prepare_region(values)?;
                self.stream_context.sync_scope(|| {
                    // SAFETY: the initialized read-write pixels and lookup data remain borrowed through completion.
                    unsafe {
                        color::lookup_table_trilinear_ac4_in_place(
                            self.stream_context,
                            &mut image_view,
                            values,
                            levels,
                        )
                    }
                })?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = Image::<u8, AC4>::from_view(source)?;
                let mut destination_view = destination.view_mut()?;
                self.stream_context.stream().ensure_not_capturing()?;
                self.stream_context.prepare_region(&destination_view)?;
                self.stream_context.prepare_region(values)?;
                self.stream_context.sync_scope(|| {
                    // SAFETY: the initialized read-write pixels and lookup data remain borrowed through completion.
                    unsafe {
                        color::lookup_table_trilinear_ac4_in_place(
                            self.stream_context,
                            &mut destination_view,
                            values,
                            levels,
                        )
                    }
                })?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: TrilinearLookupTableImage<T, L>,
{
    /// Converts colors through a three-dimensional lookup table using trilinear interpolation.
    /// `values` stores device-resident, four-byte packed RGBX grid values as contiguous x/y
    /// planes along the z axis. `levels` contains three host arrays of source-domain breakpoints,
    /// one per grid axis. Native ordering and domain requirements are not validated.
    /// Always allocates a new owned image with the source pixel dimensions. Preserves
    /// source alpha for both provided C4 and AC4 conversions.
    /// Prepares image and device-value regions and waits for producers on other streams.
    /// Rejects capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Each level array must contain at least two entries and have a length representable
    /// as `i32`. Their checked length product must not exceed `values.len()`.
    /// Image allocation, view construction, or source preservation copies can also fail.
    pub fn lookup_table_trilinear(
        self,
        values: &DeviceMemory<u32>,
        levels: [&[T]; 3],
    ) -> Result<Self> {
        self.ensure_eager()?;
        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            if L::IGNORES_ALPHA {
                destination_view.copy_from(&source)?;
            }
            <Self as TrilinearLookupTableImage<T, L>>::lookup_table_trilinear_image(
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
            // SAFETY: completed LUT dispatch initialized all active channels; ignored alpha was copied.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}
