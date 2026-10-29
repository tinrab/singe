use singe_cuda::memory::DeviceMemory;

use crate::{
    error::Result,
    image::{Image, SupportedImage, view::C1},
    pipeline::{
        ImageBacking,
        image::{ImagePipeline, color_lookup_dispatch::LookupTableImage},
    },
};

mod into_methods;

mod channels_into_methods;

mod channels_methods;

impl<T> ImagePipeline<'_, T, C1>
where
    T: Copy + SupportedImage<C1>,
    Self: LookupTableImage<T>,
{
    /// Maps source samples through device lookup values and input levels without interpolation.
    ///
    /// `values` and `levels` must contain matching counts of at least two entries, fitting
    /// `i32`. Input levels must satisfy the selected native ordering and domain requirements;
    /// the wrapper validates table counts, not their contents.
    /// Allocates a new owned image with the same dimensions, including for owned input.
    /// Prepares images and both device tables, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if table lengths are invalid, contexts differ, capture checks, producer
    /// waits, native conversion, or stream synchronization fail. NPP may reject additional
    /// type-specific table limits or mapping parameters.
    /// Output allocation and view construction can also fail.
    pub fn lookup_table(
        self,
        values: &DeviceMemory<<Self as LookupTableImage<T>>::TableValue>,
        levels: &DeviceMemory<<Self as LookupTableImage<T>>::TableValue>,
    ) -> Result<Self> {
        self.ensure_eager()?;
        let mut destination =
            Image::<T, C1>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            <Self as LookupTableImage<T>>::lookup_table_image(
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

    /// Maps source samples through device lookup values and input levels using linear interpolation.
    ///
    /// `values` and `levels` must contain matching counts of at least two entries, fitting
    /// `i32`. Input levels must satisfy the selected native ordering and domain requirements;
    /// the wrapper validates table counts, not their contents.
    /// Allocates a new owned image with the same dimensions, including for owned input.
    /// Prepares images and both device tables, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if table lengths are invalid, contexts differ, capture checks, producer
    /// waits, native conversion, or stream synchronization fail. NPP may reject additional
    /// type-specific table limits or mapping parameters.
    /// Output allocation and view construction can also fail.
    pub fn lookup_table_linear(
        self,
        values: &DeviceMemory<<Self as LookupTableImage<T>>::TableValue>,
        levels: &DeviceMemory<<Self as LookupTableImage<T>>::TableValue>,
    ) -> Result<Self> {
        self.ensure_eager()?;
        let mut destination =
            Image::<T, C1>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            <Self as LookupTableImage<T>>::lookup_table_linear_image(
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

    /// Maps source samples through device lookup values and input levels using cubic interpolation.
    ///
    /// `values` and `levels` must contain matching counts of at least two entries, fitting
    /// `i32`. Input levels must satisfy the selected native ordering and domain requirements;
    /// the wrapper validates table counts, not their contents.
    /// Allocates a new owned image with the same dimensions, including for owned input.
    /// Prepares images and both device tables, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if table lengths are invalid, contexts differ, capture checks, producer
    /// waits, native conversion, or stream synchronization fail. NPP may reject additional
    /// type-specific table limits or mapping parameters.
    /// Output allocation and view construction can also fail.
    pub fn lookup_table_cubic(
        self,
        values: &DeviceMemory<<Self as LookupTableImage<T>>::TableValue>,
        levels: &DeviceMemory<<Self as LookupTableImage<T>>::TableValue>,
    ) -> Result<Self> {
        self.ensure_eager()?;
        let mut destination =
            Image::<T, C1>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            <Self as LookupTableImage<T>>::lookup_table_cubic_image(
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
