use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView},
    },
    pipeline::{ImageBacking, image::ImagePipeline},
};

mod ac4_methods;
mod c4_methods;

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
pub unsafe trait DuplicateToC3Image<T> {
    fn duplicate_to_c3_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageOutputView<'_, T, C3>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
pub unsafe trait DuplicateToC4Image<T> {
    fn duplicate_to_c4_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageOutputView<'_, T, C4>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
pub unsafe trait DuplicateToAC4Image<T> {
    fn duplicate_to_ac4_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageOutputView<'_, T, AC4>,
    ) -> Result<()>;
}

macro_rules! impl_duplicate_to_c3_image {
    ($ty:ty, $duplicate:path) => {
        unsafe impl<'a> DuplicateToC3Image<$ty> for ImagePipeline<'a, $ty, C1> {
            fn duplicate_to_c3_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                destination: &mut ImageOutputView<'_, $ty, C3>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $duplicate(stream_context, source, destination) })
            }
        }
    };
}

macro_rules! impl_duplicate_to_c4_image {
    ($ty:ty, $duplicate:path) => {
        unsafe impl<'a> DuplicateToC4Image<$ty> for ImagePipeline<'a, $ty, C1> {
            fn duplicate_to_c4_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                destination: &mut ImageOutputView<'_, $ty, C4>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $duplicate(stream_context, source, destination) })
            }
        }
    };
}

macro_rules! impl_duplicate_to_ac4_image {
    ($ty:ty, $duplicate:path) => {
        unsafe impl<'a> DuplicateToAC4Image<$ty> for ImagePipeline<'a, $ty, C1> {
            fn duplicate_to_ac4_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                destination: &mut ImageOutputView<'_, $ty, AC4>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $duplicate(stream_context, source, destination) })
            }
        }
    };
}

mod dispatch_impls;

impl<T> ImagePipeline<'_, T, C1>
where
    T: Copy,
    Self: DuplicateToC3Image<T>,
{
    /// Duplicates each C1 source sample into all three C3 channels.
    ///
    /// Every destination channel receives the same source sample.
    /// Source and destination pixel dimensions must match.
    /// The provided implementations retain sample type: `u8`, `u16`, `i16`, `i32`, or `f32`.
    /// Prepares source and destination regions, waits for producers on other streams,
    /// rejects capture, and returns after successful stream completion. Row padding remains
    /// unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or synchronization fail.
    /// Source and destination pixel dimensions must match.
    pub fn duplicate_to_c3_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageOutputView<'_, T, C3>,
    ) -> Result<()> {
        <Self as DuplicateToC3Image<T>>::duplicate_to_c3_image(stream_context, source, destination)
    }
}

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy + SupportedImage<C3>,
    Self: DuplicateToC3Image<T>,
{
    /// Duplicates each C1 source sample into all three C3 channels.
    ///
    /// Every destination channel receives the same source sample.
    /// Allocates a new source-sized owned image in the destination layout.
    /// The provided implementations retain sample type: `u8`, `u16`, `i16`, `i32`, or `f32`.
    /// Prepares source and destination regions, waits for producers on other streams,
    /// rejects capture, and returns after successful stream completion. Row padding remains
    /// unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or synchronization fail.
    /// Output allocation or image view construction can also fail.
    pub fn duplicate_to_c3(self) -> Result<ImagePipeline<'a, T, C3>> {
        self.ensure_eager()?;

        let mut destination =
            Image::<T, C3>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            <Self as DuplicateToC3Image<T>>::duplicate_to_c3_image(
                self.stream_context,
                &source,
                &mut destination_view,
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
