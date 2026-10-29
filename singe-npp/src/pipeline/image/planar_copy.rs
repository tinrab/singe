use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage, exchange,
        view::{
            C1, C3, C4, ChannelLayout, ImageOutputView, ImageView, PlanarImageOutputView,
            PlanarImageView,
        },
    },
    pipeline::{
        ImageBacking,
        image::{ImagePipeline, planar::PlanarImage},
    },
};

/// # Safety
/// Complete borrowed native work on every path and initialize every logical channel in the destination.
/// Preserve the native element ABI and exactly matching packed and planar channel counts.
pub unsafe trait PackedToPlanarImage<T, L, const C: usize> {
    fn copy_packed_to_planar_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut PlanarImageOutputView<'_, T, C>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and initialize every logical channel in the destination.
/// Preserve the native element ABI and exactly matching packed and planar channel counts.
pub unsafe trait PlanarToPackedImage<T, L, const C: usize> {
    fn copy_planar_to_packed_image(
        stream_context: &StreamContext,
        source: &PlanarImageView<'_, T, C>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>;
}

macro_rules! impl_packed_to_planar_image {
    ($ty:ty, $layout:ty, $planes:literal, $copy:path) => {
        unsafe impl<'a> PackedToPlanarImage<$ty, $layout, $planes>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn copy_packed_to_planar_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut PlanarImageOutputView<'_, $ty, $planes>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                for plane in destination.planes() {
                    stream_context.prepare_region(plane)?;
                }
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe { $copy(stream_context, source, destination) })
            }
        }
    };
}

macro_rules! impl_planar_to_packed_image {
    ($ty:ty, $layout:ty, $planes:literal, $copy:path) => {
        unsafe impl<'a> PlanarToPackedImage<$ty, $layout, $planes>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn copy_planar_to_packed_image(
                stream_context: &StreamContext,
                source: &PlanarImageView<'_, $ty, $planes>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(destination)?;
                for plane in source.planes() {
                    stream_context.prepare_region(plane)?;
                }
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe { $copy(stream_context, source, destination) })
            }
        }
    };
}

impl_packed_to_planar_image!(u8, C3, 3, exchange::copy_c3_to_p3);

impl_packed_to_planar_image!(u8, C4, 4, exchange::copy_c4_to_p4);

impl_packed_to_planar_image!(u16, C3, 3, exchange::copy_c3_to_p3);

impl_packed_to_planar_image!(u16, C4, 4, exchange::copy_c4_to_p4);

impl_packed_to_planar_image!(i16, C3, 3, exchange::copy_c3_to_p3);

impl_packed_to_planar_image!(i16, C4, 4, exchange::copy_c4_to_p4);

impl_packed_to_planar_image!(i32, C3, 3, exchange::copy_c3_to_p3);

impl_packed_to_planar_image!(i32, C4, 4, exchange::copy_c4_to_p4);

impl_packed_to_planar_image!(f32, C3, 3, exchange::copy_c3_to_p3);

impl_packed_to_planar_image!(f32, C4, 4, exchange::copy_c4_to_p4);

impl_planar_to_packed_image!(u8, C3, 3, exchange::copy_p3_to_c3);

impl_planar_to_packed_image!(u8, C4, 4, exchange::copy_p4_to_c4);

impl_planar_to_packed_image!(u16, C3, 3, exchange::copy_p3_to_c3);

impl_planar_to_packed_image!(u16, C4, 4, exchange::copy_p4_to_c4);

impl_planar_to_packed_image!(i16, C3, 3, exchange::copy_p3_to_c3);

impl_planar_to_packed_image!(i16, C4, 4, exchange::copy_p4_to_c4);

impl_planar_to_packed_image!(i32, C3, 3, exchange::copy_p3_to_c3);

impl_planar_to_packed_image!(i32, C4, 4, exchange::copy_p4_to_c4);

impl_planar_to_packed_image!(f32, C3, 3, exchange::copy_p3_to_c3);

impl_planar_to_packed_image!(f32, C4, 4, exchange::copy_p4_to_c4);

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    /// Copies packed channels into a new owned planar image.
    ///
    /// The provided implementations map C3 to three planes and C4 to four planes, retaining
    /// the `u8`, `u16`, `i16`, `i32`, or `f32` sample type and channel order.
    /// Allocates one source-sized image per plane and initializes every logical sample.
    /// Prepares all input and output planes or packed regions, waits for producers on other
    /// streams, rejects capture, and returns after successful native execution and stream
    /// completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, transfer or native execution, or stream synchronization fail.
    /// Allocation, copying borrowed input, or image view construction can also fail.
    pub fn copy_to_planar<const C: usize>(self) -> Result<PlanarImage<T, C>>
    where
        T: SupportedImage<C1>,
        Self: PackedToPlanarImage<T, L, C>,
    {
        self.ensure_eager()?;
        let mut destination =
            PlanarImage::<T, C>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            <Self as PackedToPlanarImage<T, L, C>>::copy_packed_to_planar_image(
                self.stream_context,
                &source,
                &mut destination_view,
            )?;
        }

        // SAFETY: completed native packing dispatch initializes every sample in every plane.
        Ok(unsafe { destination.assume_init() })
    }

    /// Copies planar channels into a new owned packed image.
    ///
    /// The provided implementations map C3 to three planes and C4 to four planes, retaining
    /// the `u8`, `u16`, `i16`, `i32`, or `f32` sample type and channel order.
    /// Allocates a packed image using the pipeline dimensions. Every source plane must
    /// match those dimensions. Existing pipeline pixel values are not copied.
    /// Prepares all input and output planes or packed regions, waits for producers on other
    /// streams, rejects capture, and returns after successful native execution and stream
    /// completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, transfer or native execution, or stream synchronization fail.
    /// Source planes and packed destination dimensions must match.
    /// Allocation, copying borrowed input, or image view construction can also fail.
    pub fn copy_from_planar<const C: usize>(
        self,
        source: &PlanarImageView<'_, T, C>,
    ) -> Result<Self>
    where
        T: SupportedImage<L>,
        Self: PlanarToPackedImage<T, L, C>,
    {
        self.ensure_eager()?;
        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let mut destination_view = destination.output()?;
            <Self as PlanarToPackedImage<T, L, C>>::copy_planar_to_packed_image(
                self.stream_context,
                source,
                &mut destination_view,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}
