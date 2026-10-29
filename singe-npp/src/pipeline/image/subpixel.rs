use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage, exchange,
        view::{AC4, C1, C3, C4, ChannelLayout, ImageOutputView, ImageWindow},
    },
    pipeline::{ImageBacking, image::ImagePipeline},
    types::{Point, Rectangle, Size},
};

/// # Safety
/// Preserve the native interpolation ABI, check the two-by-two source footprint, and complete borrowed work on every path.
/// On success, initialize every output sample except alpha for layouts that explicitly ignore it.
pub unsafe trait CopySubpixelImage<T, L> {
    fn copy_subpixel_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        x_shift: f32,
        y_shift: f32,
    ) -> Result<()>;
}

macro_rules! impl_copy_subpixel_image {
    ($ty:ty, $layout:ty, $copy:path) => {
        unsafe impl<'a> CopySubpixelImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn copy_subpixel_image(
                stream_context: &StreamContext,
                source: &ImageWindow<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                x_shift: f32,
                y_shift: f32,
            ) -> Result<()> {
                source.require_halo(Size::new(2, 2), Point { x: 0, y: 0 })?;
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $copy(
                        stream_context,
                        &source.view(),
                        destination,
                        x_shift,
                        y_shift,
                    )
                })
            }
        }
    };
}

impl_copy_subpixel_image!(u8, C1, exchange::copy_subpixel_c1);

impl_copy_subpixel_image!(u8, C3, exchange::copy_subpixel_c3);

impl_copy_subpixel_image!(u8, C4, exchange::copy_subpixel_c4);

impl_copy_subpixel_image!(u8, AC4, exchange::copy_subpixel_ac4);

impl_copy_subpixel_image!(u16, C1, exchange::copy_subpixel_c1);

impl_copy_subpixel_image!(u16, C3, exchange::copy_subpixel_c3);

impl_copy_subpixel_image!(u16, C4, exchange::copy_subpixel_c4);

impl_copy_subpixel_image!(u16, AC4, exchange::copy_subpixel_ac4);

impl_copy_subpixel_image!(i16, C1, exchange::copy_subpixel_c1);

impl_copy_subpixel_image!(i16, C3, exchange::copy_subpixel_c3);

impl_copy_subpixel_image!(i16, C4, exchange::copy_subpixel_c4);

impl_copy_subpixel_image!(i16, AC4, exchange::copy_subpixel_ac4);

impl_copy_subpixel_image!(i32, C1, exchange::copy_subpixel_c1);

impl_copy_subpixel_image!(i32, C3, exchange::copy_subpixel_c3);

impl_copy_subpixel_image!(i32, C4, exchange::copy_subpixel_c4);

impl_copy_subpixel_image!(i32, AC4, exchange::copy_subpixel_ac4);

impl_copy_subpixel_image!(f32, C1, exchange::copy_subpixel_c1);

impl_copy_subpixel_image!(f32, C3, exchange::copy_subpixel_c3);

impl_copy_subpixel_image!(f32, C4, exchange::copy_subpixel_c4);

impl_copy_subpixel_image!(f32, AC4, exchange::copy_subpixel_ac4);

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: CopySubpixelImage<T, L>,
{
    /// Copies samples at fractional pixel shifts using native linear interpolation.
    ///
    /// Both `x_shift` and `y_shift` must be finite and in `0.0..1.0`. The source window
    /// must retain a two-by-two parent-image footprint anchored at `(0, 0)`, including one
    /// extra source column on the right and row below, even when a shift is zero.
    /// Destination width and height must not exceed the source window dimensions. For AC4,
    /// alpha remains unwritten.
    /// Prepares source and destination regions, waits for producers on other streams,
    /// rejects capture, and returns after successful native execution and stream completion.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native interpolation, or synchronization fail. Invalid shifts,
    /// missing parent halo, or destination dimensions larger than the source window are
    /// rejected.
    pub fn copy_subpixel_into(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        x_shift: f32,
        y_shift: f32,
    ) -> Result<()> {
        <Self as CopySubpixelImage<T, L>>::copy_subpixel_image(
            stream_context,
            source,
            destination,
            x_shift,
            y_shift,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: CopySubpixelImage<T, L>,
{
    /// Copies samples at fractional pixel shifts using native linear interpolation.
    ///
    /// Both `x_shift` and `y_shift` must be finite and in `0.0..1.0`. The source window
    /// must retain a two-by-two parent-image footprint anchored at `(0, 0)`, including one
    /// extra source column on the right and row below, even when a shift is zero.
    /// Uses a top-left source window with the requested `size` and allocates a new owned
    /// image of that size. For AC4, copies the source window first to preserve its alpha.
    /// Prepares source and destination regions, waits for producers on other streams,
    /// rejects capture, and returns after successful native execution and stream completion.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native interpolation, or synchronization fail. Invalid shifts,
    /// missing parent halo, or destination dimensions larger than the source window are
    /// rejected.
    /// Source window construction, allocation, output view construction, or copying source
    /// alpha can also fail.
    pub fn copy_subpixel(self, size: Size, x_shift: f32, y_shift: f32) -> Result<Self> {
        self.ensure_eager()?;
        let source = self.view()?.window(Rectangle {
            x: 0,
            y: 0,
            width: size.width,
            height: size.height,
        })?;
        source.require_halo(Size::new(2, 2), Point { x: 0, y: 0 })?;
        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), size)?;

        {
            let mut destination_view = destination.output()?;
            if L::IGNORES_ALPHA {
                destination_view.copy_from(&source.view())?;
            }
            <Self as CopySubpixelImage<T, L>>::copy_subpixel_image(
                self.stream_context,
                &source,
                &mut destination_view,
                x_shift,
                y_shift,
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
