use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage, exchange,
        view::{C1, C3, C4, ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::{ImageBacking, image::ImagePipeline},
    types::Size,
};

/// # Safety
/// Complete borrowed native work on every path, preserve the native ABI, and initialize every transposed destination channel.
pub unsafe trait TransposeImage<T, L> {
    fn transpose_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>;
}

macro_rules! impl_transpose_image {
    ($ty:ty, $layout:ty, $transpose:path) => {
        unsafe impl<'a> TransposeImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn transpose_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $transpose(stream_context, source, destination) })
            }
        }
    };
}

impl_transpose_image!(u8, C1, exchange::transpose_c1);

impl_transpose_image!(u8, C3, exchange::transpose_c3);

impl_transpose_image!(u8, C4, exchange::transpose_c4);

impl_transpose_image!(u16, C1, exchange::transpose_c1);

impl_transpose_image!(u16, C3, exchange::transpose_c3);

impl_transpose_image!(u16, C4, exchange::transpose_c4);

impl_transpose_image!(i16, C1, exchange::transpose_c1);

impl_transpose_image!(i16, C3, exchange::transpose_c3);

impl_transpose_image!(i16, C4, exchange::transpose_c4);

impl_transpose_image!(i32, C1, exchange::transpose_c1);

impl_transpose_image!(i32, C3, exchange::transpose_c3);

impl_transpose_image!(i32, C4, exchange::transpose_c4);

impl_transpose_image!(f32, C1, exchange::transpose_c1);

impl_transpose_image!(f32, C3, exchange::transpose_c3);

impl_transpose_image!(f32, C4, exchange::transpose_c4);

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: TransposeImage<T, L>,
{
    /// Transposes image pixels, exchanging horizontal and vertical coordinates.
    ///
    /// The provided implementations retain `u8`, `u16`, `i16`, `i32`, or `f32` samples in
    /// C1, C3, or C4 layouts. Destination width must equal source height and destination
    /// height must equal source width.
    /// Prepares all participating image regions, waits for their producers
    /// on other streams, rejects capture, and returns after successful stream completion.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// Destination dimensions must be the exchanged source dimensions.
    pub fn transpose_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()> {
        <Self as TransposeImage<T, L>>::transpose_image(stream_context, source, destination)
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: TransposeImage<T, L>,
{
    /// Transposes image pixels, exchanging horizontal and vertical coordinates.
    ///
    /// The provided implementations retain `u8`, `u16`, `i16`, `i32`, or `f32` samples in
    /// C1, C3, or C4 layouts. Allocates a new image with source width and height
    /// exchanged. Every logical output channel is written.
    /// Prepares all participating image regions, waits for their producers
    /// on other streams, rejects capture, and returns after successful stream completion.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// Destination dimensions must be the exchanged source dimensions.
    /// Image view construction, output allocation, or copying source alpha can also fail.
    pub fn transpose(self) -> Result<Self> {
        self.ensure_eager()?;
        let size = self.size();
        let destination_size = Size {
            width: size.height,
            height: size.width,
        };
        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), destination_size)?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            <Self as TransposeImage<T, L>>::transpose_image(
                self.stream_context,
                &source,
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
