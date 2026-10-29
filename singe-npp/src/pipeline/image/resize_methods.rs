use std::sync::Arc;

use singe_cuda::types::f16;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image,
        geometry::{
            Resize, resize_f16_c1, resize_f16_c3, resize_f16_c4, resize_f32_ac4, resize_f32_c1,
            resize_f32_c3, resize_f32_c4, resize_i16_ac4, resize_i16_c1, resize_i16_c3,
            resize_i16_c4, resize_u8_ac4, resize_u8_c1, resize_u8_c3, resize_u8_c4, resize_u16_ac4,
            resize_u16_c1, resize_u16_c3, resize_u16_c4,
        },
        view::{AC4, C1, C3, C4, ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::image::{ImageBacking, ImagePipeline},
    types::InterpolationMode,
};

/// # Safety
/// Complete borrowed native work on every path and preserve the native element ABI.
/// Validate the source and destination footprint and write valid values to every processed channel.
/// Preserve unprocessed destination values; successful dispatch does not imply full-image coverage.
pub unsafe trait ResizeImage<T, L> {
    fn resize_image(
        stream_context: &StreamContext,
        resize: &Resize,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>;
}

macro_rules! impl_resize_image {
    ($ty:ty, $layout:ty, $resize:path) => {
        unsafe impl<'a> ResizeImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn resize_image(
                stream_context: &StreamContext,
                resize: &Resize,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $resize(stream_context, resize, source, destination) })
            }
        }
    };
}

impl_resize_image!(u8, C1, resize_u8_c1);

impl_resize_image!(u8, C3, resize_u8_c3);

impl_resize_image!(u8, C4, resize_u8_c4);

impl_resize_image!(u8, AC4, resize_u8_ac4);

impl_resize_image!(u16, C1, resize_u16_c1);

impl_resize_image!(u16, C3, resize_u16_c3);

impl_resize_image!(u16, C4, resize_u16_c4);

impl_resize_image!(u16, AC4, resize_u16_ac4);

impl_resize_image!(i16, C1, resize_i16_c1);

impl_resize_image!(i16, C3, resize_i16_c3);

impl_resize_image!(i16, C4, resize_i16_c4);

impl_resize_image!(i16, AC4, resize_i16_ac4);

impl_resize_image!(f16, C1, resize_f16_c1);

impl_resize_image!(f16, C3, resize_f16_c3);

impl_resize_image!(f16, C4, resize_f16_c4);

impl_resize_image!(f32, C1, resize_f32_c1);

impl_resize_image!(f32, C3, resize_f32_c3);

impl_resize_image!(f32, C4, resize_f32_c4);

impl_resize_image!(f32, AC4, resize_f32_ac4);

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: ResizeImage<T, L>,
{
    /// Resizes the selected source ROI into the destination ROI.
    ///
    /// `resize` supplies both ROIs and the interpolation method.
    /// Pixels outside the transformed source intersection, ignored alpha, and padding can
    /// remain unwritten. Initialize uncovered locations before reading a complete image.
    /// Prepares all participating image regions, waits for their producers
    /// on other streams, rejects capture, and returns after successful stream completion.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// The source ROI must have positive dimensions and representable endpoints. The
    /// destination ROI must fit within the destination image. NPP may reject geometry
    /// or interpolation parameters.
    pub fn resize_into(
        stream_context: &StreamContext,
        resize: &Resize,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()> {
        <Self as ResizeImage<T, L>>::resize_image(stream_context, resize, source, destination)
    }

    /// Resizes the selected source ROI into the destination ROI.
    ///
    /// `resize` supplies both ROIs and the interpolation method.
    /// Takes ownership of the supplied initialized `destination` and returns it in the
    /// resulting pipeline. Pixels outside native writes and ignored alpha retain their
    /// destination values.
    /// Prepares all participating image regions, waits for their producers
    /// on other streams, rejects capture, and returns after successful stream completion.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// The source ROI must have positive dimensions and representable endpoints. The
    /// destination ROI must fit within the destination image. NPP may reject geometry
    /// or interpolation parameters.
    /// Image view construction or capture-time destination retention can also fail.
    pub fn resize_with(self, resize: Resize, mut destination: Image<T, L>) -> Result<Self> {
        self.stream_context
            .stream()
            .defer_drop_until_capture_end(Arc::clone(destination.allocation()))?;
        self.ensure_eager()?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output_view()?;
            <Self as ResizeImage<T, L>>::resize_image(
                self.stream_context,
                &resize,
                &source,
                &mut destination_view,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }

    /// Resizes the full source into an initialized image of the desired dimensions.
    ///
    /// Builds a `Resize` request using both full-image ROIs and `interpolation`.
    /// Takes ownership of the supplied initialized `destination` and returns it in the
    /// resulting pipeline. Pixels outside native writes and ignored alpha retain their
    /// destination values.
    /// Prepares all participating image regions, waits for their producers
    /// on other streams, rejects capture, and returns after successful stream completion.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// The source ROI must have positive dimensions and representable endpoints. The
    /// destination ROI must fit within the destination image. NPP may reject geometry
    /// or interpolation parameters.
    /// Image view construction or capture-time destination retention can also fail.
    pub fn resize(
        self,
        interpolation: InterpolationMode,
        destination: Image<T, L>,
    ) -> Result<Self> {
        let size = destination.size();
        let source_size = self.size();
        let resize = Resize {
            source_roi: source_size.rectangle(),
            destination_roi: size.rectangle(),
            interpolation,
        };
        self.resize_with(resize, destination)
    }
}
