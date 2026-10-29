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
            filtering::{AdaptiveBoxThresholdBorderImage, WienerBorderFilterImage},
        },
    },
    types::{BorderType, Point, Size},
};

mod typed_edge_to_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    /// Applies adaptive Wiener noise removal with source border control.
    ///
    /// `source_offset` locates the output ROI in the full source image. The destination
    /// dimensions select its extent.
    /// Uses local mean and variance from `mask_size` around `anchor`. `noise` contains one
    /// variance level per processed channel: one for C1, three for C3 or AC4, and four for
    /// C4. NPP expects values in `0.0..=1.0`; the wrapper forwards the mutable array
    /// directly without validating its values. A zero noise level requests the native
    /// local variance output for that channel. Only `BorderType::Replicate` is accepted.
    /// For AC4, writes the three color channels and leaves destination alpha untouched.
    /// Row padding remains unwritten.
    /// Prepares image regions and waits for producers on other streams. Rejects stream
    /// capture and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// The source offset must address a source pixel; destination-sized ROI endpoints must
    /// not overflow or exceed the source image.
    /// Mask dimensions must be positive, the anchor must lie within the mask, and the
    /// border type must be `BorderType::Replicate`.
    pub fn filter_wiener_border_into<const CHANNELS: usize>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: Size,
        anchor: Point,
        noise: &mut [f32; CHANNELS],
        border_type: BorderType,
    ) -> Result<()>
    where
        Self: WienerBorderFilterImage<T, L, CHANNELS>,
    {
        <Self as WienerBorderFilterImage<T, L, CHANNELS>>::filter_wiener_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            anchor,
            noise,
            border_type,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: AdaptiveBoxThresholdBorderImage<T, L>,
{
    /// Applies adaptive box thresholding with source border control.
    ///
    /// `source_offset` locates the output ROI in the full source image. The destination
    /// dimensions select its extent.
    /// The provided implementation accepts C1 `u8` images. A pixel greater than its local
    /// box mean minus `delta` becomes `value_greater_than`; other pixels become
    /// `value_less_or_equal`. NPP requires a square mask with odd side lengths and supports
    /// only `BorderType::Replicate`. The wrapper checks positive mask dimensions and
    /// forwards the remaining native restrictions and `delta` without validating them.
    /// Row padding remains unwritten.
    /// Prepares image regions and waits for producers on other streams. Rejects stream
    /// capture and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// The source offset must address a source pixel; destination-sized ROI endpoints must
    /// not overflow or exceed the source image.
    /// Mask dimensions must be positive; NPP can reject other mask or border parameters.
    pub fn filter_threshold_adaptive_box_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: Size,
        delta: f32,
        value_greater_than: T,
        value_less_or_equal: T,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as AdaptiveBoxThresholdBorderImage<T, L>>::filter_threshold_adaptive_box_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            delta,
            value_greater_than,
            value_less_or_equal,
            border_type,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
{
    /// Applies adaptive Wiener noise removal with source border control.
    ///
    /// `source_offset` locates the output ROI in the full source image. Allocates a new
    /// owned image from that offset to the lower-right source edge.
    /// Uses local mean and variance from `mask_size` around `anchor`. `noise` contains one
    /// variance level per processed channel: one for C1, three for C3 or AC4, and four for
    /// C4. NPP expects values in `0.0..=1.0`; the wrapper forwards the mutable array
    /// directly without validating its values. A zero noise level requests the native
    /// local variance output for that channel. Only `BorderType::Replicate` is accepted.
    /// For AC4, copies the source window before filtering to preserve its alpha values.
    /// Row padding remains unwritten.
    /// Prepares image regions and waits for producers on other streams. Rejects stream
    /// capture and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// The source offset must address a source pixel; destination-sized ROI endpoints must
    /// not overflow or exceed the source image.
    /// Mask dimensions must be positive, the anchor must lie within the mask, and the
    /// border type must be `BorderType::Replicate`.
    /// Output allocation, image view construction, or copying source alpha can also fail.
    pub fn filter_wiener_border<const CHANNELS: usize>(
        self,
        source_offset: Point,
        mask_size: Size,
        anchor: Point,
        noise: &mut [f32; CHANNELS],
        border_type: BorderType,
    ) -> Result<Self>
    where
        Self: WienerBorderFilterImage<T, L, CHANNELS>,
    {
        self.ensure_eager()?;
        let source = self.view()?;
        let output_source = source.window_from(source_offset)?;
        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), output_source.size())?;

        {
            let mut destination_view = destination.output()?;
            if L::IGNORES_ALPHA {
                destination_view.copy_from(&output_source.view())?;
            }
            <Self as WienerBorderFilterImage<T, L, CHANNELS>>::filter_wiener_border_image(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                mask_size,
                anchor,
                noise,
                border_type,
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
