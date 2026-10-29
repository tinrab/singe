use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{C1, ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::image::{
        GradientVector, ImagePipeline,
        filtering::{GradientVectorBorderImage, HistogramOfGradientsBorderImage},
    },
    types::{BorderType, ImageNormalization, MaskSize, Point},
};

mod histogram_of_gradients_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    /// Computes Prewitt X and Y derivatives, magnitude, and angle into four owned images.
    ///
    /// Returns C1 X, Y, and magnitude images of type `G` and a C1 `f32` angle image.
    /// The provided implementations accept C1 or C3 input: `u8` produces `i16` gradients,
    /// while `i16`, `u16`, and `f32` produce `f32` gradients. For C3, the channel with the
    /// largest Euclidean gradient supplies the output gradient. Magnitude uses `norm`: L1
    /// sums absolute components, and L2 takes their Euclidean length. The angle is
    /// `atan2(y, x)`.
    /// NPP supports only a 3-by-3 mask, `ImageNormalization::L1` or
    /// `ImageNormalization::L2`, and `BorderType::Replicate`. These selectors are forwarded
    /// to NPP without a wrapper whitelist.
    /// `source_offset` locates the output ROI in the full source image. The new owned
    /// outputs extend from that offset to the lower-right source edge.
    /// Writes all logical output samples; row padding remains unwritten.
    /// Prepares participating regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail. The source offset
    /// must address a source pixel; output-sized ROI endpoints must not overflow or exceed
    /// the source image.
    /// Output allocation or image view construction can also fail.
    pub fn gradient_vector_prewitt_border_to<G>(
        self,
        source_offset: Point,
        mask_size: MaskSize,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<GradientVector<G>>
    where
        G: Copy + SupportedImage<C1>,
        f32: SupportedImage<C1>,
        Self: GradientVectorBorderImage<T, L, G>,
    {
        self.gradient_vector_border(
            source_offset,
            mask_size,
            norm,
            border_type,
            <Self as GradientVectorBorderImage<T, L, G>>::gradient_vector_prewitt_border_image,
        )
    }

    /// Computes Scharr X and Y derivatives, magnitude, and angle into four owned images.
    ///
    /// Returns C1 X, Y, and magnitude images of type `G` and a C1 `f32` angle image.
    /// The provided implementations accept C1 or C3 input: `u8` produces `i16` gradients,
    /// while `i16`, `u16`, and `f32` produce `f32` gradients. For C3, the channel with the
    /// largest Euclidean gradient supplies the output gradient. Magnitude uses `norm`: L1
    /// sums absolute components, and L2 takes their Euclidean length. The angle is
    /// `atan2(y, x)`.
    /// NPP supports only a 3-by-3 mask, `ImageNormalization::L1` or
    /// `ImageNormalization::L2`, and `BorderType::Replicate`. These selectors are forwarded
    /// to NPP without a wrapper whitelist.
    /// `source_offset` locates the output ROI in the full source image. The new owned
    /// outputs extend from that offset to the lower-right source edge.
    /// Writes all logical output samples; row padding remains unwritten.
    /// Prepares participating regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail. The source offset
    /// must address a source pixel; output-sized ROI endpoints must not overflow or exceed
    /// the source image.
    /// Output allocation or image view construction can also fail.
    pub fn gradient_vector_scharr_border_to<G>(
        self,
        source_offset: Point,
        mask_size: MaskSize,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<GradientVector<G>>
    where
        G: Copy + SupportedImage<C1>,
        f32: SupportedImage<C1>,
        Self: GradientVectorBorderImage<T, L, G>,
    {
        self.gradient_vector_border(
            source_offset,
            mask_size,
            norm,
            border_type,
            <Self as GradientVectorBorderImage<T, L, G>>::gradient_vector_scharr_border_image,
        )
    }

    /// Computes Sobel X and Y derivatives, magnitude, and angle into four owned images.
    ///
    /// Returns C1 X, Y, and magnitude images of type `G` and a C1 `f32` angle image.
    /// The provided implementations accept C1 or C3 input: `u8` produces `i16` gradients,
    /// while `i16`, `u16`, and `f32` produce `f32` gradients. For C3, the channel with the
    /// largest Euclidean gradient supplies the output gradient. Magnitude uses `norm`: L1
    /// sums absolute components, and L2 takes their Euclidean length. The angle is
    /// `atan2(y, x)`.
    /// NPP supports 3-by-3 and 5-by-5 masks, `ImageNormalization::L1` or
    /// `ImageNormalization::L2`, and `BorderType::Replicate`. These selectors are forwarded
    /// to NPP without a wrapper whitelist.
    /// `source_offset` locates the output ROI in the full source image. The new owned
    /// outputs extend from that offset to the lower-right source edge.
    /// Writes all logical output samples; row padding remains unwritten.
    /// Prepares participating regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail. The source offset
    /// must address a source pixel; output-sized ROI endpoints must not overflow or exceed
    /// the source image.
    /// Output allocation or image view construction can also fail.
    pub fn gradient_vector_sobel_border_to<G>(
        self,
        source_offset: Point,
        mask_size: MaskSize,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<GradientVector<G>>
    where
        G: Copy + SupportedImage<C1>,
        f32: SupportedImage<C1>,
        Self: GradientVectorBorderImage<T, L, G>,
    {
        self.gradient_vector_border(
            source_offset,
            mask_size,
            norm,
            border_type,
            <Self as GradientVectorBorderImage<T, L, G>>::gradient_vector_sobel_border_image,
        )
    }

    fn gradient_vector_border<G>(
        self,
        source_offset: Point,
        mask_size: MaskSize,
        norm: ImageNormalization,
        border_type: BorderType,
        operation: GradientVectorBorderCallback<G, L, T>,
    ) -> Result<GradientVector<G>>
    where
        G: Copy + SupportedImage<C1>,
        f32: SupportedImage<C1>,
    {
        self.ensure_eager()?;
        let source = self.view()?;
        let size = source.window_from(source_offset)?.size();
        let mut x = Image::<G, C1>::create_uninit(self.stream_context.cuda_context(), size)?;
        let mut y = Image::<G, C1>::create_uninit(self.stream_context.cuda_context(), size)?;
        let mut magnitude =
            Image::<G, C1>::create_uninit(self.stream_context.cuda_context(), size)?;
        let mut angle = Image::<f32, C1>::create_uninit(self.stream_context.cuda_context(), size)?;

        {
            let mut x_view = x.output()?;
            let mut y_view = y.output()?;
            let mut magnitude_view = magnitude.output()?;
            let mut angle_view = angle.output()?;
            operation(
                self.stream_context,
                &source,
                source_offset,
                &mut x_view,
                &mut y_view,
                &mut magnitude_view,
                &mut angle_view,
                mask_size,
                norm,
                border_type,
            )?;
        }

        Ok(GradientVector {
            // SAFETY: completed native dispatch initializes all four requested ROI outputs.
            x: unsafe { x.assume_init() },
            y: unsafe { y.assume_init() },
            magnitude: unsafe { magnitude.assume_init() },
            // SAFETY: the completed operation above wrote every element.
            angle: unsafe { angle.assume_init() },
        })
    }
}

type GradientVectorBorderCallback<G, L, T> = fn(
    &StreamContext,
    &ImageView<'_, T, L>,
    Point,
    &mut ImageOutputView<'_, G, C1>,
    &mut ImageOutputView<'_, G, C1>,
    &mut ImageOutputView<'_, G, C1>,
    &mut ImageOutputView<'_, f32, C1>,
    MaskSize,
    ImageNormalization,
    BorderType,
) -> Result<()>;
