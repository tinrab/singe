use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageOutputView, ImageView, ImageViewMut},
    pipeline::image::{ColorTwistMatrix, ImagePipeline, color::ColorTwistBatchImage},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: ColorTwistBatchImage<T, L>,
{
    /// Applies one color-twist matrix per image in a batch.
    ///
    /// Uses one row-major three-by-four matrix per image. For AC4, alpha is not processed.
    /// The common top-left ROI uses the smallest image width and smallest image height
    /// in the batch. Pixels outside that ROI retain their previous values when initialized;
    /// otherwise they remain unwritten. `min` and `max` specify native saturation/clamp
    /// bounds and are forwarded without validation.
    /// Uploads temporary device coefficient and descriptor arrays and retains them through
    /// native completion. Each source/destination pair must have matching dimensions.
    /// Prepares participating regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native color transformation, or synchronization fail.
    /// The batch must contain at least two images and fit `i32`; exactly one twist must
    /// be supplied per image. Source and destination batch lengths and each pair
    /// of image dimensions must match.
    /// Descriptor or coefficient allocation/upload and checked size conversions can fail.
    pub fn color_twist_batch(
        stream_context: &StreamContext,
        sources: &[ImageView<'_, T, L>],
        twists: &[ColorTwistMatrix],
        destinations: &mut [ImageOutputView<'_, T, L>],
        min: f32,
        max: f32,
    ) -> Result<()> {
        <Self as ColorTwistBatchImage<T, L>>::color_twist_batch_image(
            stream_context,
            sources,
            twists,
            destinations,
            min,
            max,
        )
    }

    /// Applies one color-twist matrix per image in a batch.
    ///
    /// Uses one row-major three-by-four matrix per image. For AC4, alpha is not processed.
    /// The common top-left ROI uses the smallest image width and smallest image height
    /// in the batch. Pixels outside that ROI retain their previous values when initialized;
    /// otherwise they remain unwritten. `min` and `max` specify native saturation/clamp
    /// bounds and are forwarded without validation.
    /// Uploads temporary device coefficient and descriptor arrays and retains them through
    /// native completion. Updates the supplied images in place.
    /// Prepares participating regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native color transformation, or synchronization fail.
    /// The batch must contain at least two images and fit `i32`; exactly one twist must
    /// be supplied per image.
    /// Descriptor or coefficient allocation/upload and checked size conversions can fail.
    pub fn color_twist_batch_in_place(
        stream_context: &StreamContext,
        images: &mut [ImageViewMut<'_, T, L>],
        twists: &[ColorTwistMatrix],
        min: f32,
        max: f32,
    ) -> Result<()> {
        <Self as ColorTwistBatchImage<T, L>>::color_twist_batch_image_in_place(
            stream_context,
            images,
            twists,
            min,
            max,
        )
    }
}
