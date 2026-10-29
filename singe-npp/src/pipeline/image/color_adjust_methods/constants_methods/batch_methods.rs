use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C4, ImageOutputView, ImageView, ImageViewMut},
    pipeline::image::{
        ColorTwistBatchConstantsMatrix4, ImagePipeline, color::ColorTwistBatchWithConstantsImage,
    },
};

impl<T> ImagePipeline<'_, T, C4>
where
    T: Copy,
    Self: ColorTwistBatchWithConstantsImage<T>,
{
    /// Applies one color-twist matrix and additive vector per image in a batch.
    ///
    /// Uses C4 transforms stored as four rows of five coefficients, with the last entry
    /// in each row supplying the additive constant.
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
    pub fn color_twist_batch_with_constants(
        stream_context: &StreamContext,
        sources: &[ImageView<'_, T, C4>],
        twists: &[ColorTwistBatchConstantsMatrix4],
        destinations: &mut [ImageOutputView<'_, T, C4>],
        min: f32,
        max: f32,
    ) -> Result<()> {
        <Self as ColorTwistBatchWithConstantsImage<T>>::color_twist_batch_with_constants_image(
            stream_context,
            sources,
            twists,
            destinations,
            min,
            max,
        )
    }

    /// Applies one color-twist matrix and additive vector per image in a batch.
    ///
    /// Uses C4 transforms stored as four rows of five coefficients, with the last entry
    /// in each row supplying the additive constant.
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
    pub fn color_twist_batch_with_constants_in_place(
        stream_context: &StreamContext,
        images: &mut [ImageViewMut<'_, T, C4>],
        twists: &[ColorTwistBatchConstantsMatrix4],
        min: f32,
        max: f32,
    ) -> Result<()> {
        <Self as ColorTwistBatchWithConstantsImage<
            T,
        >>::color_twist_batch_with_constants_image_in_place(stream_context, images, twists, min, max)
    }
}
