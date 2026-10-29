use crate::{
    context::StreamContext,
    error::Result,
    image::view::{AC4, C1, C3, ImageOutputView, ImageView},
    types::{BayerGridPosition, ImageNormalization, Rectangle},
};

/// Dispatches completed packed RGB to one-channel grayscale conversion.
///
/// # Safety
///
/// Implementations must preserve the native element and channel-layout ABI, validate
/// participating regions and output geometry, and complete borrowed native work on every
/// return path. Success must initialize every logical destination sample, including all mask
/// pixels or supplied alpha values where applicable.
pub unsafe trait RgbToGrayImage<T, L> {
    /// Converts packed RGB to one-channel gray using the native fixed weights.
    /// Uses `0.299 * R + 0.587 * G + 0.114 * B` for gamma-corrected RGB.
    /// AC4 input alpha is ignored.
    /// Writes every logical sample of the supplied destination.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Source and destination image dimensions must match.
    fn rgb_to_gray_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, C1>,
    ) -> Result<()>;
}

/// Dispatches completed grayscale conversion with caller-supplied channel weights.
///
/// # Safety
///
/// Implementations must preserve the native element and channel-layout ABI, validate
/// participating regions and output geometry, and complete borrowed native work on every
/// return path. Success must initialize every logical destination sample, including all mask
/// pixels or supplied alpha values where applicable.
pub unsafe trait ColorToGrayImage<T, L, const COEFFICIENTS: usize> {
    /// Converts packed color to one-channel gray using `coefficients`.
    /// Passes the host coefficient array directly to NPP without validating its values.
    /// Each output gray sample is the sum of the processed channels multiplied by their weights.
    /// The provided C3 and AC4 implementations use three weights, ignoring AC4 input alpha;
    /// C4 uses four weights and includes its fourth channel.
    /// Writes every logical sample of the supplied destination.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Source and destination image dimensions must match.
    fn color_to_gray_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, C1>,
        coefficients: &[f32; COEFFICIENTS],
    ) -> Result<()>;
}

/// Dispatches completed RGB to grayscale gradient conversion.
///
/// # Safety
///
/// Implementations must preserve the native element and channel-layout ABI, validate
/// participating regions and output geometry, and complete borrowed native work on every
/// return path. Success must initialize every logical destination sample, including all mask
/// pixels or supplied alpha values where applicable.
pub unsafe trait GradientColorToGrayImage<T> {
    /// Converts packed RGB to a one-channel gray gradient.
    /// `normalization` selects the native gradient distance method.
    /// Writes every logical sample of the supplied destination.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Source and destination image dimensions must match.
    fn gradient_color_to_gray_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C3>,
        destination: &mut ImageOutputView<'_, T, C1>,
        normalization: ImageNormalization,
    ) -> Result<()>;
}

/// Dispatches completed Bayer CFA to packed RGB conversion.
///
/// # Safety
///
/// Implementations must preserve the native element and channel-layout ABI, validate
/// participating regions and output geometry, and complete borrowed native work on every
/// return path. Success must initialize every logical destination sample, including all mask
/// pixels or supplied alpha values where applicable.
pub unsafe trait CfaToRgbImage<T> {
    /// Converts a one-channel Bayer CFA image to packed RGB.
    /// `source_roi` selects pixels in the full source image. `grid` specifies the Bayer
    /// registration at the ROI origin. Uses the native required undefined interpolation setting.
    /// Writes every logical sample of the supplied destination.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source ROI must have a nonnegative origin and positive dimensions, fit inside
    /// the source without coordinate overflow, and match the destination dimensions.
    fn cfa_to_rgb_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        source_roi: Rectangle,
        destination: &mut ImageOutputView<'_, T, C3>,
        grid: BayerGridPosition,
    ) -> Result<()>;
}

/// Dispatches completed Bayer CFA to packed RGB conversion with a supplied alpha value.
///
/// # Safety
///
/// Implementations must preserve the native element and channel-layout ABI, validate
/// participating regions and output geometry, and complete borrowed native work on every
/// return path. Success must initialize every logical destination sample, including all mask
/// pixels or supplied alpha values where applicable.
pub unsafe trait CfaToRgbaImage<T> {
    /// Converts a one-channel Bayer CFA image to packed RGB with supplied alpha.
    /// `source_roi` selects pixels in the full source image. `grid` specifies the Bayer
    /// registration at the ROI origin. Uses the native required undefined interpolation setting.
    /// Writes `alpha` to every destination alpha channel, including the AC4 layout.
    /// Writes every logical sample of the supplied destination.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source ROI must have a nonnegative origin and positive dimensions, fit inside
    /// the source without coordinate overflow, and match the destination dimensions.
    fn cfa_to_rgba_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        source_roi: Rectangle,
        destination: &mut ImageOutputView<'_, T, AC4>,
        grid: BayerGridPosition,
        alpha: T,
    ) -> Result<()>;
}

macro_rules! impl_rgb_to_gray_image {
    ($ty:ty, $layout:ty, $convert:path) => {
        unsafe impl<'a> RgbToGrayImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn rgb_to_gray_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, C1>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $convert(stream_context, source, destination) })
            }
        }
    };
}

macro_rules! impl_color_to_gray_image {
    ($ty:ty, $layout:ty, $coefficients:literal, $convert:path) => {
        unsafe impl<'a> ColorToGrayImage<$ty, $layout, $coefficients>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn color_to_gray_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, C1>,
                coefficients: &[f32; $coefficients],
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $convert(stream_context, source, destination, coefficients)
                })
            }
        }
    };
}

macro_rules! impl_gradient_color_to_gray_image {
    ($ty:ty, $convert:path) => {
        unsafe impl<'a> GradientColorToGrayImage<$ty> for ImagePipeline<'a, $ty, C3> {
            fn gradient_color_to_gray_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C3>,
                destination: &mut ImageOutputView<'_, $ty, C1>,
                normalization: ImageNormalization,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $convert(stream_context, source, destination, normalization)
                })
            }
        }
    };
}

macro_rules! impl_cfa_to_rgb_image {
    ($ty:ty, $cfa_to_rgb:path) => {
        unsafe impl<'a> CfaToRgbImage<$ty> for ImagePipeline<'a, $ty, C1> {
            fn cfa_to_rgb_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                source_roi: Rectangle,
                destination: &mut ImageOutputView<'_, $ty, C3>,
                grid: BayerGridPosition,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $cfa_to_rgb(stream_context, source, source_roi, destination, grid)
                })
            }
        }
    };
}

macro_rules! impl_cfa_to_rgba_image {
    ($ty:ty, $cfa_to_rgba:path) => {
        unsafe impl<'a> CfaToRgbaImage<$ty> for ImagePipeline<'a, $ty, C1> {
            fn cfa_to_rgba_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                source_roi: Rectangle,
                destination: &mut ImageOutputView<'_, $ty, AC4>,
                grid: BayerGridPosition,
                alpha: $ty,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $cfa_to_rgba(stream_context, source, source_roi, destination, grid, alpha)
                })
            }
        }
    };
}
