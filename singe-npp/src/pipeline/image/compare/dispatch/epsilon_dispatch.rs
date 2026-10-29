use crate::{
    context::StreamContext,
    error::Result,
    image::{
        compare,
        view::{AC4, C1, C3, C4, ImageView, MaskOutputView},
    },
    pipeline::image::compare::dispatch::ImagePipeline,
};

/// Dispatches completed image comparisons that initialize a one-channel byte mask.
///
/// # Safety
///
/// Implementations must preserve the native element and channel-layout ABI, validate
/// participating regions and output geometry, and complete borrowed native work on every
/// return path. Success must initialize every logical destination sample, including all mask
/// pixels or supplied alpha values where applicable.
pub unsafe trait CompareEqualEpsilonImage<L> {
    /// Compares corresponding pixels in two images for equality within `epsilon`.
    /// Writes a C1 byte mask: `255` when every processed channel satisfies the comparison,
    /// and `0` otherwise. AC4 input alpha is excluded.
    /// `epsilon` is the tolerance applied to absolute sample differences and is forwarded
    /// to NPP without validation.
    /// Writes every logical sample of the supplied destination.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    fn compare_equal_epsilon_image(
        stream_context: &StreamContext,
        source1: &ImageView<'_, f32, L>,
        source2: &ImageView<'_, f32, L>,
        destination: &mut MaskOutputView<'_>,
        epsilon: f32,
    ) -> Result<()>;
}

/// Dispatches completed image comparisons that initialize a one-channel byte mask.
///
/// # Safety
///
/// Implementations must preserve the native element and channel-layout ABI, validate
/// participating regions and output geometry, and complete borrowed native work on every
/// return path. Success must initialize every logical destination sample, including all mask
/// pixels or supplied alpha values where applicable.
pub unsafe trait CompareEqualEpsilonConstantImage<L> {
    /// Scalar constant for C1 or an array with one value per processed source channel.
    /// The provided C3 and AC4 implementations use three values; C4 uses four.
    type Constant;

    /// Compares source pixels with the supplied constant for equality within `epsilon`.
    /// Writes a C1 byte mask: `255` when every processed channel satisfies the comparison,
    /// and `0` otherwise. AC4 input alpha is excluded.
    /// The provided implementations use a scalar constant for C1, three constants for C3
    /// and AC4, and four constants for C4.
    /// `epsilon` is the tolerance applied to absolute sample differences and is forwarded
    /// to NPP without validation.
    /// Writes every logical sample of the supplied destination.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    fn compare_equal_epsilon_constant_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, f32, L>,
        constant: Self::Constant,
        destination: &mut MaskOutputView<'_>,
        epsilon: f32,
    ) -> Result<()>;
}

macro_rules! impl_compare_equal_epsilon_image {
    ($layout:ty, $compare:path) => {
        unsafe impl<'a> CompareEqualEpsilonImage<$layout> for ImagePipeline<'a, f32, $layout> {
            fn compare_equal_epsilon_image(
                stream_context: &StreamContext,
                source1: &ImageView<'_, f32, $layout>,
                source2: &ImageView<'_, f32, $layout>,
                destination: &mut MaskOutputView<'_>,
                epsilon: f32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(destination)?;
                stream_context.prepare_region(source1)?;
                stream_context.prepare_region(source2)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $compare(stream_context, source1, source2, destination, epsilon)
                })
            }
        }
    };
}

macro_rules! impl_compare_equal_epsilon_constant_image {
    ($layout:ty, $constant:ty, $compare:path) => {
        unsafe impl<'a> CompareEqualEpsilonConstantImage<$layout>
            for ImagePipeline<'a, f32, $layout>
        {
            type Constant = $constant;

            fn compare_equal_epsilon_constant_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, f32, $layout>,
                constant: Self::Constant,
                destination: &mut MaskOutputView<'_>,
                epsilon: f32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(destination)?;
                stream_context.prepare_region(source)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $compare(stream_context, source, constant, destination, epsilon)
                })
            }
        }
    };
}

impl_compare_equal_epsilon_image!(C1, compare::compare_equal_epsilon_f32_c1);

impl_compare_equal_epsilon_image!(C3, compare::compare_equal_epsilon_f32_c3);

impl_compare_equal_epsilon_image!(C4, compare::compare_equal_epsilon_f32_c4);

impl_compare_equal_epsilon_image!(AC4, compare::compare_equal_epsilon_f32_ac4);

impl_compare_equal_epsilon_constant_image!(C1, f32, compare::compare_equal_epsilon_constant_f32_c1);

impl_compare_equal_epsilon_constant_image!(
    C3,
    [f32; 3],
    compare::compare_equal_epsilon_constant_f32_c3
);

impl_compare_equal_epsilon_constant_image!(
    C4,
    [f32; 4],
    compare::compare_equal_epsilon_constant_f32_c4
);

impl_compare_equal_epsilon_constant_image!(
    AC4,
    [f32; 3],
    compare::compare_equal_epsilon_constant_f32_ac4
);
