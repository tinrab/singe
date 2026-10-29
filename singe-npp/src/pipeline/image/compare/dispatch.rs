use crate::{
    context::StreamContext,
    error::Result,
    image::{
        compare,
        view::{AC4, C1, C3, C4, ImageView, MaskOutputView},
    },
    pipeline::image::compare::ImagePipeline,
    types::ComparisonOperation,
};

mod epsilon_dispatch;

pub use epsilon_dispatch::{CompareEqualEpsilonConstantImage, CompareEqualEpsilonImage};

/// Dispatches completed image comparisons that initialize a one-channel byte mask.
///
/// # Safety
///
/// Implementations must preserve the native element and channel-layout ABI, validate
/// participating regions and output geometry, and complete borrowed native work on every
/// return path. Success must initialize every logical destination sample, including all mask
/// pixels or supplied alpha values where applicable.
pub unsafe trait CompareImage<T, L> {
    /// Compares corresponding pixels in two images using `operation`.
    /// Writes a C1 byte mask: `255` when every processed channel satisfies the comparison,
    /// and `0` otherwise. AC4 input alpha is excluded.
    /// Writes every logical sample of the supplied destination.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    fn compare_image(
        stream_context: &StreamContext,
        source1: &ImageView<'_, T, L>,
        source2: &ImageView<'_, T, L>,
        destination: &mut MaskOutputView<'_>,
        operation: ComparisonOperation,
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
pub unsafe trait CompareConstantImage<T, L> {
    /// Scalar constant for C1 or an array with one value per processed source channel.
    /// The provided C3 and AC4 implementations use three values; C4 uses four.
    type Constant;

    /// Compares source pixels with the supplied constant using `operation`.
    /// Writes a C1 byte mask: `255` when every processed channel satisfies the comparison,
    /// and `0` otherwise. AC4 input alpha is excluded.
    /// The provided implementations use a scalar constant for C1, three constants for C3
    /// and AC4, and four constants for C4.
    /// Writes every logical sample of the supplied destination.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    fn compare_constant_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: Self::Constant,
        destination: &mut MaskOutputView<'_>,
        operation: ComparisonOperation,
    ) -> Result<()>;
}

macro_rules! impl_compare_image {
    ($ty:ty, $layout:ty, $compare:path) => {
        unsafe impl<'a> CompareImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn compare_image(
                stream_context: &StreamContext,
                source1: &ImageView<'_, $ty, $layout>,
                source2: &ImageView<'_, $ty, $layout>,
                destination: &mut MaskOutputView<'_>,
                operation: ComparisonOperation,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(destination)?;
                stream_context.prepare_region(source1)?;
                stream_context.prepare_region(source2)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $compare(stream_context, source1, source2, destination, operation)
                })
            }
        }
    };
}

macro_rules! impl_compare_constant_image {
    ($ty:ty, $layout:ty, $constant:ty, $compare:path) => {
        unsafe impl<'a> CompareConstantImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            type Constant = $constant;

            fn compare_constant_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                constant: Self::Constant,
                destination: &mut MaskOutputView<'_>,
                operation: ComparisonOperation,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(destination)?;
                stream_context.prepare_region(source)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $compare(stream_context, source, constant, destination, operation)
                })
            }
        }
    };
}

impl_compare_image!(u8, C1, compare::compare_u8_c1);

impl_compare_image!(u8, C3, compare::compare_u8_c3);

impl_compare_image!(u8, C4, compare::compare_u8_c4);

impl_compare_image!(u8, AC4, compare::compare_u8_ac4);

impl_compare_image!(u16, C1, compare::compare_u16_c1);

impl_compare_image!(u16, C3, compare::compare_u16_c3);

impl_compare_image!(u16, C4, compare::compare_u16_c4);

impl_compare_image!(u16, AC4, compare::compare_u16_ac4);

impl_compare_image!(i16, C1, compare::compare_i16_c1);

impl_compare_image!(i16, C3, compare::compare_i16_c3);

impl_compare_image!(i16, C4, compare::compare_i16_c4);

impl_compare_image!(i16, AC4, compare::compare_i16_ac4);

impl_compare_image!(f32, C1, compare::compare_f32_c1);

impl_compare_image!(f32, C3, compare::compare_f32_c3);

impl_compare_image!(f32, C4, compare::compare_f32_c4);

impl_compare_image!(f32, AC4, compare::compare_f32_ac4);

impl_compare_constant_image!(u8, C1, u8, compare::compare_constant_u8_c1);

impl_compare_constant_image!(u8, C3, [u8; 3], compare::compare_constant_u8_c3);

impl_compare_constant_image!(u8, C4, [u8; 4], compare::compare_constant_u8_c4);

impl_compare_constant_image!(u8, AC4, [u8; 3], compare::compare_constant_u8_ac4);

impl_compare_constant_image!(u16, C1, u16, compare::compare_constant_u16_c1);

impl_compare_constant_image!(u16, C3, [u16; 3], compare::compare_constant_u16_c3);

impl_compare_constant_image!(u16, C4, [u16; 4], compare::compare_constant_u16_c4);

impl_compare_constant_image!(u16, AC4, [u16; 3], compare::compare_constant_u16_ac4);

impl_compare_constant_image!(i16, C1, i16, compare::compare_constant_i16_c1);

impl_compare_constant_image!(i16, C3, [i16; 3], compare::compare_constant_i16_c3);

impl_compare_constant_image!(i16, C4, [i16; 4], compare::compare_constant_i16_c4);

impl_compare_constant_image!(i16, AC4, [i16; 3], compare::compare_constant_i16_ac4);

impl_compare_constant_image!(f32, C1, f32, compare::compare_constant_f32_c1);

impl_compare_constant_image!(f32, C3, [f32; 3], compare::compare_constant_f32_c3);

impl_compare_constant_image!(f32, C4, [f32; 4], compare::compare_constant_f32_c4);

impl_compare_constant_image!(f32, AC4, [f32; 3], compare::compare_constant_f32_ac4);
