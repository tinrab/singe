use singe_cuda::types::{Complex32, Complex64};

use crate::{
    context::StreamContext,
    error::Result,
    pipeline::signal::{
        SignalPipeline,
        operations::operation_traits::{
            ComplexFixedThresholdSignal, ComplexThresholdSignal, ComplexValueThresholdSignal,
            ConstantSignal, DivideIntoConstantSignal, FixedThresholdSignal,
            SubtractFromConstantSignal, ThresholdSignal, UnarySignal, ValueThresholdSignal,
        },
    },
    signal::{
        arithmetic, conversion,
        view::{SignalOutputView, SignalView, SignalViewMut},
    },
    types::{ComparisonOperation, ComplexI16},
};

impl_constant_signal!(
    f32,
    arithmetic::add_constant_f32,
    arithmetic::add_constant_f32_in_place,
    arithmetic::subtract_constant_f32,
    arithmetic::subtract_constant_f32_in_place,
    arithmetic::multiply_constant_f32,
    arithmetic::multiply_constant_f32_in_place,
    arithmetic::divide_constant_f32,
    arithmetic::divide_constant_f32_in_place
);

impl_constant_signal!(
    f64,
    arithmetic::add_constant_f64,
    arithmetic::add_constant_f64_in_place,
    arithmetic::subtract_constant_f64,
    arithmetic::subtract_constant_f64_in_place,
    arithmetic::multiply_constant_f64,
    arithmetic::multiply_constant_f64_in_place,
    arithmetic::divide_constant_f64,
    arithmetic::divide_constant_f64_in_place
);

impl_constant_signal!(
    Complex32,
    arithmetic::add_constant_f32_complex,
    arithmetic::add_constant_f32_complex_in_place,
    arithmetic::subtract_constant_f32_complex,
    arithmetic::subtract_constant_f32_complex_in_place,
    arithmetic::multiply_constant_f32_complex,
    arithmetic::multiply_constant_f32_complex_in_place,
    arithmetic::divide_constant_f32_complex,
    arithmetic::divide_constant_f32_complex_in_place
);

impl_constant_signal!(
    Complex64,
    arithmetic::add_constant_f64_complex,
    arithmetic::add_constant_f64_complex_in_place,
    arithmetic::subtract_constant_f64_complex,
    arithmetic::subtract_constant_f64_complex_in_place,
    arithmetic::multiply_constant_f64_complex,
    arithmetic::multiply_constant_f64_complex_in_place,
    arithmetic::divide_constant_f64_complex,
    arithmetic::divide_constant_f64_complex_in_place
);

impl_subtract_from_constant_signal!(
    f32,
    arithmetic::subtract_from_constant_f32,
    arithmetic::subtract_from_constant_f32_in_place
);

impl_subtract_from_constant_signal!(
    f64,
    arithmetic::subtract_from_constant_f64,
    arithmetic::subtract_from_constant_f64_in_place
);

impl_subtract_from_constant_signal!(
    Complex32,
    arithmetic::subtract_from_constant_f32_complex,
    arithmetic::subtract_from_constant_f32_complex_in_place
);

impl_subtract_from_constant_signal!(
    Complex64,
    arithmetic::subtract_from_constant_f64_complex,
    arithmetic::subtract_from_constant_f64_complex_in_place
);

impl_divide_into_constant_signal!(
    u16,
    arithmetic::divide_into_constant_u16,
    arithmetic::divide_into_constant_u16_in_place
);

impl_divide_into_constant_signal!(
    f32,
    arithmetic::divide_into_constant_f32,
    arithmetic::divide_into_constant_f32_in_place
);

impl SignalPipeline<'_, f32> {
    /// Adds each source element multiplied by `value` to the initialized destination.
    ///
    /// Computes `destination += source * value` elementwise. The provided implementation
    /// uses `f32`. Source and destination lengths must match. Prepares both regions, waits
    /// for producers on other streams, rejects capture, and returns after successful native
    /// execution and stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits where performed, native execution, or stream synchronization fail.
    /// Source and destination lengths must match.
    pub fn add_product_constant_f32_in_place(
        stream_context: &StreamContext,
        source: &SignalView<'_, f32>,
        value: f32,
        destination: &mut SignalViewMut<'_, f32>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            arithmetic::add_product_constant_f32(stream_context, source, value, destination)
        })
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy + arithmetic::AddProductConstant,
{
    /// Adds each source element multiplied by `value` to the initialized destination.
    ///
    /// Computes `destination += source * value` elementwise. The provided implementation
    /// uses `f32`. Source and destination lengths must match. Prepares both regions, waits
    /// for producers on other streams, rejects capture, and returns after successful native
    /// execution and stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits where performed, native execution, or stream synchronization fail.
    /// Source and destination lengths must match.
    pub fn add_product_constant_in_place(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            arithmetic::add_product_constant(stream_context, source, value, destination)
        })
    }
}

impl_unary_signal!(
    f32,
    arithmetic::exponent_f32,
    arithmetic::exponent_f32_in_place,
    arithmetic::natural_logarithm_f32,
    arithmetic::natural_logarithm_f32_in_place,
    arithmetic::square_f32,
    arithmetic::square_f32_in_place,
    arithmetic::square_root_f32,
    arithmetic::square_root_f32_in_place
);

impl_unary_signal!(
    f64,
    arithmetic::exponent_f64,
    arithmetic::exponent_f64_in_place,
    arithmetic::natural_logarithm_f64,
    arithmetic::natural_logarithm_f64_in_place,
    arithmetic::square_f64,
    arithmetic::square_f64_in_place,
    arithmetic::square_root_f64,
    arithmetic::square_root_f64_in_place
);

impl_threshold_signal!(
    i16,
    conversion::threshold_i16,
    conversion::threshold_i16_in_place
);

impl_threshold_signal!(
    f32,
    conversion::threshold_f32,
    conversion::threshold_f32_in_place
);

impl_threshold_signal!(
    f64,
    conversion::threshold_f64,
    conversion::threshold_f64_in_place
);

impl_complex_threshold_signal!(
    ComplexI16,
    i16,
    conversion::threshold_i16_complex,
    conversion::threshold_i16_complex_in_place
);

impl_complex_threshold_signal!(
    Complex32,
    f32,
    conversion::threshold_f32_complex,
    conversion::threshold_f32_complex_in_place
);

impl_complex_threshold_signal!(
    Complex64,
    f64,
    conversion::threshold_f64_complex,
    conversion::threshold_f64_complex_in_place
);

impl_fixed_threshold_signal!(
    i16,
    conversion::threshold_less_i16,
    conversion::threshold_less_i16_in_place,
    conversion::threshold_greater_i16,
    conversion::threshold_greater_i16_in_place
);

impl_fixed_threshold_signal!(
    f32,
    conversion::threshold_less_f32,
    conversion::threshold_less_f32_in_place,
    conversion::threshold_greater_f32,
    conversion::threshold_greater_f32_in_place
);

impl_fixed_threshold_signal!(
    f64,
    conversion::threshold_less_f64,
    conversion::threshold_less_f64_in_place,
    conversion::threshold_greater_f64,
    conversion::threshold_greater_f64_in_place
);

impl_complex_fixed_threshold_signal!(
    ComplexI16,
    i16,
    conversion::threshold_less_i16_complex,
    conversion::threshold_less_i16_complex_in_place,
    conversion::threshold_greater_i16_complex,
    conversion::threshold_greater_i16_complex_in_place
);

impl_complex_fixed_threshold_signal!(
    Complex32,
    f32,
    conversion::threshold_less_f32_complex,
    conversion::threshold_less_f32_complex_in_place,
    conversion::threshold_greater_f32_complex,
    conversion::threshold_greater_f32_complex_in_place
);

impl_complex_fixed_threshold_signal!(
    Complex64,
    f64,
    conversion::threshold_less_f64_complex,
    conversion::threshold_less_f64_complex_in_place,
    conversion::threshold_greater_f64_complex,
    conversion::threshold_greater_f64_complex_in_place
);

impl_value_threshold_signal!(
    i16,
    conversion::threshold_less_value_i16,
    conversion::threshold_less_value_i16_in_place,
    conversion::threshold_greater_value_i16,
    conversion::threshold_greater_value_i16_in_place
);

impl_value_threshold_signal!(
    f32,
    conversion::threshold_less_value_f32,
    conversion::threshold_less_value_f32_in_place,
    conversion::threshold_greater_value_f32,
    conversion::threshold_greater_value_f32_in_place
);

impl_value_threshold_signal!(
    f64,
    conversion::threshold_less_value_f64,
    conversion::threshold_less_value_f64_in_place,
    conversion::threshold_greater_value_f64,
    conversion::threshold_greater_value_f64_in_place
);

impl_complex_value_threshold_signal!(
    ComplexI16,
    i16,
    conversion::threshold_less_value_i16_complex,
    conversion::threshold_less_value_i16_complex_in_place,
    conversion::threshold_greater_value_i16_complex,
    conversion::threshold_greater_value_i16_complex_in_place
);

impl_complex_value_threshold_signal!(
    Complex32,
    f32,
    conversion::threshold_less_value_f32_complex,
    conversion::threshold_less_value_f32_complex_in_place,
    conversion::threshold_greater_value_f32_complex,
    conversion::threshold_greater_value_f32_complex_in_place
);

impl_complex_value_threshold_signal!(
    Complex64,
    f64,
    conversion::threshold_less_value_f64_complex,
    conversion::threshold_less_value_f64_complex_in_place,
    conversion::threshold_greater_value_f64_complex,
    conversion::threshold_greater_value_f64_complex_in_place
);
