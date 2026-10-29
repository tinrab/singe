use singe_cuda::types::{Complex32, Complex64};

use crate::{
    context::StreamContext,
    error::{Error, Result},
    pipeline::{
        SignalAllocator, Workspace,
        signal::{SignalBacking, SignalPipeline},
    },
    signal::{
        arithmetic,
        view::{SignalOutputView, SignalView, SignalViewMut},
    },
    types::{ComplexI16, ComplexI32},
};

mod binary_traits;

use self::binary_traits::{
    AddHeterogeneousInPlaceSignal, AddProductSignal, BinarySignal, DivideByScaledSignal,
    DivideScaledBinarySignal, DivideSignal, HeterogeneousBinarySignal, MixedBinarySignal,
    MultiplyHeterogeneousInPlaceSignal, MultiplyLowScaledSignal, ScaledAddProductSignal,
    ScaledBinarySignal, ScaledHeterogeneousBinarySignal, ScaledHeterogeneousInPlaceSignal,
    ScaledMixedBinarySignal,
};

#[macro_use]
mod binary_macros;

impl_binary_signal!(
    i16,
    arithmetic::add_i16,
    arithmetic::add_i16_in_place,
    arithmetic::subtract_i16,
    arithmetic::subtract_i16_in_place,
    arithmetic::multiply_i16,
    arithmetic::multiply_i16_in_place
);

impl_binary_signal!(
    f32,
    arithmetic::add_f32,
    arithmetic::add_f32_in_place,
    arithmetic::subtract_f32,
    arithmetic::subtract_f32_in_place,
    arithmetic::multiply_f32,
    arithmetic::multiply_f32_in_place
);

impl_binary_signal!(
    f64,
    arithmetic::add_f64,
    arithmetic::add_f64_in_place,
    arithmetic::subtract_f64,
    arithmetic::subtract_f64_in_place,
    arithmetic::multiply_f64,
    arithmetic::multiply_f64_in_place
);

impl_binary_signal!(
    Complex32,
    arithmetic::add_f32_complex,
    arithmetic::add_f32_complex_in_place,
    arithmetic::subtract_f32_complex,
    arithmetic::subtract_f32_complex_in_place,
    arithmetic::multiply_f32_complex,
    arithmetic::multiply_f32_complex_in_place
);

impl_binary_signal!(
    Complex64,
    arithmetic::add_f64_complex,
    arithmetic::add_f64_complex_in_place,
    arithmetic::subtract_f64_complex,
    arithmetic::subtract_f64_complex_in_place,
    arithmetic::multiply_f64_complex,
    arithmetic::multiply_f64_complex_in_place
);

impl_divide_signal!(f32, arithmetic::divide_f32, arithmetic::divide_f32_in_place);

impl_divide_signal!(f64, arithmetic::divide_f64, arithmetic::divide_f64_in_place);

impl_divide_signal!(
    Complex32,
    arithmetic::divide_f32_complex,
    arithmetic::divide_f32_complex_in_place
);

impl_divide_signal!(
    Complex64,
    arithmetic::divide_f64_complex,
    arithmetic::divide_f64_complex_in_place
);

impl_scaled_binary_signal!(
    u8,
    arithmetic::add_u8_scaled,
    arithmetic::add_u8_scaled_in_place,
    arithmetic::subtract_u8_scaled,
    arithmetic::subtract_u8_scaled_in_place,
    arithmetic::multiply_u8_scaled,
    arithmetic::multiply_u8_scaled_in_place
);

impl_divide_scaled_binary_signal!(
    u8,
    arithmetic::divide_u8_scaled,
    arithmetic::divide_u8_scaled_in_place
);

impl_scaled_binary_signal!(
    u16,
    arithmetic::add_u16_scaled,
    arithmetic::add_u16_scaled_in_place,
    arithmetic::subtract_u16_scaled,
    arithmetic::subtract_u16_scaled_in_place,
    arithmetic::multiply_u16_scaled,
    arithmetic::multiply_u16_scaled_in_place
);

impl_divide_scaled_binary_signal!(
    u16,
    arithmetic::divide_u16_scaled,
    arithmetic::divide_u16_scaled_in_place
);

impl_scaled_binary_signal!(
    i16,
    arithmetic::add_i16_scaled,
    arithmetic::add_i16_scaled_in_place,
    arithmetic::subtract_i16_scaled,
    arithmetic::subtract_i16_scaled_in_place,
    arithmetic::multiply_i16_scaled,
    arithmetic::multiply_i16_scaled_in_place
);

impl_divide_scaled_binary_signal!(
    i16,
    arithmetic::divide_i16_scaled,
    arithmetic::divide_i16_scaled_in_place
);

impl_scaled_binary_signal!(
    i32,
    arithmetic::add_i32_scaled,
    arithmetic::add_i32_scaled_in_place,
    arithmetic::subtract_i32_scaled,
    arithmetic::subtract_i32_scaled_in_place,
    arithmetic::multiply_i32_scaled,
    arithmetic::multiply_i32_scaled_in_place
);

impl_divide_scaled_binary_signal!(
    i32,
    arithmetic::divide_i32_scaled,
    arithmetic::divide_i32_scaled_in_place
);

impl_scaled_binary_signal!(
    ComplexI16,
    arithmetic::add_i16_complex_scaled,
    arithmetic::add_i16_complex_scaled_in_place,
    arithmetic::subtract_i16_complex_scaled,
    arithmetic::subtract_i16_complex_scaled_in_place,
    arithmetic::multiply_i16_complex_scaled,
    arithmetic::multiply_i16_complex_scaled_in_place
);

impl_divide_scaled_binary_signal!(
    ComplexI16,
    arithmetic::divide_i16_complex_scaled,
    arithmetic::divide_i16_complex_scaled_in_place
);

impl_scaled_binary_signal!(
    ComplexI32,
    arithmetic::add_i32_complex_scaled,
    arithmetic::add_i32_complex_scaled_in_place,
    arithmetic::subtract_i32_complex_scaled,
    arithmetic::subtract_i32_complex_scaled_in_place,
    arithmetic::multiply_i32_complex_scaled,
    arithmetic::multiply_i32_complex_scaled_in_place
);

impl_mixed_binary_signal_without_subtract!(
    u8,
    u16,
    arithmetic::add_u8_to_u16,
    arithmetic::multiply_u8_to_u16
);

impl_mixed_binary_signal!(
    i16,
    f32,
    arithmetic::add_i16_to_f32,
    arithmetic::subtract_i16_to_f32,
    arithmetic::multiply_i16_to_f32
);

impl_scaled_mixed_binary_signal!(i16, i32, arithmetic::multiply_i16_to_i32_scaled);

impl_scaled_heterogeneous_binary_signal!(u16, i16, i16, arithmetic::multiply_u16_with_i16_scaled);

impl_scaled_heterogeneous_binary_signal!(
    i32,
    ComplexI32,
    ComplexI32,
    arithmetic::multiply_i32_with_i32_complex_scaled
);

unsafe impl MultiplyLowScaledSignal<i32> for SignalPipeline<'_, i32> {
    fn multiply_low_signal_scaled(
        stream_context: &StreamContext,
        left: &SignalView<'_, i32>,
        right: &SignalView<'_, i32>,
        destination: &mut SignalOutputView<'_, i32>,
        scale_factor: i32,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(left)?;
        stream_context.prepare_region(right)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            arithmetic::multiply_low_i32_scaled(
                stream_context,
                left,
                right,
                destination,
                scale_factor,
            )
        })
    }
}

unsafe impl DivideByScaledSignal<i32, i16, i16> for SignalPipeline<'_, i32> {
    fn divide_by_signal_scaled(
        stream_context: &StreamContext,
        left: &SignalView<'_, i32>,
        right: &SignalView<'_, i16>,
        destination: &mut SignalOutputView<'_, i16>,
        scale_factor: i32,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(left)?;
        stream_context.prepare_region(right)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            arithmetic::divide_i32_by_i16_scaled(
                stream_context,
                left,
                right,
                destination,
                scale_factor,
            )
        })
    }
}

unsafe impl ScaledHeterogeneousInPlaceSignal<ComplexI32, i32> for SignalPipeline<'_, ComplexI32> {
    fn multiply_heterogeneous_signal_scaled_in_place(
        stream_context: &StreamContext,
        source: &SignalView<'_, i32>,
        destination: &mut SignalViewMut<'_, ComplexI32>,
        scale_factor: i32,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            arithmetic::multiply_i32_complex_by_i32_scaled_in_place(
                stream_context,
                source,
                destination,
                scale_factor,
            )
        })
    }
}

unsafe impl AddHeterogeneousInPlaceSignal<i32, i16> for SignalPipeline<'_, i32> {
    fn add_heterogeneous_signal_in_place(
        stream_context: &StreamContext,
        source: &SignalView<'_, i16>,
        destination: &mut SignalViewMut<'_, i32>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            arithmetic::add_i16_to_i32_in_place(stream_context, source, destination)
        })
    }
}

unsafe impl MultiplyHeterogeneousInPlaceSignal<Complex32, f32> for SignalPipeline<'_, Complex32> {
    fn multiply_heterogeneous_signal_in_place(
        stream_context: &StreamContext,
        source: &SignalView<'_, f32>,
        destination: &mut SignalViewMut<'_, Complex32>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            arithmetic::multiply_f32_complex_by_f32_in_place(stream_context, source, destination)
        })
    }
}

unsafe impl HeterogeneousBinarySignal<f32, Complex32, Complex32> for SignalPipeline<'_, f32> {
    fn multiply_heterogeneous_signal(
        stream_context: &StreamContext,
        left: &SignalView<'_, f32>,
        right: &SignalView<'_, Complex32>,
        destination: &mut SignalOutputView<'_, Complex32>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(left)?;
        stream_context.prepare_region(right)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            arithmetic::multiply_f32_with_f32_complex(stream_context, left, right, destination)
        })
    }
}

impl_add_product_signal!(f32, arithmetic::add_product_f32);

impl_add_product_signal!(f64, arithmetic::add_product_f64);

impl_add_product_signal!(Complex32, arithmetic::add_product_f32_complex);

impl_add_product_signal!(Complex64, arithmetic::add_product_f64_complex);

impl_scaled_add_product_signal!(i16, i16, arithmetic::add_product_i16_scaled);

impl_scaled_add_product_signal!(i32, i32, arithmetic::add_product_i32_scaled);

impl_scaled_add_product_signal!(i16, i32, arithmetic::add_product_i16_to_i32_scaled);

impl SignalPipeline<'_, u32> {
    /// Adds `left` and `right` element by element into `destination`.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn add_u32_into(
        stream_context: &StreamContext,
        left: &SignalView<'_, u32>,
        right: &SignalView<'_, u32>,
        destination: &mut SignalOutputView<'_, u32>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(left)?;
        stream_context.prepare_region(right)?;
        stream_context.prepare_region(destination)?;
        stream_context
            // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
            .sync_scope(|| unsafe { arithmetic::add_u32(stream_context, left, right, destination) })
    }
}

impl SignalPipeline<'_, i64> {
    /// Adds `left` and `right` element by element into `destination`.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn add_i64_scaled_into(
        stream_context: &StreamContext,
        left: &SignalView<'_, i64>,
        right: &SignalView<'_, i64>,
        destination: &mut SignalOutputView<'_, i64>,
        scale_factor: i32,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(left)?;
        stream_context.prepare_region(right)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            arithmetic::add_i64_scaled(stream_context, left, right, destination, scale_factor)
        })
    }
}

impl SignalPipeline<'_, i32> {
    /// Multiplies `left` and `right` element by element into `destination`.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    /// Uses the native low multiplication routine.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn multiply_low_i32_scaled_into(
        stream_context: &StreamContext,
        left: &SignalView<'_, i32>,
        right: &SignalView<'_, i32>,
        destination: &mut SignalOutputView<'_, i32>,
        scale_factor: i32,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(left)?;
        stream_context.prepare_region(right)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            arithmetic::multiply_low_i32_scaled(
                stream_context,
                left,
                right,
                destination,
                scale_factor,
            )
        })
    }

    /// Computes the element-wise quotient `left / right` into `destination`.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn divide_i32_by_i16_scaled_into(
        stream_context: &StreamContext,
        left: &SignalView<'_, i32>,
        right: &SignalView<'_, i16>,
        destination: &mut SignalOutputView<'_, i16>,
        scale_factor: i32,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(left)?;
        stream_context.prepare_region(right)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            arithmetic::divide_i32_by_i16_scaled(
                stream_context,
                left,
                right,
                destination,
                scale_factor,
            )
        })
    }

    /// Multiplies `destination` by `source` element by element, updating `destination` in place.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn multiply_i32_complex_by_i32_scaled_in_place(
        stream_context: &StreamContext,
        source: &SignalView<'_, i32>,
        destination: &mut SignalViewMut<'_, ComplexI32>,
        scale_factor: i32,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            arithmetic::multiply_i32_complex_by_i32_scaled_in_place(
                stream_context,
                source,
                destination,
                scale_factor,
            )
        })
    }
}

impl SignalPipeline<'_, i16> {
    /// Adds `source` to `destination` element by element, updating `destination` in place.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn add_i16_to_i32_in_place(
        stream_context: &StreamContext,
        source: &SignalView<'_, i16>,
        destination: &mut SignalViewMut<'_, i32>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            arithmetic::add_i16_to_i32_in_place(stream_context, source, destination)
        })
    }
}

impl SignalPipeline<'_, f32> {
    /// Multiplies `left` and `right` element by element into `destination`.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn multiply_f32_with_f32_complex_into(
        stream_context: &StreamContext,
        left: &SignalView<'_, f32>,
        right: &SignalView<'_, Complex32>,
        destination: &mut SignalOutputView<'_, Complex32>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(left)?;
        stream_context.prepare_region(right)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            arithmetic::multiply_f32_with_f32_complex(stream_context, left, right, destination)
        })
    }

    /// Multiplies `destination` by `source` element by element, updating `destination` in place.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn multiply_f32_complex_by_f32_in_place(
        stream_context: &StreamContext,
        source: &SignalView<'_, f32>,
        destination: &mut SignalViewMut<'_, Complex32>,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        stream_context.prepare_region(destination)?;
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            arithmetic::multiply_f32_complex_by_f32_in_place(stream_context, source, destination)
        })
    }
}

mod binary_bitwise;

use self::binary_bitwise::BitwiseSignal;

mod binary_basic_methods;

mod binary_mixed_methods;

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: ScaledBinarySignal<T>,
{
    /// Adds `other` to the pipeline element by element.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    /// Returns an owned pipeline containing the result.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the pipeline source producer can also fail.
    pub fn add_scaled(self, other: &SignalView<'_, T>, scale_factor: i32) -> Result<Self> {
        self.ensure_eager()?;
        self.scaled_binary(
            other,
            scale_factor,
            <Self as ScaledBinarySignal<T>>::add_signal_scaled,
            <Self as ScaledBinarySignal<T>>::add_signal_scaled_in_place,
        )
    }

    /// Subtracts `other` from each pipeline element.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    /// Returns an owned pipeline containing the result.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the pipeline source producer can also fail.
    pub fn subtract_scaled(self, other: &SignalView<'_, T>, scale_factor: i32) -> Result<Self> {
        self.ensure_eager()?;
        self.scaled_binary(
            other,
            scale_factor,
            <Self as ScaledBinarySignal<T>>::subtract_signal_scaled,
            <Self as ScaledBinarySignal<T>>::subtract_signal_scaled_in_place,
        )
    }

    /// Multiplies the pipeline value by `other` element by element.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    /// Returns an owned pipeline containing the result.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the pipeline source producer can also fail.
    pub fn multiply_scaled(self, other: &SignalView<'_, T>, scale_factor: i32) -> Result<Self> {
        self.ensure_eager()?;
        self.scaled_binary(
            other,
            scale_factor,
            <Self as ScaledBinarySignal<T>>::multiply_signal_scaled,
            <Self as ScaledBinarySignal<T>>::multiply_signal_scaled_in_place,
        )
    }

    fn scaled_binary(
        mut self,
        other: &SignalView<'_, T>,
        scale_factor: i32,
        operation: ScaledBinaryCallback<T>,
        operation_in_place: fn(
            &StreamContext,
            &SignalView<'_, T>,
            &mut SignalViewMut<'_, T>,
            i32,
        ) -> Result<()>,
    ) -> Result<Self> {
        self.ensure_eager()?;
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut();
                operation_in_place(self.stream_context, other, &mut signal_view, scale_factor)?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self
                    .workspace
                    .signal_uninit::<T>(self.stream_context.cuda_context(), source.len())?;
                let mut destination_view = destination.output();
                operation(
                    self.stream_context,
                    source,
                    other,
                    &mut destination_view,
                    scale_factor,
                )?;
                // SAFETY: the completed operation above wrote every element.
                self.backing = SignalBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: DivideScaledBinarySignal<T>,
{
    /// Computes the element-wise quotient of the pipeline value divided by `other`.
    /// Uses native fixed-point scaling with `scale_factor`; rounding and saturation follow the selected native operation.
    /// Returns an owned pipeline containing the result.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the pipeline source producer can also fail.
    pub fn divide_scaled(self, other: &SignalView<'_, T>, scale_factor: i32) -> Result<Self> {
        self.ensure_eager()?;
        self.scaled_binary_divide(
            other,
            scale_factor,
            <Self as DivideScaledBinarySignal<T>>::divide_signal_scaled,
            <Self as DivideScaledBinarySignal<T>>::divide_signal_scaled_in_place,
        )
    }

    fn scaled_binary_divide(
        mut self,
        other: &SignalView<'_, T>,
        scale_factor: i32,
        operation: ScaledBinaryCallback<T>,
        operation_in_place: fn(
            &StreamContext,
            &SignalView<'_, T>,
            &mut SignalViewMut<'_, T>,
            i32,
        ) -> Result<()>,
    ) -> Result<Self> {
        self.ensure_eager()?;
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut();
                operation_in_place(self.stream_context, other, &mut signal_view, scale_factor)?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self
                    .workspace
                    .signal_uninit::<T>(self.stream_context.cuda_context(), source.len())?;
                let mut destination_view = destination.output();
                operation(
                    self.stream_context,
                    source,
                    other,
                    &mut destination_view,
                    scale_factor,
                )?;
                // SAFETY: the completed operation above wrote every element.
                self.backing = SignalBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: BitwiseSignal<T>,
{
    /// Computes the element-wise bitwise AND of `left` and `right` into `destination`.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn bit_and_into(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        <Self as BitwiseSignal<T>>::and_signal(stream_context, left, right, destination)
    }

    /// Computes the element-wise bitwise AND of `destination` and `signal`, updating `destination` in place.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn bit_and_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as BitwiseSignal<T>>::and_signal_in_place(stream_context, signal, destination)
    }

    /// Computes the element-wise bitwise OR of `left` and `right` into `destination`.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn bit_or_into(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        <Self as BitwiseSignal<T>>::or_signal(stream_context, left, right, destination)
    }

    /// Computes the element-wise bitwise OR of `destination` and `signal`, updating `destination` in place.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn bit_or_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as BitwiseSignal<T>>::or_signal_in_place(stream_context, signal, destination)
    }

    /// Computes the element-wise bitwise XOR of `left` and `right` into `destination`.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn bit_xor_into(
        stream_context: &StreamContext,
        left: &SignalView<'_, T>,
        right: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        <Self as BitwiseSignal<T>>::xor_signal(stream_context, left, right, destination)
    }

    /// Computes the element-wise bitwise XOR of `destination` and `signal`, updating `destination` in place.
    ///
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn bit_xor_in_place(
        stream_context: &StreamContext,
        signal: &SignalView<'_, T>,
        destination: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as BitwiseSignal<T>>::xor_signal_in_place(stream_context, signal, destination)
    }

    /// Computes the element-wise bitwise AND of the pipeline value and `other`.
    /// Returns an owned pipeline containing the result.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the pipeline source producer can also fail.
    pub fn bit_and(self, other: &SignalView<'_, T>) -> Result<Self> {
        self.ensure_eager()?;
        self.bitwise(
            other,
            <Self as BitwiseSignal<T>>::and_signal,
            <Self as BitwiseSignal<T>>::and_signal_in_place,
        )
    }

    /// Computes the element-wise bitwise OR of the pipeline value and `other`.
    /// Returns an owned pipeline containing the result.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the pipeline source producer can also fail.
    pub fn bit_or(self, other: &SignalView<'_, T>) -> Result<Self> {
        self.ensure_eager()?;
        self.bitwise(
            other,
            <Self as BitwiseSignal<T>>::or_signal,
            <Self as BitwiseSignal<T>>::or_signal_in_place,
        )
    }

    /// Computes the element-wise bitwise XOR of the pipeline value and `other`.
    /// Returns an owned pipeline containing the result.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the pipeline source producer can also fail.
    pub fn bit_xor(self, other: &SignalView<'_, T>) -> Result<Self> {
        self.ensure_eager()?;
        self.bitwise(
            other,
            <Self as BitwiseSignal<T>>::xor_signal,
            <Self as BitwiseSignal<T>>::xor_signal_in_place,
        )
    }

    fn bitwise(
        mut self,
        other: &SignalView<'_, T>,
        operation: BitwiseCallback<T>,
        operation_in_place: fn(
            &StreamContext,
            &SignalView<'_, T>,
            &mut SignalViewMut<'_, T>,
        ) -> Result<()>,
    ) -> Result<Self> {
        self.ensure_eager()?;
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut();
                operation_in_place(self.stream_context, other, &mut signal_view)?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self
                    .workspace
                    .signal_uninit::<T>(self.stream_context.cuda_context(), source.len())?;
                let mut destination_view = destination.output();
                operation(self.stream_context, source, other, &mut destination_view)?;
                // SAFETY: the completed operation above wrote every element.
                self.backing = SignalBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}

impl SignalPipeline<'_, f32> {
    /// Computes the element-wise quotient of the pipeline value divided by `other`.
    /// Returns an owned pipeline containing the result.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the pipeline source producer can also fail.
    pub fn divide(self, other: &SignalView<'_, f32>) -> Result<Self> {
        self.ensure_eager()?;
        self.binary(
            other,
            <Self as DivideSignal<f32>>::divide_signal,
            <Self as DivideSignal<f32>>::divide_signal_in_place,
        )
    }
}

impl SignalPipeline<'_, f64> {
    /// Computes the element-wise quotient of the pipeline value divided by `other`.
    /// Returns an owned pipeline containing the result.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the pipeline source producer can also fail.
    pub fn divide(self, other: &SignalView<'_, f64>) -> Result<Self> {
        self.ensure_eager()?;
        self.binary(
            other,
            <Self as DivideSignal<f64>>::divide_signal,
            <Self as DivideSignal<f64>>::divide_signal_in_place,
        )
    }
}

impl SignalPipeline<'_, Complex32> {
    /// Computes the element-wise quotient of the pipeline value divided by `other`.
    /// Returns an owned pipeline containing the result.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the pipeline source producer can also fail.
    pub fn divide(self, other: &SignalView<'_, Complex32>) -> Result<Self> {
        self.ensure_eager()?;
        self.binary(
            other,
            <Self as DivideSignal<Complex32>>::divide_signal,
            <Self as DivideSignal<Complex32>>::divide_signal_in_place,
        )
    }
}

impl SignalPipeline<'_, Complex64> {
    /// Computes the element-wise quotient of the pipeline value divided by `other`.
    /// Returns an owned pipeline containing the result.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Supported operations prepare participating regions, waiting for producers on other streams, and complete native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Participating signal lengths must match.
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length cannot be represented by the native interface, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the pipeline source producer can also fail.
    pub fn divide(self, other: &SignalView<'_, Complex64>) -> Result<Self> {
        self.ensure_eager()?;
        self.binary(
            other,
            <Self as DivideSignal<Complex64>>::divide_signal,
            <Self as DivideSignal<Complex64>>::divide_signal_in_place,
        )
    }
}

type ScaledBinaryCallback<T> = fn(
    &StreamContext,
    &SignalView<'_, T>,
    &SignalView<'_, T>,
    &mut SignalOutputView<'_, T>,
    i32,
) -> Result<()>;

type BitwiseCallback<T> = fn(
    &StreamContext,
    &SignalView<'_, T>,
    &SignalView<'_, T>,
    &mut SignalOutputView<'_, T>,
) -> Result<()>;
