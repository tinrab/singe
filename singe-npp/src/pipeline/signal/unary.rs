use singe_cuda::types::{Complex32, Complex64};

use crate::{
    context::StreamContext,
    error::Result,
    pipeline::{
        SignalAllocator, Workspace,
        signal::{SignalBacking, SignalPipeline},
    },
    signal::{
        arithmetic, initialization,
        view::{SignalOutputView, SignalView, SignalViewMut},
    },
    types::{ComplexI16, DataTypeLike},
};

/// Absolute-value operation marker for signal pipeline dispatch.
#[derive(Debug)]
pub struct Absolute;

/// Arctangent operation marker for signal pipeline dispatch.
#[derive(Debug)]
pub struct Arctangent;

/// Bitwise-complement operation marker for signal pipeline dispatch.
#[derive(Debug)]
pub struct BitNot;

/// Cauchy error-function operation marker for signal pipeline dispatch.
#[derive(Debug)]
pub struct Cauchy;

/// Cauchy first-derivative operation marker for signal pipeline dispatch.
#[derive(Debug)]
pub struct CauchyDerivative;

/// Cube-root operation marker for signal pipeline dispatch.
#[derive(Debug)]
pub struct CubeRoot;

/// Scaled cube-root conversion to `i16` operation marker for signal pipeline dispatch.
#[derive(Debug)]
pub struct CubeRootToI16Scaled;

/// Scaled exponential operation marker for signal pipeline dispatch.
#[derive(Debug)]
pub struct ExponentScaled;

/// Exponential conversion to `f64` operation marker for signal pipeline dispatch.
#[derive(Debug)]
pub struct ExponentToF64;

/// Natural-logarithm conversion to `f32` operation marker for signal pipeline dispatch.
#[derive(Debug)]
pub struct NaturalLogarithmToF32;

/// Scaled natural-logarithm conversion to `i16` operation marker for signal pipeline dispatch.
#[derive(Debug)]
pub struct NaturalLogarithmToI16Scaled;

/// Scaled natural-logarithm operation marker for signal pipeline dispatch.
#[derive(Debug)]
pub struct NaturalLogarithmScaled;

/// Scaled square operation marker for signal pipeline dispatch.
#[derive(Debug)]
pub struct SquareScaled;

/// Square operation marker for signal pipeline dispatch.
#[derive(Debug)]
pub struct Square;

/// Scaled square-root conversion to `i16` operation marker for signal pipeline dispatch.
#[derive(Debug)]
pub struct SquareRootToI16Scaled;

/// Scaled square-root operation marker for signal pipeline dispatch.
#[derive(Debug)]
pub struct SquareRootScaled;

/// Square-root operation marker for signal pipeline dispatch.
#[derive(Debug)]
pub struct SquareRoot;

/// Scaled ten-times-base-ten-logarithm operation marker for signal pipeline dispatch.
#[derive(Debug)]
pub struct TenTimesLog10Scaled;

/// Dispatches a unary operation selected by `O` without changing the element type.
///
/// # Safety
/// Implementations must preserve native element representations and the operation selected by `O`.
/// Complete borrowed native work on every return path, including errors, and initialize all
/// destination elements on success. Uphold the completed dispatch contract of
/// [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait UnaryTransformSignal<T, O> {
    /// Applies the unary operation selected by `O` to each signal element. Writes the result to `destination`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Source and destination lengths must match.
    fn unary_transform_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()>;

    /// Applies the unary operation selected by `O` to each signal element. Updates the signal in place.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    fn unary_transform_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
    ) -> Result<()>;
}

/// Dispatches a scaled unary operation selected by `O` without changing the element type.
///
/// # Safety
/// Implementations must preserve native element representations and the operation selected by `O`.
/// Complete borrowed native work on every return path, including errors, and initialize all
/// destination elements on success. Uphold the completed dispatch contract of
/// [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait ScaledUnaryTransformSignal<T, O> {
    /// Applies the unary operation selected by `O` to each signal element. Writes the result to `destination`.
    /// Applies the selected native operation's scaling with `scale_factor`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Source and destination lengths must match.
    fn scaled_unary_transform_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scale_factor: i32,
    ) -> Result<()>;

    /// Applies the unary operation selected by `O` to each signal element. Updates the signal in place.
    /// Applies the selected native operation's scaling with `scale_factor`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    fn scaled_unary_transform_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()>;
}

/// Dispatches a unary operation selected by `O` into a destination element type.
///
/// # Safety
/// Implementations must preserve native element representations and the operation selected by `O`.
/// Complete borrowed native work on every return path, including errors, and initialize all
/// destination elements on success. Uphold the completed dispatch contract of
/// [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait UnaryOutputSignal<T, U, O> {
    /// Applies the unary operation selected by `O` to each signal element. Writes the result to `destination`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Source and destination lengths must match.
    fn unary_output_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, U>,
    ) -> Result<()>;
}

/// Dispatches a scaled unary operation selected by `O` into a destination element type.
///
/// # Safety
/// Implementations must preserve native element representations and the operation selected by `O`.
/// Complete borrowed native work on every return path, including errors, and initialize all
/// destination elements on success. Uphold the completed dispatch contract of
/// [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait ScaledUnaryOutputSignal<T, U, O> {
    /// Applies the unary operation selected by `O` to each signal element. Writes the result to `destination`.
    /// Applies the selected native operation's scaling with `scale_factor`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Source and destination lengths must match.
    fn scaled_unary_output_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, U>,
        scale_factor: i32,
    ) -> Result<()>;
}

/// Dispatches an in-place unary operation selected by `O` with a scalar parameter.
///
/// # Safety
/// Implementations must preserve native element representations and the operation selected by `O`.
/// Complete borrowed native work on every return path, including errors, and initialize all
/// destination elements on success. Uphold the completed dispatch contract of
/// [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait ParameterizedUnaryTransformSignal<T, P, O> {
    /// Applies the unary operation selected by `O` to each signal element. Updates the signal in place.
    /// Passes `parameter` as the constant in the native Cauchy formula for the provided Cauchy operations.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    fn parameterized_unary_transform_signal_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        parameter: P,
    ) -> Result<()>;
}

macro_rules! impl_unary_transform_signal {
    ($operation_ty:ty, $ty:ty, $operation:path, $operation_in_place:path) => {
        unsafe impl<'a> UnaryTransformSignal<$ty, $operation_ty> for SignalPipeline<'a, $ty> {
            fn unary_transform_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalOutputView<'_, $ty>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $operation(stream_context, source, destination) })
            }

            fn unary_transform_signal_in_place(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, $ty>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(signal)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe { $operation_in_place(stream_context, signal) })
            }
        }
    };
}

macro_rules! impl_scaled_unary_transform_signal {
    ($operation_ty:ty, $ty:ty, $operation:path, $operation_in_place:path) => {
        unsafe impl<'a> ScaledUnaryTransformSignal<$ty, $operation_ty> for SignalPipeline<'a, $ty> {
            fn scaled_unary_transform_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalOutputView<'_, $ty>,
                scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $operation(stream_context, source, destination, scale_factor)
                })
            }

            fn scaled_unary_transform_signal_in_place(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, $ty>,
                scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(signal)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $operation_in_place(stream_context, signal, scale_factor)
                })
            }
        }
    };
}

macro_rules! impl_unary_output_signal {
    ($operation_ty:ty, $source:ty, $destination:ty, $operation:path) => {
        unsafe impl<'a> UnaryOutputSignal<$source, $destination, $operation_ty>
            for SignalPipeline<'a, $source>
        {
            fn unary_output_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $source>,
                destination: &mut SignalOutputView<'_, $destination>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $operation(stream_context, source, destination) })
            }
        }
    };
}

macro_rules! impl_scaled_unary_output_signal {
    ($operation_ty:ty, $source:ty, $destination:ty, $operation:path) => {
        unsafe impl<'a> ScaledUnaryOutputSignal<$source, $destination, $operation_ty>
            for SignalPipeline<'a, $source>
        {
            fn scaled_unary_output_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $source>,
                destination: &mut SignalOutputView<'_, $destination>,
                scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $operation(stream_context, source, destination, scale_factor)
                })
            }
        }
    };
}

macro_rules! impl_parameterized_unary_transform_signal {
    ($operation_ty:ty, $ty:ty, $parameter_ty:ty, $operation_in_place:path) => {
        unsafe impl<'a> ParameterizedUnaryTransformSignal<$ty, $parameter_ty, $operation_ty>
            for SignalPipeline<'a, $ty>
        {
            fn parameterized_unary_transform_signal_in_place(
                stream_context: &StreamContext,
                signal: &mut SignalViewMut<'_, $ty>,
                parameter: $parameter_ty,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(signal)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $operation_in_place(stream_context, signal, parameter)
                })
            }
        }
    };
}

impl_unary_transform_signal!(
    Absolute,
    i16,
    arithmetic::absolute_i16,
    arithmetic::absolute_i16_in_place
);

impl_unary_transform_signal!(
    Absolute,
    i32,
    arithmetic::absolute_i32,
    arithmetic::absolute_i32_in_place
);

impl_unary_transform_signal!(
    Absolute,
    f32,
    arithmetic::absolute_f32,
    arithmetic::absolute_f32_in_place
);

impl_unary_transform_signal!(
    Absolute,
    f64,
    arithmetic::absolute_f64,
    arithmetic::absolute_f64_in_place
);

impl_unary_transform_signal!(
    Arctangent,
    f32,
    arithmetic::arctangent_f32,
    arithmetic::arctangent_f32_in_place
);

impl_unary_transform_signal!(
    Arctangent,
    f64,
    arithmetic::arctangent_f64,
    arithmetic::arctangent_f64_in_place
);

impl_unary_output_signal!(CubeRoot, f32, f32, arithmetic::cube_root_f32);

impl_unary_output_signal!(ExponentToF64, f32, f64, arithmetic::exponent_f32_to_f64);

impl_unary_output_signal!(
    NaturalLogarithmToF32,
    f64,
    f32,
    arithmetic::natural_logarithm_f64_to_f32
);

impl_parameterized_unary_transform_signal!(Cauchy, f32, f32, arithmetic::cauchy_f32_in_place);

impl_parameterized_unary_transform_signal!(
    CauchyDerivative,
    f32,
    f32,
    arithmetic::cauchy_derivative_f32_in_place
);

impl_unary_transform_signal!(BitNot, u8, arithmetic::not_u8, arithmetic::not_u8_in_place);

impl_unary_transform_signal!(
    BitNot,
    u16,
    arithmetic::not_u16,
    arithmetic::not_u16_in_place
);

impl_unary_transform_signal!(
    BitNot,
    u32,
    arithmetic::not_u32,
    arithmetic::not_u32_in_place
);

impl_scaled_unary_transform_signal!(
    ExponentScaled,
    i16,
    arithmetic::exponent_i16_scaled,
    arithmetic::exponent_i16_scaled_in_place
);

impl_scaled_unary_transform_signal!(
    ExponentScaled,
    i32,
    arithmetic::exponent_i32_scaled,
    arithmetic::exponent_i32_scaled_in_place
);

impl_scaled_unary_transform_signal!(
    ExponentScaled,
    i64,
    arithmetic::exponent_i64_scaled,
    arithmetic::exponent_i64_scaled_in_place
);

impl_scaled_unary_transform_signal!(
    NaturalLogarithmScaled,
    i16,
    arithmetic::natural_logarithm_i16_scaled,
    arithmetic::natural_logarithm_i16_scaled_in_place
);

impl_scaled_unary_transform_signal!(
    NaturalLogarithmScaled,
    i32,
    arithmetic::natural_logarithm_i32_scaled,
    arithmetic::natural_logarithm_i32_scaled_in_place
);

impl_scaled_unary_transform_signal!(
    TenTimesLog10Scaled,
    i32,
    arithmetic::ten_times_log10_i32_scaled,
    arithmetic::ten_times_log10_i32_scaled_in_place
);

impl_scaled_unary_output_signal!(
    NaturalLogarithmToI16Scaled,
    i32,
    i16,
    arithmetic::natural_logarithm_i32_to_i16_scaled
);

impl_scaled_unary_output_signal!(
    CubeRootToI16Scaled,
    i32,
    i16,
    arithmetic::cube_root_i32_to_i16_scaled
);

impl_scaled_unary_output_signal!(
    SquareRootToI16Scaled,
    i32,
    i16,
    arithmetic::square_root_i32_to_i16_scaled
);

impl_scaled_unary_output_signal!(
    SquareRootToI16Scaled,
    i64,
    i16,
    arithmetic::square_root_i64_to_i16_scaled
);

impl_scaled_unary_transform_signal!(
    SquareScaled,
    u8,
    arithmetic::square_u8_scaled,
    arithmetic::square_u8_scaled_in_place
);

impl_scaled_unary_transform_signal!(
    SquareScaled,
    u16,
    arithmetic::square_u16_scaled,
    arithmetic::square_u16_scaled_in_place
);

impl_scaled_unary_transform_signal!(
    SquareScaled,
    i16,
    arithmetic::square_i16_scaled,
    arithmetic::square_i16_scaled_in_place
);

impl_scaled_unary_transform_signal!(
    SquareScaled,
    ComplexI16,
    arithmetic::square_i16_complex_scaled,
    arithmetic::square_i16_complex_scaled_in_place
);

impl_unary_transform_signal!(
    Square,
    Complex32,
    arithmetic::square_f32_complex,
    arithmetic::square_f32_complex_in_place
);

impl_unary_transform_signal!(
    Square,
    Complex64,
    arithmetic::square_f64_complex,
    arithmetic::square_f64_complex_in_place
);

impl_scaled_unary_transform_signal!(
    SquareRootScaled,
    u8,
    arithmetic::square_root_u8_scaled,
    arithmetic::square_root_u8_scaled_in_place
);

impl_scaled_unary_transform_signal!(
    SquareRootScaled,
    u16,
    arithmetic::square_root_u16_scaled,
    arithmetic::square_root_u16_scaled_in_place
);

impl_scaled_unary_transform_signal!(
    SquareRootScaled,
    i16,
    arithmetic::square_root_i16_scaled,
    arithmetic::square_root_i16_scaled_in_place
);

impl_scaled_unary_transform_signal!(
    SquareRootScaled,
    ComplexI16,
    arithmetic::square_root_i16_complex_scaled,
    arithmetic::square_root_i16_complex_scaled_in_place
);

impl_unary_transform_signal!(
    SquareRoot,
    Complex32,
    arithmetic::square_root_f32_complex,
    arithmetic::square_root_f32_complex_in_place
);

impl_unary_transform_signal!(
    SquareRoot,
    Complex64,
    arithmetic::square_root_f64_complex,
    arithmetic::square_root_f64_complex_in_place
);

impl_scaled_unary_transform_signal!(
    SquareRootScaled,
    i64,
    arithmetic::square_root_i64_scaled,
    arithmetic::square_root_i64_scaled_in_place
);

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: UnaryTransformSignal<T, Absolute>,
{
    /// Computes the absolute value of each element. Returns an owned pipeline containing the result.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    /// Waiting for a region's producer can also fail.
    pub fn absolute(self) -> Result<Self> {
        self.ensure_eager()?;
        self.unary_transform::<Absolute>()
    }
}

impl<'a> SignalPipeline<'a, f32> {
    /// Computes the cube root of each element. Returns an owned pipeline containing the result.
    ///
    /// Allocates output through the workspace.
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    /// Waiting for a region's producer can also fail.
    pub fn cube_root(self) -> Result<Self> {
        self.ensure_eager()?;
        self.unary_output::<f32, CubeRoot>()
    }

    /// Computes the native Cauchy error function for each element. Returns an owned pipeline containing the result.
    /// Passes `parameter` as the constant in the native Cauchy formula for the provided Cauchy operations.
    ///
    /// Reuses owned signal storage, or copies a borrowed signal into workspace storage before the in-place operation.
    /// A borrowed source must already be ready on the selected stream; the copy does not wait for a producer on another stream.
    /// Prepares the in-place region and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    /// Waiting for a region's producer can also fail.
    pub fn cauchy(self, parameter: f32) -> Result<Self> {
        self.ensure_eager()?;
        self.parameterized_unary_transform::<Cauchy>(parameter)
    }

    /// Computes the first derivative of the native Cauchy error function for each element. Returns an owned pipeline containing the result.
    /// Passes `parameter` as the constant in the native Cauchy formula for the provided Cauchy operations.
    ///
    /// Reuses owned signal storage, or copies a borrowed signal into workspace storage before the in-place operation.
    /// A borrowed source must already be ready on the selected stream; the copy does not wait for a producer on another stream.
    /// Prepares the in-place region and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    /// Waiting for a region's producer can also fail.
    pub fn cauchy_derivative(self, parameter: f32) -> Result<Self> {
        self.ensure_eager()?;
        self.parameterized_unary_transform::<CauchyDerivative>(parameter)
    }

    /// Computes the natural exponential of each element. Returns an owned pipeline containing the result.
    /// Uses `f64` output elements.
    ///
    /// Allocates output through the workspace.
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    /// Waiting for a region's producer can also fail.
    pub fn exponent_to_f64(self) -> Result<SignalPipeline<'a, f64>> {
        self.ensure_eager()?;
        self.unary_output::<f64, ExponentToF64>()
    }
}

impl<'a> SignalPipeline<'a, f64> {
    /// Computes the natural logarithm of each element. Returns an owned pipeline containing the result.
    /// Uses `f32` output elements.
    ///
    /// Allocates output through the workspace.
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    /// Waiting for a region's producer can also fail.
    pub fn natural_logarithm_to_f32(self) -> Result<SignalPipeline<'a, f32>> {
        self.ensure_eager()?;
        self.unary_output::<f32, NaturalLogarithmToF32>()
    }
}

impl<'a> SignalPipeline<'a, i32> {
    /// Computes the natural logarithm of each element. Returns an owned pipeline containing the result.
    /// Applies the selected native operation's scaling with `scale_factor`.
    /// Uses `i16` output elements.
    ///
    /// Allocates output through the workspace.
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    /// Waiting for a region's producer can also fail.
    pub fn natural_logarithm_to_i16_scaled(
        self,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i16>> {
        self.ensure_eager()?;
        self.scaled_unary_output::<i16, NaturalLogarithmToI16Scaled>(scale_factor)
    }

    /// Computes the cube root of each element. Returns an owned pipeline containing the result.
    /// Applies the selected native operation's scaling with `scale_factor`.
    /// Uses `i16` output elements.
    ///
    /// Allocates output through the workspace.
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    /// Waiting for a region's producer can also fail.
    pub fn cube_root_to_i16_scaled(self, scale_factor: i32) -> Result<SignalPipeline<'a, i16>> {
        self.ensure_eager()?;
        self.scaled_unary_output::<i16, CubeRootToI16Scaled>(scale_factor)
    }

    /// Computes the square root of each element. Returns an owned pipeline containing the result.
    /// Applies the selected native operation's scaling with `scale_factor`.
    /// Uses `i16` output elements.
    ///
    /// Allocates output through the workspace.
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    /// Waiting for a region's producer can also fail.
    pub fn square_root_to_i16_scaled(self, scale_factor: i32) -> Result<SignalPipeline<'a, i16>> {
        self.ensure_eager()?;
        self.scaled_unary_output::<i16, SquareRootToI16Scaled>(scale_factor)
    }
}

impl<'a> SignalPipeline<'a, i64> {
    /// Computes the square root of each element. Returns an owned pipeline containing the result.
    /// Applies the selected native operation's scaling with `scale_factor`.
    /// Uses `i16` output elements.
    ///
    /// Allocates output through the workspace.
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    /// Waiting for a region's producer can also fail.
    pub fn square_root_to_i16_scaled(self, scale_factor: i32) -> Result<SignalPipeline<'a, i16>> {
        self.ensure_eager()?;
        self.scaled_unary_output::<i16, SquareRootToI16Scaled>(scale_factor)
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: ScaledUnaryTransformSignal<T, ExponentScaled>,
{
    /// Computes the natural exponential of each element. Returns an owned pipeline containing the result.
    /// Applies the selected native operation's scaling with `scale_factor`.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    /// Waiting for a region's producer can also fail.
    pub fn exponent_scaled(self, scale_factor: i32) -> Result<Self> {
        self.ensure_eager()?;
        self.scaled_unary_transform::<ExponentScaled>(scale_factor)
    }
}

impl SignalPipeline<'_, f32> {
    /// Applies the unary operation selected by `O` to each signal element. Updates the signal in place.
    /// Passes `parameter` as the constant in the native Cauchy formula for the provided Cauchy operations.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    pub fn parameterized_unary_transform_in_place<O>(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, f32>,
        parameter: f32,
    ) -> Result<()>
    where
        Self: ParameterizedUnaryTransformSignal<f32, f32, O>,
    {
        <Self as ParameterizedUnaryTransformSignal<
            f32,
            f32,
            O,
        >>::parameterized_unary_transform_signal_in_place(stream_context, signal, parameter)
    }

    fn parameterized_unary_transform<O>(mut self, parameter: f32) -> Result<Self>
    where
        Self: ParameterizedUnaryTransformSignal<f32, f32, O>,
    {
        self.ensure_eager()?;
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut();
                <Self as ParameterizedUnaryTransformSignal<
                    f32,
                    f32,
                    O,
                >>::parameterized_unary_transform_signal_in_place(
                    self.stream_context,
                    &mut signal_view,
                    parameter,
                )?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self
                    .workspace
                    .signal_uninit::<f32>(self.stream_context.cuda_context(), source.len())?;
                let mut destination_view = destination.output();
                self.stream_context.sync_scope(|| {
                    // SAFETY: this helper retains its source, destination, and native resources through completion.
                    unsafe {
                        initialization::copy_f32(self.stream_context, source, &mut destination_view)
                    }
                })?;
                // SAFETY: the completed copy established every destination value.
                let mut destination_view = unsafe { destination_view.assume_init() };
                <Self as ParameterizedUnaryTransformSignal<
                    f32,
                    f32,
                    O,
                >>::parameterized_unary_transform_signal_in_place(
                    self.stream_context,
                    &mut destination_view,
                    parameter,
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
    Self: ScaledUnaryTransformSignal<T, NaturalLogarithmScaled>,
{
    /// Computes the natural logarithm of each element. Returns an owned pipeline containing the result.
    /// Applies the selected native operation's scaling with `scale_factor`.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    /// Waiting for a region's producer can also fail.
    pub fn natural_logarithm_scaled(self, scale_factor: i32) -> Result<Self> {
        self.ensure_eager()?;
        self.scaled_unary_transform::<NaturalLogarithmScaled>(scale_factor)
    }
}

impl SignalPipeline<'_, i32> {
    /// Computes ten times the base-ten logarithm of each element. Returns an owned pipeline containing the result.
    /// Applies the selected native operation's scaling with `scale_factor`.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    /// Waiting for a region's producer can also fail.
    pub fn ten_times_log10_scaled(self, scale_factor: i32) -> Result<Self> {
        self.ensure_eager()?;
        self.scaled_unary_transform::<TenTimesLog10Scaled>(scale_factor)
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: ScaledUnaryTransformSignal<T, SquareScaled>,
{
    /// Squares each element. Returns an owned pipeline containing the result.
    /// Applies the selected native operation's scaling with `scale_factor`.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    /// Waiting for a region's producer can also fail.
    pub fn square_scaled(self, scale_factor: i32) -> Result<Self> {
        self.ensure_eager()?;
        self.scaled_unary_transform::<SquareScaled>(scale_factor)
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: ScaledUnaryTransformSignal<T, SquareRootScaled>,
{
    /// Computes the square root of each element. Returns an owned pipeline containing the result.
    /// Applies the selected native operation's scaling with `scale_factor`.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    /// Waiting for a region's producer can also fail.
    pub fn square_root_scaled(self, scale_factor: i32) -> Result<Self> {
        self.ensure_eager()?;
        self.scaled_unary_transform::<SquareRootScaled>(scale_factor)
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: UnaryTransformSignal<T, Arctangent>,
{
    /// Computes the arctangent of each element. Returns an owned pipeline containing the result.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    /// Waiting for a region's producer can also fail.
    pub fn arctangent(self) -> Result<Self> {
        self.ensure_eager()?;
        self.unary_transform::<Arctangent>()
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: UnaryTransformSignal<T, BitNot>,
{
    /// Complements every bit of each element. Returns an owned pipeline containing the result.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    /// Waiting for a region's producer can also fail.
    pub fn bit_not(self) -> Result<Self> {
        self.ensure_eager()?;
        self.unary_transform::<BitNot>()
    }
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
{
    /// Applies the unary operation selected by `O` to each signal element. Writes the result to `destination`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Source and destination lengths must match.
    pub fn unary_transform_into<O>(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()>
    where
        Self: UnaryTransformSignal<T, O>,
    {
        <Self as UnaryTransformSignal<T, O>>::unary_transform_signal(
            stream_context,
            source,
            destination,
        )
    }

    /// Applies the unary operation selected by `O` to each signal element. Updates the signal in place.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    pub fn unary_transform_in_place<O>(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
    ) -> Result<()>
    where
        Self: UnaryTransformSignal<T, O>,
    {
        <Self as UnaryTransformSignal<T, O>>::unary_transform_signal_in_place(
            stream_context,
            signal,
        )
    }

    /// Applies the unary operation selected by `O` to each signal element. Writes the result to `destination`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Source and destination lengths must match.
    pub fn unary_output_into<U, O>(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, U>,
    ) -> Result<()>
    where
        U: Copy,
        Self: UnaryOutputSignal<T, U, O>,
    {
        <Self as UnaryOutputSignal<T, U, O>>::unary_output_signal(
            stream_context,
            source,
            destination,
        )
    }

    /// Applies the unary operation selected by `O` to each signal element. Writes the result to `destination`.
    /// Applies the selected native operation's scaling with `scale_factor`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Source and destination lengths must match.
    pub fn scaled_unary_output_into<U, O>(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, U>,
        scale_factor: i32,
    ) -> Result<()>
    where
        U: Copy,
        Self: ScaledUnaryOutputSignal<T, U, O>,
    {
        <Self as ScaledUnaryOutputSignal<T, U, O>>::scaled_unary_output_signal(
            stream_context,
            source,
            destination,
            scale_factor,
        )
    }

    /// Applies the unary operation selected by `O` to each signal element. Writes the result to `destination`.
    /// Applies the selected native operation's scaling with `scale_factor`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    /// Source and destination lengths must match.
    pub fn scaled_unary_transform_into<O>(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        scale_factor: i32,
    ) -> Result<()>
    where
        Self: ScaledUnaryTransformSignal<T, O>,
    {
        <Self as ScaledUnaryTransformSignal<T, O>>::scaled_unary_transform_signal(
            stream_context,
            source,
            destination,
            scale_factor,
        )
    }

    /// Applies the unary operation selected by `O` to each signal element. Updates the signal in place.
    /// Applies the selected native operation's scaling with `scale_factor`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Waiting for a region's producer can also fail.
    pub fn scaled_unary_transform_in_place<O>(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        scale_factor: i32,
    ) -> Result<()>
    where
        Self: ScaledUnaryTransformSignal<T, O>,
    {
        <Self as ScaledUnaryTransformSignal<T, O>>::scaled_unary_transform_signal_in_place(
            stream_context,
            signal,
            scale_factor,
        )
    }

    /// Computes the natural exponential of each element. Returns an owned pipeline containing the result.
    ///
    /// Allocates output through the workspace and waits for successful stream completion.
    /// The source must already be ready on the selected stream; this method does not wait for a producer on another stream.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn exponent_to<U>(self) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        Workspace: SignalAllocator<U>,
        T: arithmetic::ExponentTo<U>,
    {
        self.ensure_eager()?;
        let mut destination = self
            .workspace
            .signal_uninit::<U>(self.stream_context.cuda_context(), self.len())?;
        {
            let source = self.view();
            let mut destination_view = destination.output();
            self.stream_context.sync_scope(|| {
                // SAFETY: this helper retains its source, destination, and native resources through completion.
                unsafe {
                    arithmetic::exponent_to(self.stream_context, &source, &mut destination_view)
                }
            })?;
        }

        Ok(SignalPipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: SignalBacking::Owned(unsafe { destination.assume_init() }),
        })
    }

    /// Computes the natural logarithm of each element. Returns an owned pipeline containing the result.
    ///
    /// Allocates output through the workspace and waits for successful stream completion.
    /// The source must already be ready on the selected stream; this method does not wait for a producer on another stream.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn natural_logarithm_to<U>(self) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        Workspace: SignalAllocator<U>,
        T: arithmetic::NaturalLogarithmTo<U>,
    {
        self.ensure_eager()?;
        let mut destination = self
            .workspace
            .signal_uninit::<U>(self.stream_context.cuda_context(), self.len())?;
        {
            let source = self.view();
            let mut destination_view = destination.output();
            self.stream_context.sync_scope(|| {
                // SAFETY: this helper retains its source, destination, and native resources through completion.
                unsafe {
                    arithmetic::natural_logarithm_to(
                        self.stream_context,
                        &source,
                        &mut destination_view,
                    )
                }
            })?;
        }

        Ok(SignalPipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: SignalBacking::Owned(unsafe { destination.assume_init() }),
        })
    }

    /// Computes the natural logarithm of each element. Returns an owned pipeline containing the result.
    /// Applies the selected native operation's scaling with `scale_factor`.
    ///
    /// Allocates output through the workspace and waits for successful stream completion.
    /// The source must already be ready on the selected stream; this method does not wait for a producer on another stream.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn natural_logarithm_scaled_to<U>(self, scale_factor: i32) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        Workspace: SignalAllocator<U>,
        T: arithmetic::NaturalLogarithmScaledTo<U>,
    {
        self.ensure_eager()?;
        let mut destination = self
            .workspace
            .signal_uninit::<U>(self.stream_context.cuda_context(), self.len())?;
        {
            let source = self.view();
            let mut destination_view = destination.output();
            self.stream_context.sync_scope(|| {
                // SAFETY: this helper retains its source, destination, and native resources through completion.
                unsafe {
                    arithmetic::natural_logarithm_scaled_to(
                        self.stream_context,
                        &source,
                        &mut destination_view,
                        scale_factor,
                    )
                }
            })?;
        }

        Ok(SignalPipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: SignalBacking::Owned(unsafe { destination.assume_init() }),
        })
    }

    /// Computes the natural logarithm of each element. Returns an owned pipeline containing the result.
    /// Applies the selected native operation's scaling with `scale_factor`.
    ///
    /// Allocates output through the workspace and waits for successful stream completion.
    /// The source must already be ready on the selected stream; this method does not wait for a producer on another stream.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn natural_logarithm_to_scaled<U>(self, scale_factor: i32) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        Workspace: SignalAllocator<U>,
        T: arithmetic::NaturalLogarithmScaledTo<U>,
    {
        self.ensure_eager()?;
        self.natural_logarithm_scaled_to::<U>(scale_factor)
    }

    /// Computes the cube root of each element. Returns an owned pipeline containing the result.
    /// Applies the selected native operation's scaling with `scale_factor`.
    ///
    /// Allocates output through the workspace and waits for successful stream completion.
    /// The source must already be ready on the selected stream; this method does not wait for a producer on another stream.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn cube_root_scaled_to<U>(self, scale_factor: i32) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        Workspace: SignalAllocator<U>,
        T: arithmetic::CubeRootScaledTo<U>,
    {
        self.ensure_eager()?;
        let mut destination = self
            .workspace
            .signal_uninit::<U>(self.stream_context.cuda_context(), self.len())?;
        {
            let source = self.view();
            let mut destination_view = destination.output();
            self.stream_context.sync_scope(|| {
                // SAFETY: this helper retains its source, destination, and native resources through completion.
                unsafe {
                    arithmetic::cube_root_scaled_to(
                        self.stream_context,
                        &source,
                        &mut destination_view,
                        scale_factor,
                    )
                }
            })?;
        }

        Ok(SignalPipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: SignalBacking::Owned(unsafe { destination.assume_init() }),
        })
    }

    /// Computes the cube root of each element. Returns an owned pipeline containing the result.
    /// Applies the selected native operation's scaling with `scale_factor`.
    ///
    /// Allocates output through the workspace and waits for successful stream completion.
    /// The source must already be ready on the selected stream; this method does not wait for a producer on another stream.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn cube_root_to_scaled<U>(self, scale_factor: i32) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        Workspace: SignalAllocator<U>,
        T: arithmetic::CubeRootScaledTo<U>,
    {
        self.ensure_eager()?;
        self.cube_root_scaled_to::<U>(scale_factor)
    }

    /// Computes the square root of each element. Returns an owned pipeline containing the result.
    /// Applies the selected native operation's scaling with `scale_factor`.
    ///
    /// Allocates output through the workspace and waits for successful stream completion.
    /// The source must already be ready on the selected stream; this method does not wait for a producer on another stream.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn square_root_scaled_to<U>(self, scale_factor: i32) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        Workspace: SignalAllocator<U>,
        T: arithmetic::SquareRootScaledTo<U>,
    {
        self.ensure_eager()?;
        let mut destination = self
            .workspace
            .signal_uninit::<U>(self.stream_context.cuda_context(), self.len())?;
        {
            let source = self.view();
            let mut destination_view = destination.output();
            self.stream_context.sync_scope(|| {
                // SAFETY: this helper retains its source, destination, and native resources through completion.
                unsafe {
                    arithmetic::square_root_scaled_to(
                        self.stream_context,
                        &source,
                        &mut destination_view,
                        scale_factor,
                    )
                }
            })?;
        }

        Ok(SignalPipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: SignalBacking::Owned(unsafe { destination.assume_init() }),
        })
    }

    /// Computes the square root of each element. Returns an owned pipeline containing the result.
    /// Applies the selected native operation's scaling with `scale_factor`.
    ///
    /// Allocates output through the workspace and waits for successful stream completion.
    /// The source must already be ready on the selected stream; this method does not wait for a producer on another stream.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if a length exceeds the native range, if NPP rejects the operation, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn square_root_to_scaled<U>(self, scale_factor: i32) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy + DataTypeLike,
        Workspace: SignalAllocator<U>,
        T: arithmetic::SquareRootScaledTo<U>,
    {
        self.ensure_eager()?;
        self.square_root_scaled_to::<U>(scale_factor)
    }

    fn unary_transform<O>(mut self) -> Result<Self>
    where
        Self: UnaryTransformSignal<T, O>,
    {
        self.ensure_eager()?;
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut();
                <Self as UnaryTransformSignal<T, O>>::unary_transform_signal_in_place(
                    self.stream_context,
                    &mut signal_view,
                )?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self
                    .workspace
                    .signal_uninit::<T>(self.stream_context.cuda_context(), source.len())?;
                let mut destination_view = destination.output();
                <Self as UnaryTransformSignal<T, O>>::unary_transform_signal(
                    self.stream_context,
                    source,
                    &mut destination_view,
                )?;
                // SAFETY: the completed operation above wrote every element.
                self.backing = SignalBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }

    fn unary_output<U, O>(self) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy,
        Workspace: SignalAllocator<U>,
        Self: UnaryOutputSignal<T, U, O>,
    {
        self.ensure_eager()?;
        let mut destination = self
            .workspace
            .signal_uninit::<U>(self.stream_context.cuda_context(), self.len())?;
        {
            let source = self.view();
            let mut destination_view = destination.output();
            <Self as UnaryOutputSignal<T, U, O>>::unary_output_signal(
                self.stream_context,
                &source,
                &mut destination_view,
            )?;
        }

        Ok(SignalPipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: SignalBacking::Owned(unsafe { destination.assume_init() }),
        })
    }

    fn scaled_unary_output<U, O>(self, scale_factor: i32) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy,
        Workspace: SignalAllocator<U>,
        Self: ScaledUnaryOutputSignal<T, U, O>,
    {
        self.ensure_eager()?;
        let mut destination = self
            .workspace
            .signal_uninit::<U>(self.stream_context.cuda_context(), self.len())?;
        {
            let source = self.view();
            let mut destination_view = destination.output();
            <Self as ScaledUnaryOutputSignal<T, U, O>>::scaled_unary_output_signal(
                self.stream_context,
                &source,
                &mut destination_view,
                scale_factor,
            )?;
        }

        Ok(SignalPipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: SignalBacking::Owned(unsafe { destination.assume_init() }),
        })
    }

    fn scaled_unary_transform<O>(mut self, scale_factor: i32) -> Result<Self>
    where
        Self: ScaledUnaryTransformSignal<T, O>,
    {
        self.ensure_eager()?;
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut();
                <Self as ScaledUnaryTransformSignal<T, O>>::scaled_unary_transform_signal_in_place(
                    self.stream_context,
                    &mut signal_view,
                    scale_factor,
                )?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self
                    .workspace
                    .signal_uninit::<T>(self.stream_context.cuda_context(), source.len())?;
                let mut destination_view = destination.output();
                <Self as ScaledUnaryTransformSignal<T, O>>::scaled_unary_transform_signal(
                    self.stream_context,
                    source,
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
