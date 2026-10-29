use singe_cuda::types::{Complex32, Complex64};

use crate::{
    context::StreamContext,
    error::Result,
    pipeline::{
        SignalAllocator, Workspace,
        signal::{SignalBacking, SignalPipeline},
    },
    signal::{
        arithmetic,
        view::{SignalOutputView, SignalView},
    },
    types::ComplexI16,
};

/// Dispatches completed signal normalization.
///
/// # Safety
///
/// Implementations must preserve the native element ABI and normalization semantics,
/// initialize every output element on success, and complete borrowed work on every path.
/// Uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait NormalizeSignal<T> {
    /// Computes `(source - subtrahend) / divisor` for each sample.
    /// Scalar operand values are forwarded to NPP without numerical validation.
    /// Writes into an equally sized `destination`.
    /// Prepares both regions, waits for producers on other streams, rejects capture, and
    /// completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native length conversion, normalization, or stream synchronization fail.
    /// Source and destination lengths must match.
    fn normalize_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        subtrahend: T,
        divisor: T,
    ) -> Result<()>;
}

/// Dispatches completed complex signal normalization.
///
/// # Safety
///
/// Implementations must preserve the native element ABI and normalization semantics,
/// initialize every output element on success, and complete borrowed work on every path.
/// Uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait ComplexNormalizeSignal<T, D> {
    /// Computes `(source - subtrahend) / divisor` for each sample.
    /// Subtracts the complex `subtrahend` and divides both components by the real scalar divisor.
    /// Scalar operand values are forwarded to NPP without numerical validation.
    /// Writes into an equally sized `destination`.
    /// Prepares both regions, waits for producers on other streams, rejects capture, and
    /// completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native length conversion, normalization, or stream synchronization fail.
    /// Source and destination lengths must match.
    fn normalize_complex_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        subtrahend: T,
        divisor: D,
    ) -> Result<()>;
}

/// Dispatches completed fixed-point signal normalization.
///
/// # Safety
///
/// Implementations must preserve the native element ABI and normalization semantics,
/// initialize every output element on success, and complete borrowed work on every path.
/// Uphold the completed native dispatch contract of [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait ScaledNormalizeSignal<T> {
    /// Computes `(source - subtrahend) / divisor` for each sample.
    /// Applies native fixed-point scaling with `scale_factor` and saturation to the output type.
    /// Scalar operand values are forwarded to NPP without numerical validation.
    /// Writes into an equally sized `destination`.
    /// Prepares both regions, waits for producers on other streams, rejects capture, and
    /// completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native length conversion, normalization, or stream synchronization fail.
    /// Source and destination lengths must match.
    fn normalize_signal_scaled(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        subtrahend: T,
        divisor: i32,
        scale_factor: i32,
    ) -> Result<()>;
}

macro_rules! impl_normalize_signal {
    ($ty:ty, $normalize:path) => {
        unsafe impl<'a> NormalizeSignal<$ty> for SignalPipeline<'a, $ty> {
            fn normalize_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalOutputView<'_, $ty>,
                subtrahend: $ty,
                divisor: $ty,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $normalize(stream_context, source, destination, subtrahend, divisor)
                })
            }
        }
    };
}

macro_rules! impl_scaled_normalize_signal {
    ($ty:ty, $normalize:path) => {
        unsafe impl<'a> ScaledNormalizeSignal<$ty> for SignalPipeline<'a, $ty> {
            fn normalize_signal_scaled(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalOutputView<'_, $ty>,
                subtrahend: $ty,
                divisor: i32,
                scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $normalize(
                        stream_context,
                        source,
                        destination,
                        subtrahend,
                        divisor,
                        scale_factor,
                    )
                })
            }
        }
    };
}

macro_rules! impl_complex_normalize_signal {
    ($ty:ty, $divisor:ty, $normalize:path) => {
        unsafe impl<'a> ComplexNormalizeSignal<$ty, $divisor> for SignalPipeline<'a, $ty> {
            fn normalize_complex_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $ty>,
                destination: &mut SignalOutputView<'_, $ty>,
                subtrahend: $ty,
                divisor: $divisor,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $normalize(stream_context, source, destination, subtrahend, divisor)
                })
            }
        }
    };
}

impl_normalize_signal!(f32, arithmetic::normalize_f32);

impl_normalize_signal!(f64, arithmetic::normalize_f64);

impl_complex_normalize_signal!(Complex32, f32, arithmetic::normalize_f32_complex);

impl_complex_normalize_signal!(Complex64, f64, arithmetic::normalize_f64_complex);

impl_scaled_normalize_signal!(i16, arithmetic::normalize_i16_scaled);

impl_scaled_normalize_signal!(ComplexI16, arithmetic::normalize_i16_complex_scaled);

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: NormalizeSignal<T>,
{
    /// Computes `(source - subtrahend) / divisor` for each sample.
    /// Scalar operand values are forwarded to NPP without numerical validation.
    /// Allocates a new owned signal through the workspace, including for owned input.
    /// Prepares both regions, waits for producers on other streams, rejects capture, and
    /// completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native length conversion, normalization, or stream synchronization fail.
    /// Workspace allocation or checking the source producer for capture can also fail.
    pub fn normalize(self, subtrahend: T, divisor: T) -> Result<Self> {
        self.ensure_eager()?;
        self.normalize_out_of_place(|stream_context, source, destination| {
            <Self as NormalizeSignal<T>>::normalize_signal(
                stream_context,
                source,
                destination,
                subtrahend,
                divisor,
            )
        })
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Self: NormalizeSignal<T>,
{
    /// Computes `(source - subtrahend) / divisor` for each sample.
    /// Scalar operand values are forwarded to NPP without numerical validation.
    /// Writes into an equally sized `destination`.
    /// Prepares both regions, waits for producers on other streams, rejects capture, and
    /// completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native length conversion, normalization, or stream synchronization fail.
    /// Source and destination lengths must match.
    pub fn normalize_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        subtrahend: T,
        divisor: T,
    ) -> Result<()> {
        <Self as NormalizeSignal<T>>::normalize_signal(
            stream_context,
            source,
            destination,
            subtrahend,
            divisor,
        )
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
{
    /// Computes `(source - subtrahend) / divisor` for each sample.
    /// Subtracts the complex `subtrahend` and divides both components by the real scalar divisor.
    /// Scalar operand values are forwarded to NPP without numerical validation.
    /// Allocates a new owned signal through the workspace, including for owned input.
    /// Prepares both regions, waits for producers on other streams, rejects capture, and
    /// completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native length conversion, normalization, or stream synchronization fail.
    /// Workspace allocation or checking the source producer for capture can also fail.
    pub fn normalize_complex<D>(self, subtrahend: T, divisor: D) -> Result<Self>
    where
        D: Copy,
        Self: ComplexNormalizeSignal<T, D>,
    {
        self.ensure_eager()?;
        self.normalize_out_of_place(|stream_context, source, destination| {
            <Self as ComplexNormalizeSignal<T, D>>::normalize_complex_signal(
                stream_context,
                source,
                destination,
                subtrahend,
                divisor,
            )
        })
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
{
    /// Computes `(source - subtrahend) / divisor` for each sample.
    /// Subtracts the complex `subtrahend` and divides both components by the real scalar divisor.
    /// Scalar operand values are forwarded to NPP without numerical validation.
    /// Writes into an equally sized `destination`.
    /// Prepares both regions, waits for producers on other streams, rejects capture, and
    /// completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native length conversion, normalization, or stream synchronization fail.
    /// Source and destination lengths must match.
    pub fn normalize_complex_into<D>(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        subtrahend: T,
        divisor: D,
    ) -> Result<()>
    where
        D: Copy,
        Self: ComplexNormalizeSignal<T, D>,
    {
        <Self as ComplexNormalizeSignal<T, D>>::normalize_complex_signal(
            stream_context,
            source,
            destination,
            subtrahend,
            divisor,
        )
    }
}

impl SignalPipeline<'_, i16> {
    /// Computes `(source - subtrahend) / divisor` for each sample.
    /// Applies native fixed-point scaling with `scale_factor` and saturation to the output type.
    /// Scalar operand values are forwarded to NPP without numerical validation.
    /// Writes into an equally sized `destination`.
    /// Prepares both regions, waits for producers on other streams, rejects capture, and
    /// completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native length conversion, normalization, or stream synchronization fail.
    /// Source and destination lengths must match.
    pub fn normalize_scaled_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, i16>,
        destination: &mut SignalOutputView<'_, i16>,
        subtrahend: i16,
        divisor: i32,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ScaledNormalizeSignal<i16>>::normalize_signal_scaled(
            stream_context,
            source,
            destination,
            subtrahend,
            divisor,
            scale_factor,
        )
    }

    /// Computes `(source - subtrahend) / divisor` for each sample.
    /// Applies native fixed-point scaling with `scale_factor` and saturation to the output type.
    /// Scalar operand values are forwarded to NPP without numerical validation.
    /// Allocates a new owned signal through the workspace, including for owned input.
    /// Prepares both regions, waits for producers on other streams, rejects capture, and
    /// completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native length conversion, normalization, or stream synchronization fail.
    /// Workspace allocation or checking the source producer for capture can also fail.
    pub fn normalize_scaled(
        self,
        subtrahend: i16,
        divisor: i32,
        scale_factor: i32,
    ) -> Result<Self> {
        self.ensure_eager()?;
        self.normalize_out_of_place(|stream_context, source, destination| {
            <Self as ScaledNormalizeSignal<i16>>::normalize_signal_scaled(
                stream_context,
                source,
                destination,
                subtrahend,
                divisor,
                scale_factor,
            )
        })
    }
}

impl SignalPipeline<'_, ComplexI16> {
    /// Computes `(source - subtrahend) / divisor` for each sample.
    /// Subtracts the complex `subtrahend` and divides both components by the real scalar divisor.
    /// Applies native fixed-point scaling with `scale_factor` and saturation to the output type.
    /// Scalar operand values are forwarded to NPP without numerical validation.
    /// Writes into an equally sized `destination`.
    /// Prepares both regions, waits for producers on other streams, rejects capture, and
    /// completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native length conversion, normalization, or stream synchronization fail.
    /// Source and destination lengths must match.
    pub fn normalize_scaled_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, ComplexI16>,
        destination: &mut SignalOutputView<'_, ComplexI16>,
        subtrahend: ComplexI16,
        divisor: i32,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as ScaledNormalizeSignal<ComplexI16>>::normalize_signal_scaled(
            stream_context,
            source,
            destination,
            subtrahend,
            divisor,
            scale_factor,
        )
    }

    /// Computes `(source - subtrahend) / divisor` for each sample.
    /// Subtracts the complex `subtrahend` and divides both components by the real scalar divisor.
    /// Applies native fixed-point scaling with `scale_factor` and saturation to the output type.
    /// Scalar operand values are forwarded to NPP without numerical validation.
    /// Allocates a new owned signal through the workspace, including for owned input.
    /// Prepares both regions, waits for producers on other streams, rejects capture, and
    /// completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native length conversion, normalization, or stream synchronization fail.
    /// Workspace allocation or checking the source producer for capture can also fail.
    pub fn normalize_scaled(
        self,
        subtrahend: ComplexI16,
        divisor: i32,
        scale_factor: i32,
    ) -> Result<Self> {
        self.ensure_eager()?;
        self.normalize_out_of_place(|stream_context, source, destination| {
            <Self as ScaledNormalizeSignal<ComplexI16>>::normalize_signal_scaled(
                stream_context,
                source,
                destination,
                subtrahend,
                divisor,
                scale_factor,
            )
        })
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
{
    fn normalize_out_of_place(
        mut self,
        normalize: impl FnOnce(
            &StreamContext,
            &SignalView<'_, T>,
            &mut SignalOutputView<'_, T>,
        ) -> Result<()>,
    ) -> Result<Self> {
        self.ensure_eager()?;
        let mut destination = self
            .workspace
            .signal_uninit::<T>(self.stream_context.cuda_context(), self.len())?;
        {
            let source = self.view();
            let mut destination_view = destination.output();
            normalize(self.stream_context, &source, &mut destination_view)?;
        }
        // SAFETY: the completed operation above wrote every element.
        self.backing = SignalBacking::Owned(unsafe { destination.assume_init() });
        Ok(self)
    }
}
