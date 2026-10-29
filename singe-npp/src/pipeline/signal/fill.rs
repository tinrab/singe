use singe_cuda::types::{Complex32, Complex64};

use crate::{
    context::StreamContext,
    error::Result,
    pipeline::{
        SignalAllocator, Workspace,
        signal::{SignalBacking, SignalPipeline},
    },
    signal::{initialization, view::SignalOutputView},
    types::{ComplexI16, ComplexI32, ComplexI64},
};

/// # Safety
/// Uphold the completed native dispatch contract of
/// [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait SetSignal<T> {
    fn set_signal(
        stream_context: &StreamContext,
        value: T,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()>;
}

/// # Safety
/// Uphold the completed native dispatch contract of
/// [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait ZeroSignal<T> {
    fn zero_signal(
        stream_context: &StreamContext,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()>;
}

macro_rules! impl_set_signal {
    ($ty:ty, $set:path) => {
        unsafe impl<'a> SetSignal<$ty> for SignalPipeline<'a, $ty> {
            fn set_signal(
                stream_context: &StreamContext,
                value: $ty,
                destination: &mut SignalOutputView<'_, $ty>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe { $set(stream_context, value, destination) })
            }
        }
    };
}

macro_rules! impl_zero_signal {
    ($ty:ty, $zero:path) => {
        unsafe impl<'a> ZeroSignal<$ty> for SignalPipeline<'a, $ty> {
            fn zero_signal(
                stream_context: &StreamContext,
                destination: &mut SignalOutputView<'_, $ty>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe { $zero(stream_context, destination) })
            }
        }
    };
}

impl_set_signal!(u8, initialization::set_u8);

impl_set_signal!(i8, initialization::set_i8);

impl_set_signal!(u16, initialization::set_u16);

impl_set_signal!(i16, initialization::set_i16);

impl_set_signal!(u32, initialization::set_u32);

impl_set_signal!(i32, initialization::set_i32);

impl_set_signal!(i64, initialization::set_i64);

impl_set_signal!(ComplexI16, initialization::set_i16_complex);

impl_set_signal!(ComplexI32, initialization::set_i32_complex);

impl_set_signal!(ComplexI64, initialization::set_i64_complex);

impl_set_signal!(f32, initialization::set_f32);

impl_set_signal!(Complex32, initialization::set_f32_complex);

impl_set_signal!(f64, initialization::set_f64);

impl_set_signal!(Complex64, initialization::set_f64_complex);

impl_zero_signal!(u8, initialization::zero_u8);

impl_zero_signal!(i16, initialization::zero_i16);

impl_zero_signal!(i32, initialization::zero_i32);

impl_zero_signal!(i64, initialization::zero_i64);

impl_zero_signal!(ComplexI16, initialization::zero_i16_complex);

impl_zero_signal!(ComplexI32, initialization::zero_i32_complex);

impl_zero_signal!(ComplexI64, initialization::zero_i64_complex);

impl_zero_signal!(f32, initialization::zero_f32);

impl_zero_signal!(Complex32, initialization::zero_f32_complex);

impl_zero_signal!(f64, initialization::zero_f64);

impl_zero_signal!(Complex64, initialization::zero_f64_complex);

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Self: SetSignal<T>,
{
    /// Fills every signal element with `value`.
    ///
    /// Prepares the output region, waits for its producers on other streams, rejects
    /// capture, and returns after successful native execution and stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits where performed, native execution, or stream synchronization fail.
    pub fn set_into(
        stream_context: &StreamContext,
        value: T,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        <Self as SetSignal<T>>::set_signal(stream_context, value, destination)
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Self: ZeroSignal<T>,
{
    /// Fills every signal element with zero.
    ///
    /// Prepares the output region, waits for its producers on other streams, rejects
    /// capture, and returns after successful native execution and stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits where performed, native execution, or stream synchronization fail.
    pub fn zero_into(
        stream_context: &StreamContext,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        <Self as ZeroSignal<T>>::zero_signal(stream_context, destination)
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: SetSignal<T>,
{
    /// Fills every signal element with `value`.
    ///
    /// Reuses owned storage in place. For borrowed input, obtains a new buffer of the same
    /// length from the workspace and initializes every output element.
    /// Prepares the output region, waits for its producers on other streams, rejects
    /// capture, and returns after successful native execution and stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits where performed, native execution, or stream synchronization fail.
    /// Workspace output allocation can also fail.
    pub fn set(mut self, value: T) -> Result<Self> {
        self.ensure_eager()?;
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.output_view();
                <Self as SetSignal<T>>::set_signal(self.stream_context, value, &mut signal_view)?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self
                    .workspace
                    .signal_uninit::<T>(self.stream_context.cuda_context(), source.len())?;
                let mut destination_view = destination.output();
                <Self as SetSignal<T>>::set_signal(
                    self.stream_context,
                    value,
                    &mut destination_view,
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
    Self: ZeroSignal<T>,
{
    /// Fills every signal element with zero.
    ///
    /// Reuses owned storage in place. For borrowed input, obtains a new buffer of the same
    /// length from the workspace and initializes every output element.
    /// Prepares the output region, waits for its producers on other streams, rejects
    /// capture, and returns after successful native execution and stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits where performed, native execution, or stream synchronization fail.
    /// Workspace output allocation can also fail.
    pub fn zero(mut self) -> Result<Self> {
        self.ensure_eager()?;
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.output_view();
                <Self as ZeroSignal<T>>::zero_signal(self.stream_context, &mut signal_view)?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self
                    .workspace
                    .signal_uninit::<T>(self.stream_context.cuda_context(), source.len())?;
                let mut destination_view = destination.output();
                <Self as ZeroSignal<T>>::zero_signal(self.stream_context, &mut destination_view)?;
                // SAFETY: the completed operation above wrote every element.
                self.backing = SignalBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}

// These element types have set operations but no dedicated native zero entry point.
macro_rules! impl_zero_via_set {
    ($ty:ty) => {
        unsafe impl<'a> ZeroSignal<$ty> for SignalPipeline<'a, $ty> {
            fn zero_signal(
                stream_context: &StreamContext,
                destination: &mut SignalOutputView<'_, $ty>,
            ) -> Result<()> {
                <Self as SetSignal<$ty>>::set_signal(stream_context, 0, destination)
            }
        }
    };
}

impl_zero_via_set!(i8);

impl_zero_via_set!(u16);

impl_zero_via_set!(u32);
