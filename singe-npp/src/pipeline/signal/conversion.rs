use crate::{
    context::StreamContext,
    error::Result,
    pipeline::{
        SignalAllocator, Workspace,
        signal::{SignalBacking, SignalPipeline},
    },
    signal::{
        conversion,
        view::{SignalOutputView, SignalView},
    },
    types::RoundMode,
};

/// Dispatches an element-type conversion with native conversion rules.
///
/// # Safety
/// Implementations must preserve native element representations and conversion semantics.
/// Complete borrowed native work on every return path, including errors, and initialize all
/// destination elements on success. Uphold the completed dispatch contract of
/// [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait ConvertSignal<T, U> {
    /// Converts each source element to the destination element type. Writes the result to `destination`.
    ///
    /// Prepares source and destination regions, waiting for producers on other streams.
    /// Completes the conversion before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the conversion, or if stream synchronization fails.
    /// Source and destination lengths must match.
    fn convert_signal(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, U>,
    ) -> Result<()>;
}

/// Dispatches an element-type conversion with native scaling.
///
/// # Safety
/// Implementations must preserve native element representations and conversion semantics.
/// Complete borrowed native work on every return path, including errors, and initialize all
/// destination elements on success. Uphold the completed dispatch contract of
/// [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait ScaledConvertSignal<T, U> {
    /// Converts each source element to the destination element type. Writes the result to `destination`.
    /// Applies the selected native conversion's scaling with `scale_factor`.
    ///
    /// Prepares source and destination regions, waiting for producers on other streams.
    /// Completes the conversion before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the conversion, or if stream synchronization fails.
    /// Source and destination lengths must match.
    fn convert_signal_scaled(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, U>,
        scale_factor: i32,
    ) -> Result<()>;
}

/// Dispatches an element-type conversion with native scaling and explicit rounding.
///
/// # Safety
/// Implementations must preserve native element representations and conversion semantics.
/// Complete borrowed native work on every return path, including errors, and initialize all
/// destination elements on success. Uphold the completed dispatch contract of
/// [`SignalPipeline`](crate::pipeline::SignalPipeline).
pub unsafe trait ScaledRoundConvertSignal<T, U> {
    /// Converts each source element to the destination element type. Writes the result to `destination`.
    /// Applies the selected native conversion's scaling with `scale_factor`.
    /// Uses `round_mode` for the native rounding step.
    ///
    /// Prepares source and destination regions, waiting for producers on other streams.
    /// Completes the conversion before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the conversion, or if stream synchronization fails.
    /// Source and destination lengths must match.
    fn convert_signal_scaled_round(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, U>,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<()>;
}

macro_rules! impl_convert_signal {
    ($source:ty, $destination:ty, $convert:path) => {
        unsafe impl<'a> ConvertSignal<$source, $destination> for SignalPipeline<'a, $source> {
            fn convert_signal(
                stream_context: &StreamContext,
                source: &SignalView<'_, $source>,
                destination: &mut SignalOutputView<'_, $destination>,
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

macro_rules! impl_scaled_convert_signal {
    ($source:ty, $destination:ty, $convert:path) => {
        unsafe impl<'a> ScaledConvertSignal<$source, $destination> for SignalPipeline<'a, $source> {
            fn convert_signal_scaled(
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
                    $convert(stream_context, source, destination, scale_factor)
                })
            }
        }
    };
}

macro_rules! impl_scaled_round_convert_signal {
    ($source:ty, $destination:ty, $convert:path) => {
        unsafe impl<'a> ScaledRoundConvertSignal<$source, $destination>
            for SignalPipeline<'a, $source>
        {
            fn convert_signal_scaled_round(
                stream_context: &StreamContext,
                source: &SignalView<'_, $source>,
                destination: &mut SignalOutputView<'_, $destination>,
                round_mode: RoundMode,
                scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $convert(
                        stream_context,
                        source,
                        destination,
                        round_mode,
                        scale_factor,
                    )
                })
            }
        }
    };
}

impl_convert_signal!(i8, f32, conversion::convert_i8_to_f32);

impl_convert_signal!(i8, i16, conversion::convert_i8_to_i16);

impl_convert_signal!(u8, f32, conversion::convert_u8_to_f32);

impl_convert_signal!(i16, f32, conversion::convert_i16_to_f32);

impl_convert_signal!(i16, i32, conversion::convert_i16_to_i32);

impl_convert_signal!(u16, f32, conversion::convert_u16_to_f32);

impl_convert_signal!(i32, i16, conversion::convert_i32_to_i16);

impl_convert_signal!(i32, f32, conversion::convert_i32_to_f32);

impl_convert_signal!(i32, f64, conversion::convert_i32_to_f64);

impl_convert_signal!(f32, f64, conversion::convert_f32_to_f64);

impl_convert_signal!(i64, f64, conversion::convert_i64_to_f64);

impl_convert_signal!(f64, f32, conversion::convert_f64_to_f32);

impl_scaled_convert_signal!(i16, f32, conversion::convert_i16_to_f32_scaled);

impl_scaled_convert_signal!(i16, f64, conversion::convert_i16_to_f64_scaled);

impl_scaled_convert_signal!(i32, i16, conversion::convert_i32_to_i16_scaled);

impl_scaled_convert_signal!(i32, f32, conversion::convert_i32_to_f32_scaled);

impl_scaled_convert_signal!(i32, f64, conversion::convert_i32_to_f64_scaled);

impl_scaled_round_convert_signal!(i16, i8, conversion::convert_i16_to_i8_scaled);

impl_scaled_round_convert_signal!(f32, i8, conversion::convert_f32_to_i8_scaled);

impl_scaled_round_convert_signal!(f32, u8, conversion::convert_f32_to_u8_scaled);

impl_scaled_round_convert_signal!(f32, i16, conversion::convert_f32_to_i16_scaled);

impl_scaled_round_convert_signal!(f32, u16, conversion::convert_f32_to_u16_scaled);

impl_scaled_round_convert_signal!(f32, i32, conversion::convert_f32_to_i32_scaled);

impl_scaled_round_convert_signal!(i64, i32, conversion::convert_i64_to_i32_scaled);

impl_scaled_round_convert_signal!(f64, i16, conversion::convert_f64_to_i16_scaled);

impl_scaled_round_convert_signal!(f64, i32, conversion::convert_f64_to_i32_scaled);

impl_scaled_round_convert_signal!(f64, i64, conversion::convert_f64_to_i64_scaled);

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
{
    /// Converts each source element to the destination element type. Returns an owned pipeline.
    ///
    /// Allocates destination storage through the workspace.
    /// Prepares source and destination regions, waiting for producers on other streams.
    /// Completes the conversion before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the conversion, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn convert_to<U>(self) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy,
        Workspace: SignalAllocator<U>,
        Self: ConvertSignal<T, U>,
    {
        self.ensure_eager()?;
        let mut destination = self
            .workspace
            .signal_uninit::<U>(self.stream_context.cuda_context(), self.len())?;

        {
            let source = self.view();
            let mut destination_view = destination.output();
            <Self as ConvertSignal<T, U>>::convert_signal(
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

    /// Converts each source element to the destination element type. Returns an owned pipeline.
    /// Applies the selected native conversion's scaling with `scale_factor`.
    ///
    /// Allocates destination storage through the workspace.
    /// Prepares source and destination regions, waiting for producers on other streams.
    /// Completes the conversion before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the conversion, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn convert_to_scaled<U>(self, scale_factor: i32) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy,
        Workspace: SignalAllocator<U>,
        Self: ScaledConvertSignal<T, U>,
    {
        self.ensure_eager()?;
        let mut destination = self
            .workspace
            .signal_uninit::<U>(self.stream_context.cuda_context(), self.len())?;

        {
            let source = self.view();
            let mut destination_view = destination.output();
            <Self as ScaledConvertSignal<T, U>>::convert_signal_scaled(
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

    /// Converts each source element to the destination element type. Returns an owned pipeline.
    /// Applies the selected native conversion's scaling with `scale_factor`.
    /// Uses `round_mode` for the native rounding step.
    ///
    /// Allocates destination storage through the workspace.
    /// Prepares source and destination regions, waiting for producers on other streams.
    /// Completes the conversion before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the conversion, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn convert_to_scaled_round<U>(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, U>>
    where
        U: Copy,
        Workspace: SignalAllocator<U>,
        Self: ScaledRoundConvertSignal<T, U>,
    {
        self.ensure_eager()?;
        let mut destination = self
            .workspace
            .signal_uninit::<U>(self.stream_context.cuda_context(), self.len())?;

        {
            let source = self.view();
            let mut destination_view = destination.output();
            <Self as ScaledRoundConvertSignal<T, U>>::convert_signal_scaled_round(
                self.stream_context,
                &source,
                &mut destination_view,
                round_mode,
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

    /// Converts each source element to the destination element type. Writes the result to `destination`.
    ///
    /// Prepares source and destination regions, waiting for producers on other streams.
    /// Completes the conversion before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the conversion, or if stream synchronization fails.
    /// Source and destination lengths must match.
    pub fn convert_into<U>(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, U>,
    ) -> Result<()>
    where
        U: Copy,
        Self: ConvertSignal<T, U>,
    {
        <Self as ConvertSignal<T, U>>::convert_signal(stream_context, source, destination)
    }

    /// Converts each source element to the destination element type. Writes the result to `destination`.
    /// Applies the selected native conversion's scaling with `scale_factor`.
    ///
    /// Prepares source and destination regions, waiting for producers on other streams.
    /// Completes the conversion before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the conversion, or if stream synchronization fails.
    /// Source and destination lengths must match.
    pub fn convert_scaled_into<U>(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, U>,
        scale_factor: i32,
    ) -> Result<()>
    where
        U: Copy,
        Self: ScaledConvertSignal<T, U>,
    {
        <Self as ScaledConvertSignal<T, U>>::convert_signal_scaled(
            stream_context,
            source,
            destination,
            scale_factor,
        )
    }

    /// Converts each source element to the destination element type. Writes the result to `destination`.
    /// Applies the selected native conversion's scaling with `scale_factor`.
    /// Uses `round_mode` for the native rounding step.
    ///
    /// Prepares source and destination regions, waiting for producers on other streams.
    /// Completes the conversion before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the conversion, or if stream synchronization fails.
    /// Source and destination lengths must match.
    pub fn convert_scaled_round_into<U>(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, U>,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<()>
    where
        U: Copy,
        Self: ScaledRoundConvertSignal<T, U>,
    {
        <Self as ScaledRoundConvertSignal<T, U>>::convert_signal_scaled_round(
            stream_context,
            source,
            destination,
            round_mode,
            scale_factor,
        )
    }
}

impl<'a> SignalPipeline<'a, u8> {
    /// Converts each `u8` element to `f32` and returns an owned pipeline.
    /// Uses [`convert_u8_to_f32`](crate::signal::raw::conversion::convert_u8_to_f32).
    ///
    /// Allocates destination storage through the workspace.
    /// Prepares source and destination regions, waiting for producers on other streams.
    /// Completes the conversion before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the conversion, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn convert_to_f32(self) -> Result<SignalPipeline<'a, f32>> {
        self.ensure_eager()?;
        self.convert_to()
    }
}

impl<'a> SignalPipeline<'a, i8> {
    /// Converts each `i8` element to `f32` and returns an owned pipeline.
    /// Uses [`convert_i8_to_f32`](crate::signal::raw::conversion::convert_i8_to_f32).
    ///
    /// Allocates destination storage through the workspace.
    /// Prepares source and destination regions, waiting for producers on other streams.
    /// Completes the conversion before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the conversion, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn convert_to_f32(self) -> Result<SignalPipeline<'a, f32>> {
        self.ensure_eager()?;
        self.convert_to()
    }

    /// Converts each `i8` element to `i16` and returns an owned pipeline.
    /// Uses [`convert_i8_to_i16`](crate::signal::raw::conversion::convert_i8_to_i16).
    ///
    /// Allocates destination storage through the workspace.
    /// Prepares source and destination regions, waiting for producers on other streams.
    /// Completes the conversion before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the conversion, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn convert_to_i16(self) -> Result<SignalPipeline<'a, i16>> {
        self.ensure_eager()?;
        self.convert_to()
    }
}

impl<'a> SignalPipeline<'a, u16> {
    /// Converts each `u16` element to `f32` and returns an owned pipeline.
    /// Uses [`convert_u16_to_f32`](crate::signal::raw::conversion::convert_u16_to_f32).
    ///
    /// Allocates destination storage through the workspace.
    /// Prepares source and destination regions, waiting for producers on other streams.
    /// Completes the conversion before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the conversion, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn convert_to_f32(self) -> Result<SignalPipeline<'a, f32>> {
        self.ensure_eager()?;
        self.convert_to()
    }
}

impl<'a> SignalPipeline<'a, i16> {
    /// Converts each `i16` element to `f32` and returns an owned pipeline.
    /// Uses [`convert_i16_to_f32`](crate::signal::raw::conversion::convert_i16_to_f32).
    ///
    /// Allocates destination storage through the workspace.
    /// Prepares source and destination regions, waiting for producers on other streams.
    /// Completes the conversion before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the conversion, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn convert_to_f32(self) -> Result<SignalPipeline<'a, f32>> {
        self.ensure_eager()?;
        self.convert_to()
    }

    /// Converts each `i16` element to `i32` and returns an owned pipeline.
    /// Uses [`convert_i16_to_i32`](crate::signal::raw::conversion::convert_i16_to_i32).
    ///
    /// Allocates destination storage through the workspace.
    /// Prepares source and destination regions, waiting for producers on other streams.
    /// Completes the conversion before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the conversion, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn convert_to_i32(self) -> Result<SignalPipeline<'a, i32>> {
        self.ensure_eager()?;
        self.convert_to()
    }

    /// Converts each `i16` element to `f32` and returns an owned pipeline.
    /// Uses [`convert_i16_to_f32_scaled`](crate::signal::raw::conversion::convert_i16_to_f32_scaled).
    /// Applies the selected native conversion's scaling with `scale_factor`.
    ///
    /// Allocates destination storage through the workspace.
    /// Prepares source and destination regions, waiting for producers on other streams.
    /// Completes the conversion before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the conversion, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn convert_to_f32_scaled(self, scale_factor: i32) -> Result<SignalPipeline<'a, f32>> {
        self.ensure_eager()?;
        self.convert_to_scaled(scale_factor)
    }

    /// Converts each `i16` element to `f64` and returns an owned pipeline.
    /// Uses [`convert_i16_to_f64_scaled`](crate::signal::raw::conversion::convert_i16_to_f64_scaled).
    /// Applies the selected native conversion's scaling with `scale_factor`.
    ///
    /// Allocates destination storage through the workspace.
    /// Prepares source and destination regions, waiting for producers on other streams.
    /// Completes the conversion before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the conversion, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn convert_to_f64_scaled(self, scale_factor: i32) -> Result<SignalPipeline<'a, f64>> {
        self.ensure_eager()?;
        self.convert_to_scaled(scale_factor)
    }

    /// Converts each `i16` element to `i8` and returns an owned pipeline.
    /// Uses [`convert_i16_to_i8_scaled`](crate::signal::raw::conversion::convert_i16_to_i8_scaled).
    /// Applies the selected native conversion's scaling with `scale_factor`.
    /// Uses `round_mode` for the native rounding step.
    ///
    /// Allocates destination storage through the workspace.
    /// Prepares source and destination regions, waiting for producers on other streams.
    /// Completes the conversion before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the conversion, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn convert_to_i8_scaled(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i8>> {
        self.ensure_eager()?;
        self.convert_to_scaled_round(round_mode, scale_factor)
    }
}

impl<'a> SignalPipeline<'a, i32> {
    /// Converts each `i32` element to `i16` and returns an owned pipeline.
    /// Uses [`convert_i32_to_i16`](crate::signal::raw::conversion::convert_i32_to_i16).
    ///
    /// Allocates destination storage through the workspace.
    /// Prepares source and destination regions, waiting for producers on other streams.
    /// Completes the conversion before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the conversion, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn convert_to_i16(self) -> Result<SignalPipeline<'a, i16>> {
        self.ensure_eager()?;
        self.convert_to()
    }

    /// Converts each `i32` element to `f32` and returns an owned pipeline.
    /// Uses [`convert_i32_to_f32`](crate::signal::raw::conversion::convert_i32_to_f32).
    ///
    /// Allocates destination storage through the workspace.
    /// Prepares source and destination regions, waiting for producers on other streams.
    /// Completes the conversion before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the conversion, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn convert_to_f32(self) -> Result<SignalPipeline<'a, f32>> {
        self.ensure_eager()?;
        self.convert_to()
    }

    /// Converts each `i32` element to `f64` and returns an owned pipeline.
    /// Uses [`convert_i32_to_f64`](crate::signal::raw::conversion::convert_i32_to_f64).
    ///
    /// Allocates destination storage through the workspace.
    /// Prepares source and destination regions, waiting for producers on other streams.
    /// Completes the conversion before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the conversion, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn convert_to_f64(self) -> Result<SignalPipeline<'a, f64>> {
        self.ensure_eager()?;
        self.convert_to()
    }

    /// Converts each `i32` element to `i16` and returns an owned pipeline.
    /// Uses [`convert_i32_to_i16_scaled`](crate::signal::raw::conversion::convert_i32_to_i16_scaled).
    /// Applies the selected native conversion's scaling with `scale_factor`.
    ///
    /// Allocates destination storage through the workspace.
    /// Prepares source and destination regions, waiting for producers on other streams.
    /// Completes the conversion before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the conversion, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn convert_to_i16_scaled(self, scale_factor: i32) -> Result<SignalPipeline<'a, i16>> {
        self.ensure_eager()?;
        self.convert_to_scaled(scale_factor)
    }

    /// Converts each `i32` element to `f32` and returns an owned pipeline.
    /// Uses [`convert_i32_to_f32_scaled`](crate::signal::raw::conversion::convert_i32_to_f32_scaled).
    /// Applies the selected native conversion's scaling with `scale_factor`.
    ///
    /// Allocates destination storage through the workspace.
    /// Prepares source and destination regions, waiting for producers on other streams.
    /// Completes the conversion before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the conversion, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn convert_to_f32_scaled(self, scale_factor: i32) -> Result<SignalPipeline<'a, f32>> {
        self.ensure_eager()?;
        self.convert_to_scaled(scale_factor)
    }

    /// Converts each `i32` element to `f64` and returns an owned pipeline.
    /// Uses [`convert_i32_to_f64_scaled`](crate::signal::raw::conversion::convert_i32_to_f64_scaled).
    /// Applies the selected native conversion's scaling with `scale_factor`.
    ///
    /// Allocates destination storage through the workspace.
    /// Prepares source and destination regions, waiting for producers on other streams.
    /// Completes the conversion before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the conversion, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn convert_to_f64_scaled(self, scale_factor: i32) -> Result<SignalPipeline<'a, f64>> {
        self.ensure_eager()?;
        self.convert_to_scaled(scale_factor)
    }
}

impl<'a> SignalPipeline<'a, f32> {
    /// Converts each `f32` element to `f64` and returns an owned pipeline.
    /// Uses [`convert_f32_to_f64`](crate::signal::raw::conversion::convert_f32_to_f64).
    ///
    /// Allocates destination storage through the workspace.
    /// Prepares source and destination regions, waiting for producers on other streams.
    /// Completes the conversion before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the conversion, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn convert_to_f64(self) -> Result<SignalPipeline<'a, f64>> {
        self.ensure_eager()?;
        self.convert_to()
    }

    /// Converts each `f32` element to `u8` and returns an owned pipeline.
    /// Uses [`convert_f32_to_u8_scaled`](crate::signal::raw::conversion::convert_f32_to_u8_scaled).
    /// Applies the selected native conversion's scaling with `scale_factor`.
    /// Uses `round_mode` for the native rounding step.
    ///
    /// Allocates destination storage through the workspace.
    /// Prepares source and destination regions, waiting for producers on other streams.
    /// Completes the conversion before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the conversion, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn convert_to_u8_scaled(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, u8>> {
        self.ensure_eager()?;
        self.convert_to_scaled_round(round_mode, scale_factor)
    }

    /// Converts each `f32` element to `i8` and returns an owned pipeline.
    /// Uses [`convert_f32_to_i8_scaled`](crate::signal::raw::conversion::convert_f32_to_i8_scaled).
    /// Applies the selected native conversion's scaling with `scale_factor`.
    /// Uses `round_mode` for the native rounding step.
    ///
    /// Allocates destination storage through the workspace.
    /// Prepares source and destination regions, waiting for producers on other streams.
    /// Completes the conversion before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the conversion, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn convert_to_i8_scaled(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i8>> {
        self.ensure_eager()?;
        self.convert_to_scaled_round(round_mode, scale_factor)
    }

    /// Converts each `f32` element to `i16` and returns an owned pipeline.
    /// Uses [`convert_f32_to_i16_scaled`](crate::signal::raw::conversion::convert_f32_to_i16_scaled).
    /// Applies the selected native conversion's scaling with `scale_factor`.
    /// Uses `round_mode` for the native rounding step.
    ///
    /// Allocates destination storage through the workspace.
    /// Prepares source and destination regions, waiting for producers on other streams.
    /// Completes the conversion before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the conversion, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn convert_to_i16_scaled(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i16>> {
        self.ensure_eager()?;
        self.convert_to_scaled_round(round_mode, scale_factor)
    }

    /// Converts each `f32` element to `u16` and returns an owned pipeline.
    /// Uses [`convert_f32_to_u16_scaled`](crate::signal::raw::conversion::convert_f32_to_u16_scaled).
    /// Applies the selected native conversion's scaling with `scale_factor`.
    /// Uses `round_mode` for the native rounding step.
    ///
    /// Allocates destination storage through the workspace.
    /// Prepares source and destination regions, waiting for producers on other streams.
    /// Completes the conversion before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the conversion, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn convert_to_u16_scaled(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, u16>> {
        self.ensure_eager()?;
        self.convert_to_scaled_round(round_mode, scale_factor)
    }

    /// Converts each `f32` element to `i32` and returns an owned pipeline.
    /// Uses [`convert_f32_to_i32_scaled`](crate::signal::raw::conversion::convert_f32_to_i32_scaled).
    /// Applies the selected native conversion's scaling with `scale_factor`.
    /// Uses `round_mode` for the native rounding step.
    ///
    /// Allocates destination storage through the workspace.
    /// Prepares source and destination regions, waiting for producers on other streams.
    /// Completes the conversion before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the conversion, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn convert_to_i32_scaled(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i32>> {
        self.ensure_eager()?;
        self.convert_to_scaled_round(round_mode, scale_factor)
    }
}

impl<'a> SignalPipeline<'a, i64> {
    /// Converts each `i64` element to `f64` and returns an owned pipeline.
    /// Uses [`convert_i64_to_f64`](crate::signal::raw::conversion::convert_i64_to_f64).
    ///
    /// Allocates destination storage through the workspace.
    /// Prepares source and destination regions, waiting for producers on other streams.
    /// Completes the conversion before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the conversion, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn convert_to_f64(self) -> Result<SignalPipeline<'a, f64>> {
        self.ensure_eager()?;
        self.convert_to()
    }

    /// Converts each `i64` element to `i32` and returns an owned pipeline.
    /// Uses [`convert_i64_to_i32_scaled`](crate::signal::raw::conversion::convert_i64_to_i32_scaled).
    /// Applies the selected native conversion's scaling with `scale_factor`.
    /// Uses `round_mode` for the native rounding step.
    ///
    /// Allocates destination storage through the workspace.
    /// Prepares source and destination regions, waiting for producers on other streams.
    /// Completes the conversion before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the conversion, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn convert_to_i32_scaled(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i32>> {
        self.ensure_eager()?;
        self.convert_to_scaled_round(round_mode, scale_factor)
    }
}

impl<'a> SignalPipeline<'a, f64> {
    /// Converts each `f64` element to `f32` and returns an owned pipeline.
    /// Uses [`convert_f64_to_f32`](crate::signal::raw::conversion::convert_f64_to_f32).
    ///
    /// Allocates destination storage through the workspace.
    /// Prepares source and destination regions, waiting for producers on other streams.
    /// Completes the conversion before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the conversion, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn convert_to_f32(self) -> Result<SignalPipeline<'a, f32>> {
        self.ensure_eager()?;
        self.convert_to()
    }

    /// Converts each `f64` element to `i16` and returns an owned pipeline.
    /// Uses [`convert_f64_to_i16_scaled`](crate::signal::raw::conversion::convert_f64_to_i16_scaled).
    /// Applies the selected native conversion's scaling with `scale_factor`.
    /// Uses `round_mode` for the native rounding step.
    ///
    /// Allocates destination storage through the workspace.
    /// Prepares source and destination regions, waiting for producers on other streams.
    /// Completes the conversion before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the conversion, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn convert_to_i16_scaled(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i16>> {
        self.ensure_eager()?;
        self.convert_to_scaled_round(round_mode, scale_factor)
    }

    /// Converts each `f64` element to `i32` and returns an owned pipeline.
    /// Uses [`convert_f64_to_i32_scaled`](crate::signal::raw::conversion::convert_f64_to_i32_scaled).
    /// Applies the selected native conversion's scaling with `scale_factor`.
    /// Uses `round_mode` for the native rounding step.
    ///
    /// Allocates destination storage through the workspace.
    /// Prepares source and destination regions, waiting for producers on other streams.
    /// Completes the conversion before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the conversion, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn convert_to_i32_scaled(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i32>> {
        self.ensure_eager()?;
        self.convert_to_scaled_round(round_mode, scale_factor)
    }

    /// Converts each `f64` element to `i64` and returns an owned pipeline.
    /// Uses [`convert_f64_to_i64_scaled`](crate::signal::raw::conversion::convert_f64_to_i64_scaled).
    /// Applies the selected native conversion's scaling with `scale_factor`.
    /// Uses `round_mode` for the native rounding step.
    ///
    /// Allocates destination storage through the workspace.
    /// Prepares source and destination regions, waiting for producers on other streams.
    /// Completes the conversion before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the conversion, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn convert_to_i64_scaled(
        self,
        round_mode: RoundMode,
        scale_factor: i32,
    ) -> Result<SignalPipeline<'a, i64>> {
        self.ensure_eager()?;
        self.convert_to_scaled_round(round_mode, scale_factor)
    }
}
