mod binary;
mod conversion;
mod copy;
mod fill;
mod filtering;
mod normalize;
mod operations;
mod scaled_constant;
mod statistics;
mod statistics_error_methods;
mod statistics_extremum_methods;
mod statistics_methods;
mod statistics_output_methods;
mod unary;

use std::{mem::MaybeUninit, sync::Arc};

pub use copy::CopySignal;
use singe_cuda::view::{DeviceRepr, DeviceSlice, DeviceView};
pub use statistics::{SignalIndexedValue, SignalMeanStandardDeviation, SignalMinMax};

use crate::{
    context::StreamContext,
    error::{Error, Result},
    pipeline::{SignalAllocator, Workspace},
    signal::{
        memory::Signal,
        view::{SignalOutputView, SignalView},
    },
};

/// Storage retained by a signal pipeline, either borrowed or owned.
#[derive(Debug)]
#[non_exhaustive]
pub enum SignalBacking<'a, T> {
    /// A readable signal view borrowed from caller-owned storage.
    Borrowed(SignalView<'a, T>),
    /// An owned device signal allocation.
    Owned(Signal<T>),
}

/// An eager signal pipeline whose operations complete before returning.
///
/// Ordinary NPP allocation and synchronous free are incompatible with capture.
/// Rejected consuming operations retain owned backing until capture ends so its
/// native destructor cannot invalidate capture. Use public unsafe raw signal
/// operations with caller-retained storage when recording graphs.
///
/// # Native dispatch implementations
/// Unsafe dispatch traits must preserve their native element ABI and operation
/// semantics, validate contexts and dimensions, and report accurate scratch sizes.
/// Success establishes valid values in the operation's declared result ranges.
/// Every return and unwind must complete borrowed work, and capture must be
/// rejected before submission or temporary allocation.
#[derive(Debug)]
pub struct SignalPipeline<'a, T> {
    stream_context: &'a StreamContext,
    workspace: &'a mut Workspace,
    backing: SignalBacking<'a, T>,
}

impl<'a, T> SignalPipeline<'a, T>
where
    T: Copy,
{
    /// Allocates an uninitialized signal through `workspace` for later initialization.
    ///
    /// Returns a pending pipeline retaining the stream context and workspace borrow.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture or if workspace allocation fails.
    pub fn create_uninit(
        stream_context: &'a StreamContext,
        workspace: &'a mut Workspace,
        len: usize,
    ) -> Result<UninitSignalPipeline<'a, T>>
    where
        Workspace: SignalAllocator<T>,
    {
        stream_context.stream().ensure_not_capturing()?;
        let signal = workspace.signal_uninit::<T>(stream_context.cuda_context(), len)?;
        Ok(UninitSignalPipeline {
            stream_context,
            workspace,
            signal,
        })
    }

    /// Creates a pipeline over a borrowed signal without copying it.
    ///
    /// Checks the CUDA context but does not wait for pending producers or reject capture here.
    /// Individual eager operations prepare their input regions as documented.
    ///
    /// # Errors
    ///
    /// Returns an error if `source` belongs to another CUDA context.
    pub fn from_view(
        stream_context: &'a StreamContext,
        workspace: &'a mut Workspace,
        source: SignalView<'a, T>,
    ) -> Result<Self> {
        if source.context() != stream_context.cuda_context() {
            return Err(Error::StreamContextMismatch);
        }
        Ok(Self {
            stream_context,
            workspace,
            backing: SignalBacking::Borrowed(source),
        })
    }

    /// Creates a pipeline by taking ownership of `signal` without copying its pixels.
    ///
    /// Checks the CUDA context and rejects capture; it does not synchronize a pending producer.
    /// If rejected during capture, retains the allocation until capture ends.
    ///
    /// # Errors
    ///
    /// Returns an error if capture-retention setup or the capture check fails, or if the signal
    /// belongs to another CUDA context.
    pub fn from_owned(
        stream_context: &'a StreamContext,
        workspace: &'a mut Workspace,
        signal: Signal<T>,
    ) -> Result<Self> {
        stream_context
            .stream()
            .defer_drop_until_capture_end(Arc::clone(signal.allocation()))?;
        stream_context.stream().ensure_not_capturing()?;
        if signal.context() != stream_context.cuda_context() {
            return Err(Error::StreamContextMismatch);
        }
        Ok(Self {
            stream_context,
            workspace,
            backing: SignalBacking::Owned(signal),
        })
    }

    /// Borrows the first `len` elements of `memory` as a readable signal pipeline.
    ///
    /// Does not copy the buffer or wait for pending producers.
    ///
    /// # Errors
    ///
    /// Returns an error if `len` exceeds the buffer length, constructing the device or signal
    /// view fails, or the buffer belongs to another CUDA context.
    pub fn from_buffer<B>(
        stream_context: &'a StreamContext,
        workspace: &'a mut Workspace,
        memory: &'a B,
        len: usize,
    ) -> Result<Self>
    where
        T: DeviceRepr,
        B: DeviceSlice<T> + ?Sized,
    {
        Self::from_view(
            stream_context,
            workspace,
            SignalView::from_view(DeviceView::from_buffer(memory).slice(..len)?)?,
        )
    }

    fn ensure_eager(&self) -> Result<()> {
        if let SignalBacking::Owned(signal) = &self.backing {
            self.stream_context
                .stream()
                .defer_drop_until_capture_end(Arc::clone(signal.allocation()))?;
        }
        self.stream_context.stream().ensure_not_capturing()?;
        if let Some(producer) = self.view().region().stream() {
            producer.ensure_not_capturing()?;
        }
        Ok(())
    }

    /// Returns the number of samples in the current backing.
    #[must_use]
    pub const fn len(&self) -> usize {
        match &self.backing {
            SignalBacking::Borrowed(view) => view.len(),
            SignalBacking::Owned(signal) => signal.len(),
        }
    }

    /// Returns whether the current backing contains no samples.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns whether this pipeline owns its signal storage.
    #[must_use]
    pub const fn is_owned(&self) -> bool {
        matches!(self.backing, SignalBacking::Owned(_))
    }

    fn view(&self) -> SignalView<'_, T> {
        match &self.backing {
            SignalBacking::Borrowed(view) => *view,
            SignalBacking::Owned(signal) => signal.view(),
        }
    }

    /// Consumes the pipeline and returns its current borrowed or owned storage.
    ///
    /// Does not copy samples or synchronize the stream.
    #[must_use]
    pub fn into_backing(self) -> SignalBacking<'a, T> {
        self.backing
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: CopySignal<T>,
{
    /// Consumes the pipeline and returns an owned signal.
    ///
    /// Transfers existing owned storage directly. For borrowed storage, allocates through the
    /// workspace, prepares source and destination regions, waits for producers on other streams,
    /// and completes the copy before returning. The direct ownership-transfer path does not
    /// synchronize a pending producer.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if the source producer is being captured,
    /// if capture-retention setup or workspace allocation fails, or if preparing regions,
    /// waiting for producers, copying samples, or synchronizing the copy fails.
    pub fn finish(self) -> Result<Signal<T>> {
        self.ensure_eager()?;
        match self.backing {
            SignalBacking::Owned(signal) => Ok(signal),
            SignalBacking::Borrowed(source) => {
                let mut destination = self
                    .workspace
                    .signal_uninit::<T>(self.stream_context.cuda_context(), source.len())?;
                let mut destination_view = destination.output();
                <Self as CopySignal<T>>::copy(self.stream_context, &source, &mut destination_view)?;
                // SAFETY: the completed operation above wrote every element.
                Ok(unsafe { destination.assume_init() })
            }
        }
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Self: CopySignal<T>,
{
    /// Copies all source samples into an equally sized `destination`.
    ///
    /// Prepares both regions, waits for producers on other streams, rejects stream capture,
    /// and completes the copy before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if lengths differ, a region belongs to another CUDA context,
    /// capture checks or producer waits fail, native length conversion or copying fails,
    /// or stream synchronization fails.
    pub fn copy_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        <Self as CopySignal<T>>::copy(stream_context, source, destination)
    }
}

/// A pending signal pipeline that exposes initialization before readable operations.
#[derive(Debug)]
pub struct UninitSignalPipeline<'a, T> {
    stream_context: &'a StreamContext,
    workspace: &'a mut Workspace,
    signal: Signal<MaybeUninit<T>>,
}

impl<'a, T: Copy> UninitSignalPipeline<'a, T> {
    fn ensure_eager(&self) -> Result<()> {
        self.stream_context
            .stream()
            .defer_drop_until_capture_end(Arc::clone(self.signal.allocation()))?;
        self.stream_context.stream().ensure_not_capturing()?;
        Ok(())
    }

    /// Initializes every sample to `value` and returns an owned readable pipeline.
    ///
    /// Prepares the output region, waits for producers on other streams, rejects capture,
    /// and completes the native fill before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if capture-retention setup, capture checks, CUDA context validation,
    /// region preparation, producer waits, native length conversion, filling samples,
    /// or stream synchronization fails.
    pub fn set(mut self, value: T) -> Result<SignalPipeline<'a, T>>
    where
        SignalPipeline<'a, T>: fill::SetSignal<T>,
    {
        self.ensure_eager()?;
        <SignalPipeline<'a, T> as fill::SetSignal<T>>::set_signal(
            self.stream_context,
            value,
            &mut self.signal.output(),
        )?;
        // SAFETY: the completed set initialized every requested element.
        let signal = unsafe { self.signal.assume_init() };
        SignalPipeline::from_owned(self.stream_context, self.workspace, signal)
    }

    /// Initializes every sample to the native zero value and returns an owned readable pipeline.
    ///
    /// Prepares the output region, waits for producers on other streams, rejects capture,
    /// and completes the native fill before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if capture-retention setup, capture checks, CUDA context validation,
    /// region preparation, producer waits, native length conversion, zeroing samples,
    /// or stream synchronization fails.
    pub fn zero(mut self) -> Result<SignalPipeline<'a, T>>
    where
        SignalPipeline<'a, T>: fill::ZeroSignal<T>,
    {
        self.ensure_eager()?;
        <SignalPipeline<'a, T> as fill::ZeroSignal<T>>::zero_signal(
            self.stream_context,
            &mut self.signal.output(),
        )?;
        // SAFETY: the completed zero operation initialized every requested element.
        let signal = unsafe { self.signal.assume_init() };
        SignalPipeline::from_owned(self.stream_context, self.workspace, signal)
    }

    /// Borrows the pending signal as writable output without initializing its samples.
    ///
    /// Does not submit work or wait for a producer.
    pub fn output(&mut self) -> SignalOutputView<'_, T> {
        self.signal.output()
    }

    /// Consumes the pending pipeline and returns its uninitialized signal allocation.
    ///
    /// The samples retain their current initialization state. This method does not synchronize
    /// pending work or promote the allocation to readable `T` samples.
    #[must_use]
    pub fn into_signal(self) -> Signal<MaybeUninit<T>> {
        self.signal
    }
}
