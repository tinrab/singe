use crate::{
    context::StreamContext,
    error::Result,
    pipeline::{
        signal::{SignalBacking, SignalPipeline, operations::operation_traits::UnarySignal},
        workspace::{SignalAllocator, Workspace},
    },
    signal::view::{SignalOutputView, SignalView, SignalViewMut},
};

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: UnarySignal<T>,
{
    /// Computes the natural exponential of each sample.
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    ///
    /// Prepares participating regions, waiting for producers on other streams, rejects capture,
    /// and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails,
    /// if waiting for a producer fails, if a length exceeds the native range, if NPP rejects
    /// the operation, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn exponent(mut self) -> Result<Self> {
        self.ensure_eager()?;
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut();
                <Self as UnarySignal<T>>::exponent_signal_in_place(
                    self.stream_context,
                    &mut signal_view,
                )?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self
                    .workspace
                    .signal_uninit::<T>(self.stream_context.cuda_context(), source.len())?;
                let mut destination_view = destination.output();
                <Self as UnarySignal<T>>::exponent_signal(
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

    /// Computes the natural logarithm of each sample.
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    ///
    /// Prepares participating regions, waiting for producers on other streams, rejects capture,
    /// and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails,
    /// if waiting for a producer fails, if a length exceeds the native range, if NPP rejects
    /// the operation, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn natural_logarithm(mut self) -> Result<Self> {
        self.ensure_eager()?;
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut();
                <Self as UnarySignal<T>>::natural_logarithm_signal_in_place(
                    self.stream_context,
                    &mut signal_view,
                )?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self
                    .workspace
                    .signal_uninit::<T>(self.stream_context.cuda_context(), source.len())?;
                let mut destination_view = destination.output();
                <Self as UnarySignal<T>>::natural_logarithm_signal(
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

    /// Squares each sample.
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    ///
    /// Prepares participating regions, waiting for producers on other streams, rejects capture,
    /// and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails,
    /// if waiting for a producer fails, if a length exceeds the native range, if NPP rejects
    /// the operation, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn square(mut self) -> Result<Self> {
        self.ensure_eager()?;
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut();
                <Self as UnarySignal<T>>::square_signal_in_place(
                    self.stream_context,
                    &mut signal_view,
                )?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self
                    .workspace
                    .signal_uninit::<T>(self.stream_context.cuda_context(), source.len())?;
                let mut destination_view = destination.output();
                <Self as UnarySignal<T>>::square_signal(
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

    /// Computes the square root of each sample.
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    ///
    /// Prepares participating regions, waiting for producers on other streams, rejects capture,
    /// and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails,
    /// if waiting for a producer fails, if a length exceeds the native range, if NPP rejects
    /// the operation, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn square_root(mut self) -> Result<Self> {
        self.ensure_eager()?;
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut();
                <Self as UnarySignal<T>>::square_root_signal_in_place(
                    self.stream_context,
                    &mut signal_view,
                )?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self
                    .workspace
                    .signal_uninit::<T>(self.stream_context.cuda_context(), source.len())?;
                let mut destination_view = destination.output();
                <Self as UnarySignal<T>>::square_root_signal(
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
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Self: UnarySignal<T>,
{
    /// Computes the natural exponential of each sample. Writes the result to `destination`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, rejects capture,
    /// and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails,
    /// if waiting for a producer fails, if a length exceeds the native range, if NPP rejects
    /// the operation, or if stream synchronization fails.
    /// Source and destination lengths must match.
    pub fn exponent_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        <Self as UnarySignal<T>>::exponent_signal(stream_context, source, destination)
    }

    /// Computes the natural exponential of each sample. Updates the signal in place.
    ///
    /// Prepares participating regions, waiting for producers on other streams, rejects capture,
    /// and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails,
    /// if waiting for a producer fails, if a length exceeds the native range, if NPP rejects
    /// the operation, or if stream synchronization fails.
    pub fn exponent_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as UnarySignal<T>>::exponent_signal_in_place(stream_context, signal)
    }

    /// Computes the natural logarithm of each sample. Writes the result to `destination`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, rejects capture,
    /// and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails,
    /// if waiting for a producer fails, if a length exceeds the native range, if NPP rejects
    /// the operation, or if stream synchronization fails.
    /// Source and destination lengths must match.
    pub fn natural_logarithm_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        <Self as UnarySignal<T>>::natural_logarithm_signal(stream_context, source, destination)
    }

    /// Computes the natural logarithm of each sample. Updates the signal in place.
    ///
    /// Prepares participating regions, waiting for producers on other streams, rejects capture,
    /// and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails,
    /// if waiting for a producer fails, if a length exceeds the native range, if NPP rejects
    /// the operation, or if stream synchronization fails.
    pub fn natural_logarithm_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as UnarySignal<T>>::natural_logarithm_signal_in_place(stream_context, signal)
    }

    /// Squares each sample. Writes the result to `destination`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, rejects capture,
    /// and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails,
    /// if waiting for a producer fails, if a length exceeds the native range, if NPP rejects
    /// the operation, or if stream synchronization fails.
    /// Source and destination lengths must match.
    pub fn square_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        <Self as UnarySignal<T>>::square_signal(stream_context, source, destination)
    }

    /// Squares each sample. Updates the signal in place.
    ///
    /// Prepares participating regions, waiting for producers on other streams, rejects capture,
    /// and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails,
    /// if waiting for a producer fails, if a length exceeds the native range, if NPP rejects
    /// the operation, or if stream synchronization fails.
    pub fn square_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as UnarySignal<T>>::square_signal_in_place(stream_context, signal)
    }

    /// Computes the square root of each sample. Writes the result to `destination`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, rejects capture,
    /// and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails,
    /// if waiting for a producer fails, if a length exceeds the native range, if NPP rejects
    /// the operation, or if stream synchronization fails.
    /// Source and destination lengths must match.
    pub fn square_root_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        <Self as UnarySignal<T>>::square_root_signal(stream_context, source, destination)
    }

    /// Computes the square root of each sample. Updates the signal in place.
    ///
    /// Prepares participating regions, waiting for producers on other streams, rejects capture,
    /// and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails,
    /// if waiting for a producer fails, if a length exceeds the native range, if NPP rejects
    /// the operation, or if stream synchronization fails.
    pub fn square_root_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
    ) -> Result<()> {
        <Self as UnarySignal<T>>::square_root_signal_in_place(stream_context, signal)
    }
}
