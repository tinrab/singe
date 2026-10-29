use crate::{
    context::StreamContext,
    error::Result,
    pipeline::{
        signal::{
            SignalBacking, SignalPipeline,
            operations::operation_traits::{
                ConstantSignal, DivideIntoConstantSignal, SubtractFromConstantSignal,
            },
        },
        workspace::{SignalAllocator, Workspace},
    },
    signal::view::{SignalOutputView, SignalView, SignalViewMut},
};

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: ConstantSignal<T>,
{
    /// Adds `value` to each pipeline element. Returns an owned pipeline containing the result.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn add_constant(mut self, value: T) -> Result<Self> {
        self.ensure_eager()?;
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut();
                <Self as ConstantSignal<T>>::add_constant_signal_in_place(
                    self.stream_context,
                    &mut signal_view,
                    value,
                )?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self
                    .workspace
                    .signal_uninit::<T>(self.stream_context.cuda_context(), source.len())?;
                let mut destination_view = destination.output();
                <Self as ConstantSignal<T>>::add_constant_signal(
                    self.stream_context,
                    source,
                    value,
                    &mut destination_view,
                )?;
                // SAFETY: the completed operation above wrote every element.
                self.backing = SignalBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }

    /// Subtracts `value` from each pipeline element. Returns an owned pipeline containing the result.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn subtract_constant(mut self, value: T) -> Result<Self> {
        self.ensure_eager()?;
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut();
                <Self as ConstantSignal<T>>::subtract_constant_signal_in_place(
                    self.stream_context,
                    &mut signal_view,
                    value,
                )?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self
                    .workspace
                    .signal_uninit::<T>(self.stream_context.cuda_context(), source.len())?;
                let mut destination_view = destination.output();
                <Self as ConstantSignal<T>>::subtract_constant_signal(
                    self.stream_context,
                    source,
                    value,
                    &mut destination_view,
                )?;
                // SAFETY: the completed operation above wrote every element.
                self.backing = SignalBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }

    /// Multiplies each pipeline element by `value`. Returns an owned pipeline containing the result.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn multiply_constant(mut self, value: T) -> Result<Self> {
        self.ensure_eager()?;
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut();
                <Self as ConstantSignal<T>>::multiply_constant_signal_in_place(
                    self.stream_context,
                    &mut signal_view,
                    value,
                )?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self
                    .workspace
                    .signal_uninit::<T>(self.stream_context.cuda_context(), source.len())?;
                let mut destination_view = destination.output();
                <Self as ConstantSignal<T>>::multiply_constant_signal(
                    self.stream_context,
                    source,
                    value,
                    &mut destination_view,
                )?;
                // SAFETY: the completed operation above wrote every element.
                self.backing = SignalBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }

    /// Divides each pipeline element by `value`. Returns an owned pipeline containing the result.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn divide_constant(mut self, value: T) -> Result<Self> {
        self.ensure_eager()?;
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut();
                <Self as ConstantSignal<T>>::divide_constant_signal_in_place(
                    self.stream_context,
                    &mut signal_view,
                    value,
                )?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self
                    .workspace
                    .signal_uninit::<T>(self.stream_context.cuda_context(), source.len())?;
                let mut destination_view = destination.output();
                <Self as ConstantSignal<T>>::divide_constant_signal(
                    self.stream_context,
                    source,
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
    Self: ConstantSignal<T>,
{
    /// Adds `value` to each source element. Writes the result to `destination`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Source and destination lengths must match.
    pub fn add_constant_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        <Self as ConstantSignal<T>>::add_constant_signal(stream_context, source, value, destination)
    }

    /// Adds `value` to each signal element. Updates the signal in place.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn add_constant_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
    ) -> Result<()> {
        <Self as ConstantSignal<T>>::add_constant_signal_in_place(stream_context, signal, value)
    }

    /// Subtracts `value` from each source element. Writes the result to `destination`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Source and destination lengths must match.
    pub fn subtract_constant_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        <Self as ConstantSignal<T>>::subtract_constant_signal(
            stream_context,
            source,
            value,
            destination,
        )
    }

    /// Subtracts `value` from each signal element. Updates the signal in place.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn subtract_constant_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
    ) -> Result<()> {
        <Self as ConstantSignal<T>>::subtract_constant_signal_in_place(
            stream_context,
            signal,
            value,
        )
    }

    /// Multiplies each source element by `value`. Writes the result to `destination`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Source and destination lengths must match.
    pub fn multiply_constant_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        <Self as ConstantSignal<T>>::multiply_constant_signal(
            stream_context,
            source,
            value,
            destination,
        )
    }

    /// Multiplies each signal element by `value`. Updates the signal in place.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn multiply_constant_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
    ) -> Result<()> {
        <Self as ConstantSignal<T>>::multiply_constant_signal_in_place(
            stream_context,
            signal,
            value,
        )
    }

    /// Divides each source element by `value`. Writes the result to `destination`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Source and destination lengths must match.
    pub fn divide_constant_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        <Self as ConstantSignal<T>>::divide_constant_signal(
            stream_context,
            source,
            value,
            destination,
        )
    }

    /// Divides each signal element by `value`. Updates the signal in place.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn divide_constant_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
    ) -> Result<()> {
        <Self as ConstantSignal<T>>::divide_constant_signal_in_place(stream_context, signal, value)
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: SubtractFromConstantSignal<T>,
{
    /// Subtracts each pipeline element from `value`. Returns an owned pipeline containing the result.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn subtract_from_constant(self, value: T) -> Result<Self> {
        self.ensure_eager()?;
        self.reverse_constant_arithmetic(
            value,
            <Self as SubtractFromConstantSignal<T>>::subtract_from_constant_signal,
            <Self as SubtractFromConstantSignal<T>>::subtract_from_constant_signal_in_place,
        )
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: DivideIntoConstantSignal<T>,
{
    /// Divides `value` by each pipeline element. Returns an owned pipeline containing the result.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn divide_into_constant(self, value: T) -> Result<Self> {
        self.ensure_eager()?;
        self.reverse_constant_arithmetic(
            value,
            <Self as DivideIntoConstantSignal<T>>::divide_into_constant_signal,
            <Self as DivideIntoConstantSignal<T>>::divide_into_constant_signal_in_place,
        )
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Self: SubtractFromConstantSignal<T>,
{
    /// Subtracts each source element from `value`. Writes the result to `destination`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Source and destination lengths must match.
    pub fn subtract_from_constant_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        <Self as SubtractFromConstantSignal<T>>::subtract_from_constant_signal(
            stream_context,
            source,
            value,
            destination,
        )
    }

    /// Subtracts each signal element from `value`. Updates the signal in place.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn subtract_from_constant_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
    ) -> Result<()> {
        <Self as SubtractFromConstantSignal<T>>::subtract_from_constant_signal_in_place(
            stream_context,
            signal,
            value,
        )
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Self: DivideIntoConstantSignal<T>,
{
    /// Divides `value` by each source element. Writes the result to `destination`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    /// Source and destination lengths must match.
    pub fn divide_into_constant_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        value: T,
        destination: &mut SignalOutputView<'_, T>,
    ) -> Result<()> {
        <Self as DivideIntoConstantSignal<T>>::divide_into_constant_signal(
            stream_context,
            source,
            value,
            destination,
        )
    }

    /// Divides `value` by each signal element. Updates the signal in place.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the arithmetic parameters, or if stream synchronization fails.
    pub fn divide_into_constant_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        value: T,
    ) -> Result<()> {
        <Self as DivideIntoConstantSignal<T>>::divide_into_constant_signal_in_place(
            stream_context,
            signal,
            value,
        )
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
{
    fn reverse_constant_arithmetic(
        mut self,
        value: T,
        operation: fn(
            &StreamContext,
            &SignalView<'_, T>,
            T,
            &mut SignalOutputView<'_, T>,
        ) -> Result<()>,
        operation_in_place: fn(&StreamContext, &mut SignalViewMut<'_, T>, T) -> Result<()>,
    ) -> Result<Self> {
        self.ensure_eager()?;
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut();
                operation_in_place(self.stream_context, &mut signal_view, value)?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self
                    .workspace
                    .signal_uninit::<T>(self.stream_context.cuda_context(), source.len())?;
                let mut destination_view = destination.output();
                operation(self.stream_context, source, value, &mut destination_view)?;
                // SAFETY: the completed operation above wrote every element.
                self.backing = SignalBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}
