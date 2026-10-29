use crate::{
    context::StreamContext,
    error::Result,
    pipeline::{
        signal::{
            SignalBacking, SignalPipeline,
            operations::operation_traits::{
                ComplexFixedThresholdSignal, ComplexThresholdSignal, ComplexValueThresholdSignal,
                FixedThresholdSignal, ThresholdSignal, ValueThresholdSignal,
            },
        },
        workspace::{SignalAllocator, Workspace},
    },
    signal::view::{SignalOutputView, SignalView, SignalViewMut},
    types::ComparisonOperation,
};

type ComparisonThresholdCallback<T, L> = fn(
    &StreamContext,
    &SignalView<'_, T>,
    &mut SignalOutputView<'_, T>,
    L,
    ComparisonOperation,
) -> Result<()>;

type ValueThresholdCallback<T> =
    fn(&StreamContext, &SignalView<'_, T>, &mut SignalOutputView<'_, T>, T, T) -> Result<()>;

type ComplexValueThresholdCallback<T, L> =
    fn(&StreamContext, &SignalView<'_, T>, &mut SignalOutputView<'_, T>, L, T) -> Result<()>;

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: ThresholdSignal<T>,
{
    /// Applies the native selected comparison threshold at `level` to each sample. Returns an owned pipeline containing the result.
    /// Native threshold dispatch accepts `ComparisonOperation::Less` or `ComparisonOperation::Greater`. The wrapper forwards the selected operation to NPP.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn threshold(mut self, level: T, operation: ComparisonOperation) -> Result<Self> {
        self.ensure_eager()?;
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut();
                <Self as ThresholdSignal<T>>::threshold_signal_in_place(
                    self.stream_context,
                    &mut signal_view,
                    level,
                    operation,
                )?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self
                    .workspace
                    .signal_uninit::<T>(self.stream_context.cuda_context(), source.len())?;
                let mut destination_view = destination.output();
                <Self as ThresholdSignal<T>>::threshold_signal(
                    self.stream_context,
                    source,
                    &mut destination_view,
                    level,
                    operation,
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
    Self: ThresholdSignal<T>,
{
    /// Applies the native selected comparison threshold at `level` to each sample. Writes the result to `destination`.
    /// Native threshold dispatch accepts `ComparisonOperation::Less` or `ComparisonOperation::Greater`. The wrapper forwards the selected operation to NPP.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    /// Source and destination lengths must match.
    pub fn threshold_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        level: T,
        operation: ComparisonOperation,
    ) -> Result<()> {
        <Self as ThresholdSignal<T>>::threshold_signal(
            stream_context,
            source,
            destination,
            level,
            operation,
        )
    }

    /// Applies the native selected comparison threshold at `level` to each sample. Updates the signal in place.
    /// Native threshold dispatch accepts `ComparisonOperation::Less` or `ComparisonOperation::Greater`. The wrapper forwards the selected operation to NPP.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    pub fn threshold_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: T,
        operation: ComparisonOperation,
    ) -> Result<()> {
        <Self as ThresholdSignal<T>>::threshold_signal_in_place(
            stream_context,
            signal,
            level,
            operation,
        )
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
{
    /// Applies the native selected comparison threshold at `level` to each complex sample. Returns an owned pipeline containing the result.
    /// Native threshold dispatch accepts `ComparisonOperation::Less` or `ComparisonOperation::Greater`. The wrapper forwards the selected operation to NPP.
    /// `level` is the real-valued threshold parameter for the native complex operation and must be greater than zero. The wrapper forwards its value without checking that restriction.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn threshold_complex<L>(self, level: L, operation: ComparisonOperation) -> Result<Self>
    where
        L: Copy,
        Self: ComplexThresholdSignal<T, L>,
    {
        self.ensure_eager()?;
        self.complex_threshold(
            level,
            operation,
            <Self as ComplexThresholdSignal<T, L>>::threshold_complex_signal,
            <Self as ComplexThresholdSignal<T, L>>::threshold_complex_signal_in_place,
        )
    }

    fn complex_threshold<L>(
        mut self,
        level: L,
        operation: ComparisonOperation,
        threshold: ComparisonThresholdCallback<T, L>,
        threshold_in_place: fn(
            &StreamContext,
            &mut SignalViewMut<'_, T>,
            L,
            ComparisonOperation,
        ) -> Result<()>,
    ) -> Result<Self>
    where
        L: Copy,
    {
        self.ensure_eager()?;
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut();
                threshold_in_place(self.stream_context, &mut signal_view, level, operation)?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self
                    .workspace
                    .signal_uninit::<T>(self.stream_context.cuda_context(), source.len())?;
                let mut destination_view = destination.output();
                threshold(
                    self.stream_context,
                    source,
                    &mut destination_view,
                    level,
                    operation,
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
{
    /// Applies the native selected comparison threshold at `level` to each complex sample. Writes the result to `destination`.
    /// Native threshold dispatch accepts `ComparisonOperation::Less` or `ComparisonOperation::Greater`. The wrapper forwards the selected operation to NPP.
    /// `level` is the real-valued threshold parameter for the native complex operation and must be greater than zero. The wrapper forwards its value without checking that restriction.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    /// Source and destination lengths must match.
    pub fn threshold_complex_into<L>(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        level: L,
        operation: ComparisonOperation,
    ) -> Result<()>
    where
        L: Copy,
        Self: ComplexThresholdSignal<T, L>,
    {
        <Self as ComplexThresholdSignal<T, L>>::threshold_complex_signal(
            stream_context,
            source,
            destination,
            level,
            operation,
        )
    }

    /// Applies the native selected comparison threshold at `level` to each complex sample. Updates the signal in place.
    /// Native threshold dispatch accepts `ComparisonOperation::Less` or `ComparisonOperation::Greater`. The wrapper forwards the selected operation to NPP.
    /// `level` is the real-valued threshold parameter for the native complex operation and must be greater than zero. The wrapper forwards its value without checking that restriction.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    pub fn threshold_complex_in_place<L>(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: L,
        operation: ComparisonOperation,
    ) -> Result<()>
    where
        L: Copy,
        Self: ComplexThresholdSignal<T, L>,
    {
        <Self as ComplexThresholdSignal<T, L>>::threshold_complex_signal_in_place(
            stream_context,
            signal,
            level,
            operation,
        )
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: FixedThresholdSignal<T>,
{
    /// Applies the native less-than threshold at `level` to each sample. Returns an owned pipeline containing the result.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn threshold_less(self, level: T) -> Result<Self> {
        self.ensure_eager()?;
        self.fixed_threshold(
            level,
            <Self as FixedThresholdSignal<T>>::threshold_less_signal,
            <Self as FixedThresholdSignal<T>>::threshold_less_signal_in_place,
        )
    }

    /// Applies the native greater-than threshold at `level` to each sample. Returns an owned pipeline containing the result.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn threshold_greater(self, level: T) -> Result<Self> {
        self.ensure_eager()?;
        self.fixed_threshold(
            level,
            <Self as FixedThresholdSignal<T>>::threshold_greater_signal,
            <Self as FixedThresholdSignal<T>>::threshold_greater_signal_in_place,
        )
    }

    fn fixed_threshold(
        mut self,
        level: T,
        threshold: fn(
            &StreamContext,
            &SignalView<'_, T>,
            &mut SignalOutputView<'_, T>,
            T,
        ) -> Result<()>,
        threshold_in_place: fn(&StreamContext, &mut SignalViewMut<'_, T>, T) -> Result<()>,
    ) -> Result<Self> {
        self.ensure_eager()?;
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut();
                threshold_in_place(self.stream_context, &mut signal_view, level)?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self
                    .workspace
                    .signal_uninit::<T>(self.stream_context.cuda_context(), source.len())?;
                let mut destination_view = destination.output();
                threshold(self.stream_context, source, &mut destination_view, level)?;
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
    Self: FixedThresholdSignal<T>,
{
    /// Applies the native less-than threshold at `level` to each sample. Writes the result to `destination`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    /// Source and destination lengths must match.
    pub fn threshold_less_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        level: T,
    ) -> Result<()> {
        <Self as FixedThresholdSignal<T>>::threshold_less_signal(
            stream_context,
            source,
            destination,
            level,
        )
    }

    /// Applies the native less-than threshold at `level` to each sample. Updates the signal in place.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    pub fn threshold_less_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: T,
    ) -> Result<()> {
        <Self as FixedThresholdSignal<T>>::threshold_less_signal_in_place(
            stream_context,
            signal,
            level,
        )
    }

    /// Applies the native greater-than threshold at `level` to each sample. Writes the result to `destination`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    /// Source and destination lengths must match.
    pub fn threshold_greater_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        level: T,
    ) -> Result<()> {
        <Self as FixedThresholdSignal<T>>::threshold_greater_signal(
            stream_context,
            source,
            destination,
            level,
        )
    }

    /// Applies the native greater-than threshold at `level` to each sample. Updates the signal in place.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    pub fn threshold_greater_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: T,
    ) -> Result<()> {
        <Self as FixedThresholdSignal<T>>::threshold_greater_signal_in_place(
            stream_context,
            signal,
            level,
        )
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
{
    /// Applies the native less-than threshold at `level` to each complex sample. Returns an owned pipeline containing the result.
    /// `level` is the real-valued threshold parameter for the native complex operation and must be greater than zero. The wrapper forwards its value without checking that restriction.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn threshold_less_complex<L>(self, level: L) -> Result<Self>
    where
        L: Copy,
        Self: ComplexFixedThresholdSignal<T, L>,
    {
        self.ensure_eager()?;
        self.complex_fixed_threshold(
            level,
            <Self as ComplexFixedThresholdSignal<T, L>>::threshold_less_complex_signal,
            <Self as ComplexFixedThresholdSignal<T, L>>::threshold_less_complex_signal_in_place,
        )
    }

    /// Applies the native greater-than threshold at `level` to each complex sample. Returns an owned pipeline containing the result.
    /// `level` is the real-valued threshold parameter for the native complex operation and must be greater than zero. The wrapper forwards its value without checking that restriction.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn threshold_greater_complex<L>(self, level: L) -> Result<Self>
    where
        L: Copy,
        Self: ComplexFixedThresholdSignal<T, L>,
    {
        self.ensure_eager()?;
        self.complex_fixed_threshold(
            level,
            <Self as ComplexFixedThresholdSignal<T, L>>::threshold_greater_complex_signal,
            <Self as ComplexFixedThresholdSignal<T, L>>::threshold_greater_complex_signal_in_place,
        )
    }

    fn complex_fixed_threshold<L>(
        mut self,
        level: L,
        threshold: fn(
            &StreamContext,
            &SignalView<'_, T>,
            &mut SignalOutputView<'_, T>,
            L,
        ) -> Result<()>,
        threshold_in_place: fn(&StreamContext, &mut SignalViewMut<'_, T>, L) -> Result<()>,
    ) -> Result<Self>
    where
        L: Copy,
    {
        self.ensure_eager()?;
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut();
                threshold_in_place(self.stream_context, &mut signal_view, level)?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self
                    .workspace
                    .signal_uninit::<T>(self.stream_context.cuda_context(), source.len())?;
                let mut destination_view = destination.output();
                threshold(self.stream_context, source, &mut destination_view, level)?;
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
{
    /// Applies the native less-than threshold at `level` to each complex sample. Writes the result to `destination`.
    /// `level` is the real-valued threshold parameter for the native complex operation and must be greater than zero. The wrapper forwards its value without checking that restriction.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    /// Source and destination lengths must match.
    pub fn threshold_less_complex_into<L>(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        level: L,
    ) -> Result<()>
    where
        L: Copy,
        Self: ComplexFixedThresholdSignal<T, L>,
    {
        <Self as ComplexFixedThresholdSignal<T, L>>::threshold_less_complex_signal(
            stream_context,
            source,
            destination,
            level,
        )
    }

    /// Applies the native less-than threshold at `level` to each complex sample. Updates the signal in place.
    /// `level` is the real-valued threshold parameter for the native complex operation and must be greater than zero. The wrapper forwards its value without checking that restriction.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    pub fn threshold_less_complex_in_place<L>(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: L,
    ) -> Result<()>
    where
        L: Copy,
        Self: ComplexFixedThresholdSignal<T, L>,
    {
        <Self as ComplexFixedThresholdSignal<T, L>>::threshold_less_complex_signal_in_place(
            stream_context,
            signal,
            level,
        )
    }

    /// Applies the native greater-than threshold at `level` to each complex sample. Writes the result to `destination`.
    /// `level` is the real-valued threshold parameter for the native complex operation and must be greater than zero. The wrapper forwards its value without checking that restriction.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    /// Source and destination lengths must match.
    pub fn threshold_greater_complex_into<L>(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        level: L,
    ) -> Result<()>
    where
        L: Copy,
        Self: ComplexFixedThresholdSignal<T, L>,
    {
        <Self as ComplexFixedThresholdSignal<T, L>>::threshold_greater_complex_signal(
            stream_context,
            source,
            destination,
            level,
        )
    }

    /// Applies the native greater-than threshold at `level` to each complex sample. Updates the signal in place.
    /// `level` is the real-valued threshold parameter for the native complex operation and must be greater than zero. The wrapper forwards its value without checking that restriction.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    pub fn threshold_greater_complex_in_place<L>(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: L,
    ) -> Result<()>
    where
        L: Copy,
        Self: ComplexFixedThresholdSignal<T, L>,
    {
        <Self as ComplexFixedThresholdSignal<T, L>>::threshold_greater_complex_signal_in_place(
            stream_context,
            signal,
            level,
        )
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
    Self: ValueThresholdSignal<T>,
{
    /// Replaces samples selected by the native less-than threshold at `level` with `value`. Returns an owned pipeline containing the result.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn threshold_less_value(self, level: T, value: T) -> Result<Self> {
        self.ensure_eager()?;
        self.value_threshold(
            level,
            value,
            <Self as ValueThresholdSignal<T>>::threshold_less_value_signal,
            <Self as ValueThresholdSignal<T>>::threshold_less_value_signal_in_place,
        )
    }

    /// Replaces samples selected by the native greater-than threshold at `level` with `value`. Returns an owned pipeline containing the result.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn threshold_greater_value(self, level: T, value: T) -> Result<Self> {
        self.ensure_eager()?;
        self.value_threshold(
            level,
            value,
            <Self as ValueThresholdSignal<T>>::threshold_greater_value_signal,
            <Self as ValueThresholdSignal<T>>::threshold_greater_value_signal_in_place,
        )
    }

    fn value_threshold(
        mut self,
        level: T,
        value: T,
        threshold: ValueThresholdCallback<T>,
        threshold_in_place: fn(&StreamContext, &mut SignalViewMut<'_, T>, T, T) -> Result<()>,
    ) -> Result<Self> {
        self.ensure_eager()?;
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut();
                threshold_in_place(self.stream_context, &mut signal_view, level, value)?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self
                    .workspace
                    .signal_uninit::<T>(self.stream_context.cuda_context(), source.len())?;
                let mut destination_view = destination.output();
                threshold(
                    self.stream_context,
                    source,
                    &mut destination_view,
                    level,
                    value,
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
    Self: ValueThresholdSignal<T>,
{
    /// Replaces samples selected by the native less-than threshold at `level` with `value`. Writes the result to `destination`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    /// Source and destination lengths must match.
    pub fn threshold_less_value_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        level: T,
        value: T,
    ) -> Result<()> {
        <Self as ValueThresholdSignal<T>>::threshold_less_value_signal(
            stream_context,
            source,
            destination,
            level,
            value,
        )
    }

    /// Replaces samples selected by the native less-than threshold at `level` with `value`. Updates the signal in place.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    pub fn threshold_less_value_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: T,
        value: T,
    ) -> Result<()> {
        <Self as ValueThresholdSignal<T>>::threshold_less_value_signal_in_place(
            stream_context,
            signal,
            level,
            value,
        )
    }

    /// Replaces samples selected by the native greater-than threshold at `level` with `value`. Writes the result to `destination`.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    /// Source and destination lengths must match.
    pub fn threshold_greater_value_into(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        level: T,
        value: T,
    ) -> Result<()> {
        <Self as ValueThresholdSignal<T>>::threshold_greater_value_signal(
            stream_context,
            source,
            destination,
            level,
            value,
        )
    }

    /// Replaces samples selected by the native greater-than threshold at `level` with `value`. Updates the signal in place.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    pub fn threshold_greater_value_in_place(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: T,
        value: T,
    ) -> Result<()> {
        <Self as ValueThresholdSignal<T>>::threshold_greater_value_signal_in_place(
            stream_context,
            signal,
            level,
            value,
        )
    }
}

impl<T> SignalPipeline<'_, T>
where
    T: Copy,
    Workspace: SignalAllocator<T>,
{
    /// Replaces samples selected by the native less-than threshold at `level` with `value`. Returns an owned pipeline containing the result.
    /// `level` is the real-valued threshold parameter for the native complex operation and must be greater than zero. The wrapper forwards its value without checking that restriction.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn threshold_less_value_complex<L>(self, level: L, value: T) -> Result<Self>
    where
        L: Copy,
        Self: ComplexValueThresholdSignal<T, L>,
    {
        self.ensure_eager()?;
        self.complex_value_threshold(
            level,
            value,
            <Self as ComplexValueThresholdSignal<T, L>>::threshold_less_value_complex_signal,
            <Self as ComplexValueThresholdSignal<T, L>>::threshold_less_value_complex_signal_in_place,
        )
    }

    /// Replaces samples selected by the native greater-than threshold at `level` with `value`. Returns an owned pipeline containing the result.
    /// `level` is the real-valued threshold parameter for the native complex operation and must be greater than zero. The wrapper forwards its value without checking that restriction.
    ///
    /// Reuses owned signal storage, or allocates output through the workspace for a borrowed signal.
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    /// Workspace allocation and capture checks on the source producer can also fail.
    pub fn threshold_greater_value_complex<L>(self, level: L, value: T) -> Result<Self>
    where
        L: Copy,
        Self: ComplexValueThresholdSignal<T, L>,
    {
        self.ensure_eager()?;
        self.complex_value_threshold(
            level,
            value,
            <Self as ComplexValueThresholdSignal<T, L>>::threshold_greater_value_complex_signal,
            <Self as ComplexValueThresholdSignal<T, L>>::threshold_greater_value_complex_signal_in_place,
        )
    }

    fn complex_value_threshold<L>(
        mut self,
        level: L,
        value: T,
        threshold: ComplexValueThresholdCallback<T, L>,
        threshold_in_place: fn(&StreamContext, &mut SignalViewMut<'_, T>, L, T) -> Result<()>,
    ) -> Result<Self>
    where
        L: Copy,
    {
        self.ensure_eager()?;
        match &mut self.backing {
            SignalBacking::Owned(signal) => {
                let mut signal_view = signal.view_mut();
                threshold_in_place(self.stream_context, &mut signal_view, level, value)?;
            }
            SignalBacking::Borrowed(source) => {
                let mut destination = self
                    .workspace
                    .signal_uninit::<T>(self.stream_context.cuda_context(), source.len())?;
                let mut destination_view = destination.output();
                threshold(
                    self.stream_context,
                    source,
                    &mut destination_view,
                    level,
                    value,
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
{
    /// Replaces samples selected by the native less-than threshold at `level` with `value`. Writes the result to `destination`.
    /// `level` is the real-valued threshold parameter for the native complex operation and must be greater than zero. The wrapper forwards its value without checking that restriction.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    /// Source and destination lengths must match.
    pub fn threshold_less_value_complex_into<L>(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        level: L,
        value: T,
    ) -> Result<()>
    where
        L: Copy,
        Self: ComplexValueThresholdSignal<T, L>,
    {
        <Self as ComplexValueThresholdSignal<T, L>>::threshold_less_value_complex_signal(
            stream_context,
            source,
            destination,
            level,
            value,
        )
    }

    /// Replaces samples selected by the native less-than threshold at `level` with `value`. Updates the signal in place.
    /// `level` is the real-valued threshold parameter for the native complex operation and must be greater than zero. The wrapper forwards its value without checking that restriction.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    pub fn threshold_less_value_complex_in_place<L>(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: L,
        value: T,
    ) -> Result<()>
    where
        L: Copy,
        Self: ComplexValueThresholdSignal<T, L>,
    {
        <Self as ComplexValueThresholdSignal<T, L>>::threshold_less_value_complex_signal_in_place(
            stream_context,
            signal,
            level,
            value,
        )
    }

    /// Replaces samples selected by the native greater-than threshold at `level` with `value`. Writes the result to `destination`.
    /// `level` is the real-valued threshold parameter for the native complex operation and must be greater than zero. The wrapper forwards its value without checking that restriction.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    /// Source and destination lengths must match.
    pub fn threshold_greater_value_complex_into<L>(
        stream_context: &StreamContext,
        source: &SignalView<'_, T>,
        destination: &mut SignalOutputView<'_, T>,
        level: L,
        value: T,
    ) -> Result<()>
    where
        L: Copy,
        Self: ComplexValueThresholdSignal<T, L>,
    {
        <Self as ComplexValueThresholdSignal<T, L>>::threshold_greater_value_complex_signal(
            stream_context,
            source,
            destination,
            level,
            value,
        )
    }

    /// Replaces samples selected by the native greater-than threshold at `level` with `value`. Updates the signal in place.
    /// `level` is the real-valued threshold parameter for the native complex operation and must be greater than zero. The wrapper forwards its value without checking that restriction.
    ///
    /// Prepares participating regions, waiting for producers on other streams, and completes native work before returning successfully.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if CUDA context binding or region validation fails, if waiting for a producer fails, if a length exceeds the native range, if NPP rejects the threshold parameters, or if stream synchronization fails.
    pub fn threshold_greater_value_complex_in_place<L>(
        stream_context: &StreamContext,
        signal: &mut SignalViewMut<'_, T>,
        level: L,
        value: T,
    ) -> Result<()>
    where
        L: Copy,
        Self: ComplexValueThresholdSignal<T, L>,
    {
        <Self as ComplexValueThresholdSignal<T, L>>::threshold_greater_value_complex_signal_in_place(
            stream_context,
            signal,
            level,
            value,
        )
    }
}
